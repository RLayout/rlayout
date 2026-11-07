use std::io;

pub struct Fixture {
    pub bytes: Vec<u8>,
}

impl Fixture {
    pub fn new() -> io::Result<Self> {
        let mut fixture = Self { bytes: Vec::new() };
        fixture.record(0x00, 2, &600_u16.to_be_bytes())?;
        fixture.record(0x01, 2, &[0; 24])?;
        fixture.string(0x02, "RLAYOUT_TEST")?;
        let mut units = encode_real(0.001)?.to_vec();
        units.extend(encode_real(1e-9)?);
        fixture.record(0x03, 5, &units)?;
        Ok(fixture)
    }

    pub fn record(&mut self, kind: u8, datatype: u8, data: &[u8]) -> io::Result<()> {
        let length = u16::try_from(data.len() + 4)
            .map_err(|_| io::Error::other("Fixture record is too large"))?;
        if length % 2 != 0 {
            return Err(io::Error::other("Fixture record length must be even"));
        }
        self.bytes.extend(length.to_be_bytes());
        self.bytes.extend([kind, datatype]);
        self.bytes.extend(data);
        Ok(())
    }

    pub fn string(&mut self, kind: u8, text: &str) -> io::Result<()> {
        let mut bytes = text.as_bytes().to_vec();
        if !bytes.len().is_multiple_of(2) {
            bytes.push(0);
        }
        self.record(kind, 6, &bytes)
    }

    pub fn cell(&mut self, name: &str) -> io::Result<()> {
        self.record(0x05, 2, &[0; 24])?;
        self.string(0x06, name)
    }

    pub fn end_cell(&mut self) -> io::Result<()> {
        self.record(0x07, 0, &[])
    }
    pub fn end(&mut self) -> io::Result<()> {
        self.record(0x04, 0, &[])
    }
    pub fn short(&mut self, kind: u8, value: u16) -> io::Result<()> {
        self.record(kind, 2, &value.to_be_bytes())
    }
    pub fn integer(&mut self, kind: u8, value: i32) -> io::Result<()> {
        self.record(kind, 3, &value.to_be_bytes())
    }
    pub fn real(&mut self, kind: u8, value: f64) -> io::Result<()> {
        self.record(kind, 5, &encode_real(value)?)
    }

    pub fn points(&mut self, points: &[(i32, i32)]) -> io::Result<()> {
        let bytes: Vec<_> = points
            .iter()
            .flat_map(|(x, y)| x.to_be_bytes().into_iter().chain(y.to_be_bytes()))
            .collect();
        self.record(0x10, 3, &bytes)
    }

    pub fn polygon(&mut self, layer: u16, points: &[(i32, i32)]) -> io::Result<()> {
        self.record(0x08, 0, &[])?;
        self.short(0x0d, layer)?;
        self.short(0x0e, 0)?;
        self.points(points)?;
        self.record(0x11, 0, &[])
    }

    pub fn path(
        &mut self,
        layer: u16,
        style: u16,
        width: i32,
        points: &[(i32, i32)],
    ) -> io::Result<()> {
        self.record(0x09, 0, &[])?;
        self.short(0x0d, layer)?;
        self.short(0x0e, 1)?;
        self.short(0x21, style)?;
        self.integer(0x0f, width)?;
        if style == 4 {
            self.integer(0x30, 20)?;
            self.integer(0x31, 40)?;
        }
        self.points(points)?;
        self.record(0x11, 0, &[])
    }

    pub fn reference(
        &mut self,
        name: &str,
        origin: (i32, i32),
        reflected: bool,
        angle: f64,
        magnification: f64,
    ) -> io::Result<()> {
        self.record(0x0a, 0, &[])?;
        self.string(0x12, name)?;
        self.record(
            0x1a,
            1,
            &(if reflected { 0x8000_u16 } else { 0_u16 }).to_be_bytes(),
        )?;
        self.real(0x1b, magnification)?;
        self.real(0x1c, angle)?;
        self.points(&[origin])?;
        self.record(0x11, 0, &[])
    }

    pub fn array(
        &mut self,
        name: &str,
        columns: u16,
        rows: u16,
        points: &[(i32, i32)],
    ) -> io::Result<()> {
        self.record(0x0b, 0, &[])?;
        self.string(0x12, name)?;
        let mut counts = columns.to_be_bytes().to_vec();
        counts.extend(rows.to_be_bytes());
        self.record(0x13, 2, &counts)?;
        self.points(points)?;
        self.record(0x11, 0, &[])
    }

    pub fn label(&mut self, text: &str, origin: (i32, i32)) -> io::Result<()> {
        self.record(0x0c, 0, &[])?;
        self.short(0x0d, 10)?;
        self.short(0x16, 0)?;
        self.record(0x17, 1, &5_u16.to_be_bytes())?;
        self.points(&[origin])?;
        self.string(0x19, text)?;
        self.record(0x11, 0, &[])
    }
}

pub fn encode_real(value: f64) -> io::Result<[u8; 8]> {
    if !value.is_finite() {
        return Err(io::Error::other("Non-finite fixture real"));
    }
    if value == 0.0 {
        return Ok([0; 8]);
    }
    let mut fraction = value.abs();
    let mut exponent = 64_i32;
    while fraction >= 1.0 {
        fraction /= 16.0;
        exponent += 1;
    }
    while fraction < 0.0625 {
        fraction *= 16.0;
        exponent -= 1;
    }
    if !(0..=127).contains(&exponent) {
        return Err(io::Error::other("Fixture real exponent overflow"));
    }
    let mantissa = (fraction * 72_057_594_037_927_936.0) as u64;
    let mut bytes = mantissa.to_be_bytes();
    if let Some(first) = bytes.first_mut() {
        *first = exponent as u8 | if value < 0.0 { 0x80 } else { 0 };
    }
    Ok(bytes)
}

pub fn demo() -> io::Result<Vec<u8>> {
    let mut fixture = Fixture::new()?;
    fixture.cell("TOP")?;
    fixture.reference("PART", (0, 0), false, 0.0, 1.0)?;
    fixture.reference("PART", (1600, 100), true, 30.0, 0.75)?;
    fixture.array("PART", 3, 2, &[(0, 1100), (2700, 1400), (200, 2900)])?;
    fixture.label("RLayout test", (800, 900))?;
    fixture.end_cell()?;
    fixture.cell("PART")?;
    fixture.polygon(
        1,
        &[
            (0, 0),
            (500, 0),
            (500, 150),
            (150, 150),
            (150, 500),
            (0, 500),
            (0, 0),
        ],
    )?;
    for style in 0..=2 {
        fixture.path(
            style + 2,
            style,
            70,
            &[
                (220, 250 + i32::from(style) * 100),
                (500, 250 + i32::from(style) * 100),
            ],
        )?;
    }
    fixture.path(5, 4, 50, &[(200, 600), (500, 650)])?;
    fixture.label("PART", (260, 180))?;
    fixture.end_cell()?;
    fixture.end()?;
    Ok(fixture.bytes)
}

pub fn large() -> io::Result<Vec<u8>> {
    let mut fixture = Fixture::new()?;
    fixture.cell("TOP")?;
    for index in 0..100_000_i32 {
        let x = (index % 500) * 12;
        let y = (index / 500) * 12;
        fixture.polygon(
            (index % 6 + 1) as u16,
            &[(x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10), (x, y)],
        )?;
    }
    fixture.end_cell()?;
    fixture.end()?;
    Ok(fixture.bytes)
}

#[cfg(not(test))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let mode = arguments.first().map(String::as_str).unwrap_or("demo");
    let path = arguments
        .get(1)
        .ok_or("Usage: fixture demo|large OUTPUT.gds")?;
    let bytes = match mode {
        "demo" => demo()?,
        "large" => large()?,
        _ => return Err("Choose demo or large".into()),
    };
    std::fs::write(path, &bytes)?;
    let start = std::time::Instant::now();
    let result = rlayout_gds::read(bytes.as_slice())?;
    println!(
        "{}: {} shapes, parsed in {:.2} ms",
        path,
        result.layout.shape_count(),
        start.elapsed().as_secs_f64() * 1000.0
    );
    Ok(())
}
