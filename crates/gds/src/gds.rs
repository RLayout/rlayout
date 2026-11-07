use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use rlayout_core::{
    Array, Cell, CellId, Geometry, Instance, Label, LayerKey, Layout, Path, PathEnds, Point,
    Transformation, path_contour, validate_contour, validate_polygon,
};

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub offset: u64,
    pub cell: Option<String>,
    pub message: String,
}

#[derive(Debug)]
pub struct GdsError {
    pub offset: u64,
    pub cell: Option<String>,
    pub element: Option<&'static str>,
    pub message: String,
}

impl fmt::Display for GdsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "At byte {}", self.offset)?;
        if let Some(cell) = &self.cell {
            write!(formatter, ", cell {cell}")?;
        }
        if let Some(element) = self.element {
            write!(formatter, ", {element}")?;
        }
        write!(formatter, ": {}", self.message)
    }
}

impl Error for GdsError {}

#[derive(Debug)]
pub struct LoadResult {
    pub layout: Layout,
    pub warnings: Vec<Diagnostic>,
}

pub fn read(reader: impl Read) -> Result<LoadResult, GdsError> {
    Parser {
        reader,
        offset: 0,
        record_offset: 0,
        cell: None,
        element: None,
        warnings: Vec::new(),
        warned: BTreeSet::new(),
    }
    .read_library()
}

struct Record {
    kind: u8,
    datatype: u8,
    data: Vec<u8>,
}

struct Parser<R> {
    reader: R,
    offset: u64,
    record_offset: u64,
    cell: Option<String>,
    element: Option<&'static str>,
    warnings: Vec<Diagnostic>,
    warned: BTreeSet<u8>,
}

struct DraftCell {
    name: String,
    shapes: Vec<ShapeOrReference>,
}

enum ShapeOrReference {
    Shape(rlayout_core::Shape),
    Reference {
        name: String,
        transformation: Transformation,
        array: Array,
        offset: u64,
    },
}

#[derive(Default)]
struct Element {
    layer: Option<u16>,
    datatype: Option<u16>,
    points: Option<Vec<Point>>,
    width: Option<i32>,
    path_type: Option<u16>,
    beginning: Option<i32>,
    end: Option<i32>,
    name: Option<String>,
    text: Option<String>,
    presentation: u16,
    columns: u16,
    rows: u16,
    transformation: Transformation,
}

impl<R: Read> Parser<R> {
    fn error(&self, message: impl Into<String>) -> GdsError {
        GdsError {
            offset: self.record_offset,
            cell: self.cell.clone(),
            element: self.element,
            message: message.into(),
        }
    }

    fn warn(&mut self, kind: u8, message: impl Into<String>) {
        if self.warned.insert(kind) {
            self.warnings.push(Diagnostic {
                offset: self.record_offset,
                cell: self.cell.clone(),
                message: message.into(),
            });
        }
    }

    fn next(&mut self) -> Result<Record, GdsError> {
        self.record_offset = self.offset;
        let mut header = [0_u8; 4];
        self.reader
            .read_exact(&mut header)
            .map_err(|error| self.error(format!("Reading record header: {error}")))?;
        let length = u16::from_be_bytes([header[0], header[1]]);
        if length < 4 || length % 2 != 0 {
            return Err(self.error(format!("Invalid record length {length}")));
        }
        let mut data = vec![0; usize::from(length) - 4];
        self.reader
            .read_exact(&mut data)
            .map_err(|error| self.error(format!("Reading record payload: {error}")))?;
        self.offset += u64::from(length);
        Ok(Record {
            kind: header[2],
            datatype: header[3],
            data,
        })
    }

    fn required(
        &mut self,
        kind: u8,
        datatype: u8,
        length: Option<usize>,
    ) -> Result<Record, GdsError> {
        let record = self.next()?;
        if record.kind != kind {
            return Err(self.error(format!(
                "Expected record 0x{kind:02x}, found 0x{:02x}",
                record.kind
            )));
        }
        self.check(&record, datatype, length)?;
        Ok(record)
    }

    fn check(&self, record: &Record, datatype: u8, length: Option<usize>) -> Result<(), GdsError> {
        if record.datatype != datatype || length.is_some_and(|length| record.data.len() != length) {
            return Err(self.error(format!(
                "Invalid datatype or payload size for record 0x{:02x}",
                record.kind
            )));
        }
        Ok(())
    }

    fn short(&self, record: &Record, datatype: u8) -> Result<u16, GdsError> {
        self.check(record, datatype, Some(2))?;
        let bytes: [u8; 2] = record
            .data
            .as_slice()
            .try_into()
            .map_err(|_| self.error("Invalid short integer"))?;
        Ok(u16::from_be_bytes(bytes))
    }

    fn integer(&self, record: &Record) -> Result<i32, GdsError> {
        self.check(record, 3, Some(4))?;
        let bytes: [u8; 4] = record
            .data
            .as_slice()
            .try_into()
            .map_err(|_| self.error("Invalid integer"))?;
        Ok(i32::from_be_bytes(bytes))
    }

    fn real(&self, record: &Record) -> Result<f64, GdsError> {
        self.check(record, 5, Some(8))?;
        real8(&record.data).ok_or_else(|| self.error("Invalid GDS real number"))
    }

    fn string(&self, record: &Record) -> Result<String, GdsError> {
        self.check(record, 6, None)?;
        let bytes = record.data.strip_suffix(&[0]).unwrap_or(&record.data);
        if bytes.iter().any(|byte| !byte.is_ascii() || *byte == 0) {
            return Err(self.error("Non-ASCII or embedded-NUL strings are unsupported"));
        }
        String::from_utf8(bytes.to_vec()).map_err(|error| self.error(error.to_string()))
    }

    fn read_library(mut self) -> Result<LoadResult, GdsError> {
        let header = self.required(0x00, 2, Some(2))?;
        let version = self.short(&header, 2)?;
        if version > 600 {
            return Err(self.error(format!("Unsupported GDS version {version}")));
        }
        self.required(0x01, 2, Some(24))?;
        let mut name = None;
        let mut units = None;
        let mut drafts = Vec::new();
        let mut cell_names = HashMap::new();
        loop {
            let record = self.next()?;
            match record.kind {
                0x02 if name.is_none() && drafts.is_empty() => {
                    let library_name = self.string(&record)?;
                    if library_name.is_empty() {
                        return Err(self.error("Empty library name"));
                    }
                    name = Some(library_name);
                }
                0x03 if units.is_none() && name.is_some() && drafts.is_empty() => {
                    self.check(&record, 5, Some(16))?;
                    let user_units = real8(
                        record
                            .data
                            .get(..8)
                            .ok_or_else(|| self.error("Missing user units"))?,
                    )
                    .ok_or_else(|| self.error("Invalid user units"))?;
                    let meters = real8(
                        record
                            .data
                            .get(8..)
                            .ok_or_else(|| self.error("Missing database units"))?,
                    )
                    .ok_or_else(|| self.error("Invalid database units"))?;
                    if user_units <= 0.0
                        || meters <= 0.0
                        || !user_units.is_finite()
                        || !meters.is_finite()
                    {
                        return Err(
                            self.error("Database and user units must be finite and positive")
                        );
                    }
                    units = Some((user_units, meters));
                }
                0x05 if name.is_some() && units.is_some() => {
                    self.check(&record, 2, Some(24))?;
                    let draft = self.read_cell()?;
                    let id = CellId::from_index(drafts.len());
                    if cell_names.insert(draft.name.clone(), id).is_some() {
                        return Err(
                            self.error(format!("Duplicate cell definition: {}", draft.name))
                        );
                    }
                    drafts.push(draft);
                }
                0x04 => {
                    self.check(&record, 0, Some(0))?;
                    break;
                }
                0x1f | 0x20 | 0x39 => {
                    self.check(&record, 6, None)?;
                    self.warn(
                        record.kind,
                        format!("Ignored library metadata record 0x{:02x}", record.kind),
                    );
                }
                0x22 => {
                    self.short(&record, 2)?;
                    self.warn(record.kind, "Ignored library generation metadata");
                }
                _ => {
                    return Err(self.error(format!(
                        "Unexpected or unsupported library record 0x{:02x}",
                        record.kind
                    )));
                }
            }
        }
        let name = name.ok_or_else(|| self.error("Missing library name"))?;
        let (user_units, meters) = units.ok_or_else(|| self.error("Missing database units"))?;
        let mut cells = Vec::with_capacity(drafts.len());
        for draft in drafts {
            let mut shapes = Vec::new();
            let mut instances = Vec::new();
            for entry in draft.shapes {
                match entry {
                    ShapeOrReference::Shape(shape) => shapes.push(shape),
                    ShapeOrReference::Reference {
                        name,
                        transformation,
                        array,
                        offset,
                    } => {
                        let cell = cell_names.get(&name).copied().ok_or_else(|| GdsError {
                            offset,
                            cell: Some(draft.name.clone()),
                            element: Some("reference"),
                            message: format!("Unresolved reference to {name}"),
                        })?;
                        instances.push(Instance {
                            cell,
                            transformation,
                            array,
                        });
                    }
                }
            }
            cells.push(Cell::new(draft.name, shapes, instances));
        }
        let layout = Layout::new(name, meters, user_units, cells)
            .map_err(|error| self.error(error.to_string()))?;
        let mut trailing = [0_u8; 4096];
        loop {
            let length = match self.reader.read(&mut trailing) {
                Ok(length) => length,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(self.error(format!("Reading trailing data: {error}"))),
            };
            if length == 0 {
                break;
            }
            if trailing
                .get(..length)
                .is_some_and(|data| data.iter().any(|byte| *byte != 0))
            {
                return Err(self.error("Nonzero data after ENDLIB is unsupported"));
            }
        }
        Ok(LoadResult {
            layout,
            warnings: self.warnings,
        })
    }

    fn read_cell(&mut self) -> Result<DraftCell, GdsError> {
        let name_record = self.required(0x06, 6, None)?;
        let name = self.string(&name_record)?;
        if name.is_empty() {
            return Err(self.error("Empty cell name"));
        }
        self.cell = Some(name.clone());
        let mut shapes = Vec::new();
        loop {
            let record = self.next()?;
            match record.kind {
                0x07 => {
                    self.check(&record, 0, Some(0))?;
                    break;
                }
                0x08 | 0x09 | 0x0a | 0x0b | 0x0c | 0x2d => {
                    self.check(&record, 0, Some(0))?;
                    if let Some(element) = self.read_element(record.kind)? {
                        shapes.push(element);
                    }
                }
                _ => {
                    return Err(self.error(format!(
                        "Unexpected or unsupported cell record 0x{:02x}",
                        record.kind
                    )));
                }
            }
        }
        self.cell = None;
        Ok(DraftCell { name, shapes })
    }

    fn read_element(&mut self, kind: u8) -> Result<Option<ShapeOrReference>, GdsError> {
        let offset = self.record_offset;
        self.element = Some(match kind {
            0x08 => "BOUNDARY",
            0x09 => "PATH",
            0x0a => "SREF",
            0x0b => "AREF",
            0x0c => "TEXT",
            _ => "BOX",
        });
        let mut element = Element {
            columns: 1,
            rows: 1,
            ..Element::default()
        };
        let mut seen = BTreeSet::new();
        let mut property_pending = false;
        loop {
            let record = self.next()?;
            if record.kind == 0x11 {
                self.check(&record, 0, Some(0))?;
                if property_pending {
                    return Err(self.error("PROPATTR lacks a PROPVALUE"));
                }
                break;
            }
            if !matches!(record.kind, 0x2b | 0x2c) && !seen.insert(record.kind) {
                return Err(self.error(format!(
                    "Duplicate record 0x{:02x}; multiple XY records are unsupported",
                    record.kind
                )));
            }
            let allowed = match record.kind {
                0x0d => matches!(kind, 0x08 | 0x09 | 0x0c | 0x2d),
                0x0e => matches!(kind, 0x08 | 0x09),
                0x0f | 0x21 => matches!(kind, 0x09 | 0x0c),
                0x10 | 0x26 | 0x2b | 0x2c | 0x2f => true,
                0x12 => matches!(kind, 0x0a | 0x0b),
                0x13 => kind == 0x0b,
                0x16 | 0x17 | 0x19 => kind == 0x0c,
                0x1a..=0x1c => matches!(kind, 0x0a..=0x0c),
                0x2e => kind == 0x2d,
                0x30 | 0x31 => kind == 0x09,
                _ => false,
            };
            if !allowed {
                return Err(self.error(format!(
                    "Unexpected or unsupported element record 0x{:02x}",
                    record.kind
                )));
            }
            match record.kind {
                0x0d => element.layer = Some(self.short(&record, 2)?),
                0x0e | 0x16 | 0x2e => element.datatype = Some(self.short(&record, 2)?),
                0x0f => element.width = Some(self.integer(&record)?),
                0x10 => {
                    self.check(&record, 3, None)?;
                    if record.data.len() % 8 != 0 {
                        return Err(self.error("XY payload must contain coordinate pairs"));
                    }
                    let points = record
                        .data
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|bytes| {
                            let x: [u8; 4] = bytes
                                .get(..4)
                                .ok_or_else(|| self.error("Missing X coordinate"))?
                                .try_into()
                                .map_err(|_| self.error("Invalid X coordinate"))?;
                            let y: [u8; 4] = bytes
                                .get(4..)
                                .ok_or_else(|| self.error("Missing Y coordinate"))?
                                .try_into()
                                .map_err(|_| self.error("Invalid Y coordinate"))?;
                            Ok(Point::new(
                                i64::from(i32::from_be_bytes(x)),
                                i64::from(i32::from_be_bytes(y)),
                            ))
                        })
                        .collect::<Result<Vec<_>, GdsError>>()?;
                    element.points = Some(points);
                }
                0x12 => element.name = Some(self.string(&record)?),
                0x13 => {
                    self.check(&record, 2, Some(4))?;
                    let columns: [u8; 2] = record
                        .data
                        .get(..2)
                        .ok_or_else(|| self.error("Missing columns"))?
                        .try_into()
                        .map_err(|_| self.error("Invalid columns"))?;
                    let rows: [u8; 2] = record
                        .data
                        .get(2..)
                        .ok_or_else(|| self.error("Missing rows"))?
                        .try_into()
                        .map_err(|_| self.error("Invalid rows"))?;
                    element.columns = u16::from_be_bytes(columns);
                    element.rows = u16::from_be_bytes(rows);
                    if element.columns == 0
                        || element.rows == 0
                        || element.columns > i16::MAX as u16
                        || element.rows > i16::MAX as u16
                    {
                        return Err(
                            self.error("Array counts must be positive signed 16-bit integers")
                        );
                    }
                }
                0x17 => {
                    element.presentation = self.short(&record, 1)?;
                    if element.presentation & !0x003f != 0 {
                        return Err(self.error("Reserved TEXT presentation bits are unsupported"));
                    }
                    if element.presentation & 0x30 != 0 {
                        self.warn(
                            0x17,
                            "GDS font selection is approximated using the egui font",
                        );
                    }
                }
                0x19 => element.text = Some(self.string(&record)?),
                0x1a => {
                    let flags = self.short(&record, 1)?;
                    if flags & !0x8000 != 0 {
                        return Err(self.error("Absolute or reserved STRANS flags are unsupported"));
                    }
                    element.transformation.reflected = flags & 0x8000 != 0;
                }
                0x1b => element.transformation.magnification = self.real(&record)?,
                0x1c => element.transformation.angle = self.real(&record)?.to_radians(),
                0x21 => element.path_type = Some(self.short(&record, 2)?),
                0x26 => {
                    self.short(&record, 1)?;
                    self.warn(0x26, "Ignored ELFLAGS metadata");
                }
                0x2b => {
                    self.short(&record, 2)?;
                    if property_pending {
                        return Err(self.error("PROPATTR lacks a PROPVALUE"));
                    }
                    property_pending = true;
                    self.warn(0x2b, "Element properties are not retained by this viewer");
                }
                0x2c => {
                    self.string(&record)?;
                    if !property_pending {
                        return Err(self.error("PROPVALUE lacks a PROPATTR"));
                    }
                    property_pending = false;
                }
                0x2f => {
                    self.integer(&record)?;
                    self.warn(0x2f, "Ignored PLEX grouping metadata");
                }
                0x30 => element.beginning = Some(self.integer(&record)?),
                0x31 => element.end = Some(self.integer(&record)?),
                _ => return Err(self.error("Unsupported element record")),
            }
        }
        let points = element
            .points
            .take()
            .ok_or_else(|| self.error("Missing XY record"))?;
        let result = if matches!(kind, 0x0a | 0x0b) {
            if (kind == 0x0a && points.len() != 1) || (kind == 0x0b && points.len() != 3) {
                return Err(self.error("Reference has an invalid number of XY points"));
            }
            if kind == 0x0b && !seen.contains(&0x13) {
                return Err(self.error("Missing COLROW record"));
            }
            let origin = points
                .first()
                .ok_or_else(|| self.error("Missing reference origin"))?
                .to_double();
            element.transformation.translation = origin;
            if !element.transformation.valid() {
                return Err(self.error("Invalid reference magnification or transformation"));
            }
            let array = if kind == 0x0b {
                Array {
                    columns: element.columns,
                    rows: element.rows,
                    column_step: points
                        .get(1)
                        .ok_or_else(|| self.error("Missing array column endpoint"))?
                        .to_double()
                        .minus(origin)
                        .times(1.0 / f64::from(element.columns)),
                    row_step: points
                        .get(2)
                        .ok_or_else(|| self.error("Missing array row endpoint"))?
                        .to_double()
                        .minus(origin)
                        .times(1.0 / f64::from(element.rows)),
                }
            } else {
                Array::default()
            };
            ShapeOrReference::Reference {
                name: element
                    .name
                    .ok_or_else(|| self.error("Missing SNAME record"))?,
                transformation: element.transformation,
                array,
                offset,
            }
        } else {
            let layer = LayerKey {
                layer: element
                    .layer
                    .ok_or_else(|| self.error("Missing LAYER record"))?,
                datatype: element
                    .datatype
                    .ok_or_else(|| self.error("Missing datatype record"))?,
            };
            let geometry = match kind {
                0x08 => {
                    let mut points = points;
                    if points.len() > 1 && points.first() == points.last() {
                        points.pop();
                    }
                    if points.len() < 3 {
                        self.warnings.push(Diagnostic {
                            offset,
                            cell: self.cell.clone(),
                            message: "BOUNDARY with fewer than three vertices ignored".into(),
                        });
                        self.element = None;
                        return Ok(None);
                    }
                    validate_polygon(&points).map_err(|error| self.error(error.to_string()))?;
                    Geometry::Polygon(points)
                }
                0x2d => {
                    if points.len() != 5 || points.first() != points.last() {
                        return Err(self.error("BOX must contain five closed XY points"));
                    }
                    let minimum = Point::new(
                        points
                            .iter()
                            .map(|point| point.x)
                            .min()
                            .ok_or_else(|| self.error("Empty BOX"))?,
                        points
                            .iter()
                            .map(|point| point.y)
                            .min()
                            .ok_or_else(|| self.error("Empty BOX"))?,
                    );
                    let maximum = Point::new(
                        points
                            .iter()
                            .map(|point| point.x)
                            .max()
                            .ok_or_else(|| self.error("Empty BOX"))?,
                        points
                            .iter()
                            .map(|point| point.y)
                            .max()
                            .ok_or_else(|| self.error("Empty BOX"))?,
                    );
                    let unique: BTreeSet<_> = points
                        .iter()
                        .take(4)
                        .map(|point| (point.x, point.y))
                        .collect();
                    if minimum.x == maximum.x
                        || minimum.y == maximum.y
                        || unique.len() != 4
                        || points.iter().any(|point| {
                            (point.x != minimum.x && point.x != maximum.x)
                                || (point.y != minimum.y && point.y != maximum.y)
                        })
                        || points
                            .windows(2)
                            .any(|pair| match (pair.first(), pair.get(1)) {
                                (Some(first), Some(second)) => {
                                    first.x != second.x && first.y != second.y
                                }
                                _ => true,
                            })
                    {
                        return Err(self.error("BOX must be an axis-aligned rectangle"));
                    }
                    Geometry::Box { minimum, maximum }
                }
                0x09 => {
                    let path_type = element.path_type.unwrap_or(0);
                    if path_type != 4 && (element.beginning.is_some() || element.end.is_some()) {
                        return Err(self.error("Explicit extensions require PATHTYPE 4"));
                    }
                    let ends = match path_type {
                        0 => PathEnds::Flush,
                        1 => PathEnds::Round,
                        2 => PathEnds::Square,
                        4 => PathEnds::Extended {
                            beginning: i64::from(element.beginning.unwrap_or(0)),
                            end: i64::from(element.end.unwrap_or(0)),
                        },
                        _ => return Err(self.error(format!("Unsupported PATHTYPE {path_type}"))),
                    };
                    let width = element.width.unwrap_or(0);
                    let path = Path {
                        points,
                        width: u64::from(width.unsigned_abs()),
                        absolute_width: width < 0,
                        ends,
                    };
                    if path.points.is_empty()
                        || (path.points.len() < 2 && path.ends != PathEnds::Round)
                        || path
                            .points
                            .windows(2)
                            .any(|pair| pair.first() == pair.get(1))
                    {
                        return Err(self.error("PATH has too few or repeated points"));
                    }
                    let contour =
                        path_contour(&path, 1.0).map_err(|error| self.error(error.to_string()))?;
                    if !contour.is_empty() {
                        validate_contour(&contour)
                            .map_err(|error| self.error(error.to_string()))?;
                    }
                    Geometry::Path(path)
                }
                0x0c => {
                    if points.len() != 1 {
                        return Err(self.error("TEXT must contain one XY point"));
                    }
                    let horizontal_alignment = (element.presentation & 3) as u8;
                    let vertical_alignment = ((element.presentation >> 2) & 3) as u8;
                    if horizontal_alignment > 2 || vertical_alignment > 2 {
                        return Err(self.error("Invalid TEXT alignment"));
                    }
                    element.transformation.translation = points
                        .first()
                        .ok_or_else(|| self.error("Missing text location"))?
                        .to_double();
                    if !element.transformation.valid() {
                        return Err(self.error("Invalid text transformation"));
                    }
                    if element.width.is_some() || element.path_type.is_some() {
                        self.warn(
                            0x0c,
                            "TEXT stroke width and path style are approximated using the egui font",
                        );
                    }
                    Geometry::Label(Label {
                        text: element
                            .text
                            .ok_or_else(|| self.error("Missing STRING record"))?,
                        transformation: element.transformation,
                        horizontal_alignment,
                        vertical_alignment,
                    })
                }
                _ => return Err(self.error("Unsupported geometry")),
            };
            ShapeOrReference::Shape(rlayout_core::Shape { layer, geometry })
        };
        self.element = None;
        Ok(Some(result))
    }
}

fn real8(bytes: &[u8]) -> Option<f64> {
    if bytes.len() != 8 {
        return None;
    }
    if bytes.iter().all(|byte| *byte == 0) {
        return Some(0.0);
    }
    let first = *bytes.first()?;
    let mantissa = bytes
        .get(1..)?
        .iter()
        .fold(0_u64, |value, byte| (value << 8) | u64::from(*byte));
    let fraction = mantissa as f64 / 72_057_594_037_927_936.0;
    let exponent = i32::from(first & 0x7f) - 64;
    let value = fraction * 16.0_f64.powi(exponent) * if first & 0x80 != 0 { -1.0 } else { 1.0 };
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests;
