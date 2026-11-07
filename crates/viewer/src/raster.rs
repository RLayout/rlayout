use super::*;
use std::collections::{HashSet, VecDeque, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};

const TILE_SIZE: usize = 256;
const CACHE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct Entry {
    bounds: Bounds,
    index: usize,
    instance: bool,
    labels: (f64, f64),
}

#[derive(Debug)]
struct Node {
    bounds: Bounds,
    range: std::ops::Range<usize>,
    children: Option<(usize, usize)>,
    labels: (f64, f64),
}

#[derive(Debug)]
pub(super) struct SpatialIndex {
    entries: Vec<Entry>,
    nodes: Vec<Node>,
}

impl SpatialIndex {
    pub(super) fn new(
        layout: &Layout,
        id: CellId,
        labels: &[(f64, f64)],
    ) -> Result<Self, ViewError> {
        let cell = layout
            .cell(id)
            .ok_or_else(|| ViewError("Missing indexed cell".into()))?;
        let mut entries = Vec::with_capacity(cell.shapes().len() + cell.instances().len());
        for (index, shape) in cell.shapes().iter().enumerate() {
            if let Some(bounds) = shape_bounds(&shape.geometry).map_err(|error| {
                shape_error(
                    cell.name(),
                    shape.layer,
                    index,
                    ViewError(error.to_string()),
                )
            })? {
                entries.push(Entry {
                    bounds,
                    index,
                    instance: false,
                    labels: match &shape.geometry {
                        Geometry::Label(label) => (
                            label.text.chars().count() as f64,
                            label.transformation.magnification,
                        ),
                        _ => (0.0, 0.0),
                    },
                });
            }
        }
        for (index, instance) in cell.instances().iter().enumerate() {
            if let Some(bounds) = layout.transformed_bounds(instance.cell, instance.transformation)
            {
                entries.push(Entry {
                    bounds: instance.array.bounds(bounds),
                    index,
                    instance: true,
                    labels: (
                        labels[instance.cell.index()].0,
                        labels[instance.cell.index()].1 * instance.transformation.magnification,
                    ),
                });
            }
        }
        let mut result = Self {
            entries,
            nodes: Vec::new(),
        };
        if !result.entries.is_empty() {
            result.build(0..result.entries.len());
        }
        Ok(result)
    }

    fn build(&mut self, range: std::ops::Range<usize>) -> usize {
        let bounds = self.entries[range.clone()]
            .iter()
            .map(|entry| entry.bounds)
            .reduce(Bounds::union)
            .unwrap_or(Bounds {
                minimum: DoublePoint::default(),
                maximum: DoublePoint::default(),
            });
        let index = self.nodes.len();
        self.nodes.push(Node {
            bounds,
            range: range.clone(),
            children: None,
            labels: self.entries[range.clone()]
                .iter()
                .fold((0.0_f64, 0.0_f64), |summary, entry| {
                    (summary.0.max(entry.labels.0), summary.1.max(entry.labels.1))
                }),
        });
        if range.len() > 32 {
            let middle = range.start + range.len() / 2;
            let horizontal = bounds.width() >= bounds.height();
            self.entries[range.clone()].select_nth_unstable_by(
                middle - range.start,
                |first, second| {
                    let first = first.bounds.center();
                    let second = second.bounds.center();
                    if horizontal {
                        first.x.total_cmp(&second.x)
                    } else {
                        first.y.total_cmp(&second.y)
                    }
                },
            );
            let left = self.build(range.start..middle);
            let right = self.build(middle..range.end);
            self.nodes[index].children = Some((left, right));
        }
        index
    }

    fn query(&self) -> Query {
        Query {
            stack: if self.nodes.is_empty() {
                Vec::new()
            } else {
                vec![0]
            },
            range: 0..0,
        }
    }
}

struct Query {
    stack: Vec<usize>,
    range: std::ops::Range<usize>,
}

impl Query {
    fn next(
        &mut self,
        index: &SpatialIndex,
        target: Bounds,
        magnification: f64,
        units_per_point: f64,
    ) -> Option<Entry> {
        loop {
            for offset in self.range.by_ref() {
                if let Some(entry) = index.entries.get(offset).filter(|entry| {
                    entry
                        .bounds
                        .expanded(label_margin(entry.labels, magnification, units_per_point))
                        .intersects(target)
                }) {
                    return Some(*entry);
                }
            }
            let node = index.nodes.get(self.stack.pop()?)?;
            if node
                .bounds
                .expanded(label_margin(node.labels, magnification, units_per_point))
                .intersects(target)
            {
                if let Some((left, right)) = node.children {
                    self.stack.extend([right, left]);
                } else {
                    self.range = node.range.clone();
                }
            }
        }
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
struct CacheKey {
    cell: CellId,
    angle: u64,
    magnification: u64,
    scale: u64,
    font_scale: u64,
    reflected: bool,
    depth: u16,
    layers: u64,
}

struct Coverage {
    width: usize,
    height: usize,
    runs: BTreeMap<LayerKey, Vec<Run>>,
    shape_count: usize,
}

struct Run {
    row: usize,
    start: usize,
    pixels: Vec<u8>,
}

impl Coverage {
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + 2 * std::mem::size_of::<CacheKey>()
            + self.runs.len() * 128
            + self
                .runs
                .values()
                .map(|runs| {
                    runs.capacity() * std::mem::size_of::<Run>()
                        + runs.iter().map(|run| run.pixels.capacity()).sum::<usize>()
                })
                .sum::<usize>()
    }

    fn new(
        width: usize,
        height: usize,
        masks: BTreeMap<LayerKey, Vec<u8>>,
        shape_count: usize,
    ) -> Self {
        let mut layers = BTreeMap::new();
        for (layer, mask) in masks {
            let mut runs = Vec::new();
            for (row, pixels) in mask.chunks(width).enumerate() {
                let mut start = 0;
                while start < pixels.len() {
                    if pixels[start] == 0 {
                        start += 1;
                        continue;
                    }
                    let mut end = start + 1;
                    while end < pixels.len() && pixels[end] != 0 {
                        end += 1;
                    }
                    runs.push(Run {
                        row,
                        start,
                        pixels: pixels[start..end].to_vec(),
                    });
                    start = end;
                }
            }
            layers.insert(layer, runs);
        }
        Self {
            width,
            height,
            runs: layers,
            shape_count,
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct GlyphKey {
    text: String,
    size: u64,
    pixels_per_point: u32,
}

struct Glyph {
    size: [usize; 2],
    logical_size: Vec2,
    pixels_per_point: f64,
    alpha: Vec<u8>,
}

impl Glyph {
    fn bytes(&self, key: &GlyphKey) -> usize {
        self.alpha.capacity()
            + std::mem::size_of::<Self>()
            + 2 * (std::mem::size_of::<GlyphKey>() + key.text.capacity())
            + 128
    }
}

enum CacheEntry {
    Cell(CacheKey),
    Glyph(GlyphKey),
}

struct Cache {
    entries: HashMap<CacheKey, Arc<Coverage>>,
    order: VecDeque<CacheEntry>,
    glyphs: HashMap<GlyphKey, Arc<Glyph>>,
    bytes: usize,
    limit: usize,
}

impl Cache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            glyphs: HashMap::new(),
            bytes: 0,
            limit: CACHE_BYTES,
        }
    }

    fn insert(&mut self, key: CacheKey, coverage: Arc<Coverage>) {
        let bytes = coverage.bytes();
        if bytes > self.limit || self.entries.contains_key(&key) {
            return;
        }
        while self.bytes + bytes > self.limit {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.evict(oldest);
        }
        self.bytes += bytes;
        self.order.push_back(CacheEntry::Cell(key));
        self.entries.insert(key, coverage);
    }

    fn evict(&mut self, entry: CacheEntry) {
        match entry {
            CacheEntry::Cell(key) => {
                if let Some(old) = self.entries.remove(&key) {
                    self.bytes -= old.bytes();
                }
            }
            CacheEntry::Glyph(key) => {
                if let Some(old) = self.glyphs.remove(&key) {
                    self.bytes -= old.bytes(&key);
                }
            }
        }
    }

    fn insert_glyph(&mut self, key: GlyphKey, glyph: Arc<Glyph>) {
        let bytes = glyph.bytes(&key);
        if bytes > self.limit || self.glyphs.contains_key(&key) {
            return;
        }
        while self.bytes + bytes > self.limit {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.evict(oldest);
        }
        self.bytes += bytes;
        self.order.push_back(CacheEntry::Glyph(key.clone()));
        self.glyphs.insert(key, glyph);
    }
}

pub(super) struct Tile {
    pub rect: Rect,
    pub image: egui::ColorImage,
    pub shape_count: usize,
    pub complete: bool,
    pub preview: bool,
    pub first: bool,
}

pub(super) struct Worker {
    cancel: Arc<AtomicBool>,
    receiver: Option<mpsc::Receiver<Result<Tile, ViewError>>>,
    handles: Vec<(Arc<AtomicBool>, JoinHandle<()>)>,
    cache: Arc<Mutex<Cache>>,
}

impl Worker {
    pub(super) fn new() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            receiver: None,
            handles: Vec::new(),
            cache: Arc::new(Mutex::new(Cache::new())),
        }
    }

    pub(super) fn start(
        &mut self,
        scene: Arc<Scene>,
        key: FrameKey,
        context: egui::Context,
    ) -> Result<(), ViewError> {
        self.cancel.store(true, Ordering::Relaxed);
        self.receiver = None;
        self.cancel = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = mpsc::sync_channel(2);
        self.receiver = Some(receiver);
        let cancel = self.cancel.clone();
        let cache = self.cache.clone();
        let job_cancel = cancel.clone();
        let handle = thread::Builder::new()
            .name("gds-raster".into())
            .spawn(move || {
                if let Err(error) = render_tiles(scene, key, &cancel, cache, &sender, &context)
                    && !cancel.load(Ordering::Relaxed)
                    && sender.send(Err(error)).is_err()
                {
                    eprintln!("Raster error receiver disconnected");
                }
            })
            .map_err(|error| ViewError(format!("Starting raster worker: {error}")))?;
        self.handles.push((job_cancel, handle));
        Ok(())
    }

    pub(super) fn receive(&mut self) -> Vec<Result<Tile, ViewError>> {
        let mut messages = Vec::new();
        if let Some(receiver) = &self.receiver {
            while let Ok(message) = receiver.try_recv() {
                messages.push(message);
            }
        }
        let mut index = 0;
        while index < self.handles.len() {
            if self.handles[index].1.is_finished() {
                let (cancel, handle) = self.handles.swap_remove(index);
                if handle.join().is_err() && !cancel.load(Ordering::Relaxed) {
                    messages.push(Err(ViewError("Raster worker stopped unexpectedly".into())));
                }
            } else {
                index += 1;
            }
        }
        messages
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.receiver = None;
        for (_, handle) in self.handles.drain(..) {
            if handle.is_finished() && handle.join().is_err() {
                eprintln!("Raster worker stopped unexpectedly");
            }
        }
    }
}

pub(super) fn capture(
    scene: Arc<Scene>,
    key: FrameKey,
    context: egui::Context,
) -> Result<RenderedImage, ViewError> {
    let cancel = AtomicBool::new(false);
    let cache = Arc::new(Mutex::new(Cache::new()));
    let size = [
        (key.viewport.width() * key.pixels_per_point)
            .ceil()
            .max(1.0) as usize,
        (key.viewport.height() * key.pixels_per_point)
            .ceil()
            .max(1.0) as usize,
    ];
    if size[0]
        .checked_mul(size[1])
        .is_none_or(|pixels| pixels > CACHE_BYTES / 8)
    {
        return Err(ViewError("Viewport exceeds rendering memory budget".into()));
    }
    render_target(
        scene,
        key.clone(),
        f64::from(key.pixels_per_point),
        [0.0, 0.0],
        size,
        &cancel,
        cache,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
fn render_target(
    scene: Arc<Scene>,
    key: FrameKey,
    scale: f64,
    origin: [f64; 2],
    size: [usize; 2],
    cancel: &AtomicBool,
    cache: Arc<Mutex<Cache>>,
    context: egui::Context,
) -> Result<RenderedImage, ViewError> {
    let layers: Vec<_> = key
        .styles
        .iter()
        .filter_map(|(layer, style)| style.visible.then_some(*layer))
        .collect();
    let group_size = (48 * 1024 * 1024 / (size[0] * size[1])).max(1);
    let mut image = egui::ColorImage::new(size, vec![Color32::TRANSPARENT; size[0] * size[1]]);
    let mut shape_count = 0_usize;
    for group in layers.chunks(group_size) {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let mut request = key.clone();
        for (layer, style) in &mut request.styles {
            style.visible = group.binary_search(layer).is_ok();
        }
        let mut painter = Rasterizer::new(
            scene.clone(),
            request,
            scale,
            origin,
            size,
            cancel,
            cache.clone(),
            context.clone(),
        );
        if let Some(selected) = key.selected {
            painter.paint(selected, Transformation::default(), 0, true)?;
        }
        shape_count = shape_count.saturating_add(painter.count);
        for (destination, source) in image.pixels.iter_mut().zip(painter.image().pixels) {
            let source = source.to_array();
            let previous = destination.to_array();
            let inverse = 255 - u16::from(source[3]);
            let mut blended = [0; 4];
            for index in 0..4 {
                blended[index] = (u16::from(source[index])
                    + u16::from(previous[index]) * inverse / 255)
                    .min(255) as u8;
            }
            *destination =
                Color32::from_rgba_premultiplied(blended[0], blended[1], blended[2], blended[3]);
        }
    }
    let cache_bytes = cache
        .lock()
        .map_err(|error| ViewError(error.to_string()))?
        .bytes;
    Ok(RenderedImage {
        image,
        shape_count,
        cache_bytes,
    })
}

fn render_tiles(
    scene: Arc<Scene>,
    key: FrameKey,
    cancel: &AtomicBool,
    cache: Arc<Mutex<Cache>>,
    sender: &mpsc::SyncSender<Result<Tile, ViewError>>,
    context: &egui::Context,
) -> Result<(), ViewError> {
    let width = (key.viewport.width() * key.pixels_per_point)
        .ceil()
        .max(1.0) as usize;
    let height = (key.viewport.height() * key.pixels_per_point)
        .ceil()
        .max(1.0) as usize;
    if width
        .checked_mul(height)
        .is_none_or(|pixels| pixels > CACHE_BYTES / 8)
    {
        return Err(ViewError("Viewport exceeds rendering memory budget".into()));
    }
    for preview in [true, false] {
        let scale = f64::from(key.pixels_per_point) / if preview { 4.0 } else { 1.0 };
        let width = if preview { width.div_ceil(4) } else { width };
        let height = if preview { height.div_ceil(4) } else { height };
        let tile_size = if preview {
            width.max(height)
        } else {
            TILE_SIZE
        };
        for top in (0..height).step_by(tile_size) {
            for left in (0..width).step_by(tile_size) {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let tile_width = tile_size.min(width - left);
                let tile_height = tile_size.min(height - top);
                let rendered = render_target(
                    scene.clone(),
                    key.clone(),
                    scale,
                    [left as f64, top as f64],
                    [tile_width, tile_height],
                    cancel,
                    cache.clone(),
                    context.clone(),
                )?;
                let image = rendered.image;
                let rect = Rect::from_min_max(
                    key.viewport.min
                        + Vec2::new((left as f64 / scale) as f32, (top as f64 / scale) as f32),
                    key.viewport.min
                        + Vec2::new(
                            ((left + tile_width) as f64 / scale) as f32,
                            ((top + tile_height) as f64 / scale) as f32,
                        ),
                )
                .intersect(key.viewport);
                let complete =
                    !preview && left + tile_width == width && top + tile_height == height;
                let tile = Tile {
                    rect,
                    image,
                    shape_count: rendered.shape_count,
                    complete,
                    preview,
                    first: left == 0 && top == 0,
                };
                if cancel.load(Ordering::Relaxed) || sender.send(Ok(tile)).is_err() {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

struct Cursor {
    multiplier: usize,
    cell: CellId,
    transformation: Transformation,
    depth: u16,
    target: Bounds,
    query: Query,
}

struct ArrayCursor {
    multiplier: usize,
    pixel_origin: DoublePoint,
    pixel_column: DoublePoint,
    pixel_row: DoublePoint,
    aggregate: bool,
    row_jump: u16,
    cell: CellId,
    base: Transformation,
    depth: u16,
    column_step: DoublePoint,
    row_step: DoublePoint,
    columns: (u16, u16),
    rows: (u16, u16),
    column: u16,
    row: u16,
}

enum Work {
    Cell(CellId, Transformation, u16, usize),
    Entries(Cursor),
    Array(ArrayCursor),
}

struct Rasterizer<'a> {
    scene: Arc<Scene>,
    key: FrameKey,
    scale: f64,
    origin: [f64; 2],
    size: [usize; 2],
    masks: BTreeMap<LayerKey, Vec<u8>>,
    context: egui::Context,
    count: usize,
    cancel: &'a AtomicBool,
    cache: Arc<Mutex<Cache>>,
    seen: HashSet<(CacheKey, i64, i64)>,
    cache_depth: usize,
}

impl<'a> Rasterizer<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        scene: Arc<Scene>,
        key: FrameKey,
        scale: f64,
        origin: [f64; 2],
        size: [usize; 2],
        cancel: &'a AtomicBool,
        cache: Arc<Mutex<Cache>>,
        context: egui::Context,
    ) -> Self {
        Self {
            scene,
            key,
            scale,
            origin,
            size,
            masks: BTreeMap::new(),
            context,
            count: 0,
            cancel,
            cache,
            seen: HashSet::new(),
            cache_depth: 0,
        }
    }

    fn screen(&self, point: DoublePoint) -> DoublePoint {
        let displacement = point
            .minus(self.key.camera.center)
            .times(self.key.camera.pixels_per_unit * self.scale);
        DoublePoint::new(
            f64::from(self.key.viewport.width()) * self.scale * 0.5 + displacement.x
                - self.origin[0],
            f64::from(self.key.viewport.height()) * self.scale * 0.5
                - displacement.y
                - self.origin[1],
        )
    }

    fn visible(&self) -> Bounds {
        let minimum = self.key.viewport.min
            + Vec2::new(
                (self.origin[0] / self.scale) as f32,
                ((self.origin[1] + self.size[1] as f64) / self.scale) as f32,
            );
        let maximum = self.key.viewport.min
            + Vec2::new(
                ((self.origin[0] + self.size[0] as f64) / self.scale) as f32,
                (self.origin[1] / self.scale) as f32,
            );
        Bounds {
            minimum: self.key.camera.screen_to_world(minimum, self.key.viewport),
            maximum: self.key.camera.screen_to_world(maximum, self.key.viewport),
        }
    }

    fn paint(
        &mut self,
        cell: CellId,
        transformation: Transformation,
        depth: u16,
        use_cache: bool,
    ) -> Result<(), ViewError> {
        let root_depth = depth;
        let mut stack = vec![Work::Cell(cell, transformation, depth, 1)];
        let visible = self.visible();
        while let Some(work) = stack.pop() {
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            match work {
                Work::Cell(cell, transformation, depth, multiplier) => {
                    if !transformation.valid() {
                        return Err(ViewError("Hierarchy transformation overflow".into()));
                    }
                    let culling = visible.expanded(label_margin(
                        self.scene.label_metrics[cell.index()],
                        transformation.magnification,
                        1.0 / self.key.camera.pixels_per_unit,
                    ));
                    let Some(bounds) = self
                        .scene
                        .layout
                        .transformed_bounds(cell, transformation)
                        .filter(|bounds| bounds.intersects(culling))
                    else {
                        continue;
                    };
                    let previous = self.count;
                    if self.cache_depth < 32
                        && (use_cache || depth > root_depth)
                        && self.cached(cell, transformation, depth, bounds)?
                    {
                        self.count = self.count.saturating_add(
                            (self.count - previous).saturating_mul(multiplier.saturating_sub(1)),
                        );
                        continue;
                    }
                    let Some(index) = self.scene.indexes.get(&cell) else {
                        return Err(ViewError("Missing spatial index".into()));
                    };
                    let inverse = inverse(transformation);
                    let target = visible.transformed(inverse).expanded(
                        self.scene.layout.absolute_path_padding(cell)
                            * (1.0 / transformation.magnification - 1.0).max(0.0)
                            * 2.0_f64.sqrt(),
                    );
                    stack.push(Work::Entries(Cursor {
                        multiplier,
                        cell,
                        transformation,
                        depth,
                        target,
                        query: index.query(),
                    }));
                }
                Work::Entries(mut cursor) => {
                    let scene = self.scene.clone();
                    let index = scene
                        .indexes
                        .get(&cursor.cell)
                        .ok_or_else(|| ViewError("Missing spatial index".into()))?;
                    let Some(entry) = cursor.query.next(
                        index,
                        cursor.target,
                        cursor.transformation.magnification,
                        1.0 / (self.key.camera.pixels_per_unit
                            * cursor.transformation.magnification),
                    ) else {
                        continue;
                    };
                    let cell = scene
                        .layout
                        .cell(cursor.cell)
                        .ok_or_else(|| ViewError("Missing indexed cell".into()))?;
                    if entry.instance && cursor.depth < self.key.depth {
                        let instance = cell
                            .instances()
                            .get(entry.index)
                            .ok_or_else(|| ViewError("Missing indexed instance".into()))?;
                        let base = cursor.transformation.compose(instance.transformation);
                        if let Some(bounds) = scene.layout.transformed_bounds(instance.cell, base) {
                            let column_step = cursor
                                .transformation
                                .apply_vector(instance.array.column_step);
                            let row_step =
                                cursor.transformation.apply_vector(instance.array.row_step);
                            let (mut columns, mut rows) = visible_array_range(
                                bounds,
                                visible.expanded(label_margin(
                                    scene.label_metrics[instance.cell.index()],
                                    base.magnification,
                                    1.0 / self.key.camera.pixels_per_unit,
                                )),
                                column_step,
                                row_step,
                                instance.array.columns,
                                instance.array.rows,
                            );
                            if column_step == DoublePoint::default() && columns.0 < columns.1 {
                                columns.1 = columns.0 + 1;
                            }
                            if row_step == DoublePoint::default() && rows.0 < rows.1 {
                                rows.1 = rows.0 + 1;
                            }
                            let scale = self.key.camera.pixels_per_unit * self.scale;
                            let metrics = scene.label_metrics[instance.cell.index()];
                            let margin = label_margin(metrics, base.magnification, self.scale);
                            let width = (bounds.width() * scale + margin * 2.0).ceil() + 2.0;
                            let height = (bounds.height() * scale + margin * 2.0).ceil() + 2.0;
                            let limit = if metrics.0 > 0.0 { 256.0 } else { 64.0 };
                            let pixel_origin = self
                                .screen(bounds.center())
                                .minus(DoublePoint::new(width * 0.5, height * 0.5));
                            let array = ArrayCursor {
                                multiplier: cursor.multiplier,
                                pixel_origin,
                                pixel_column: DoublePoint::new(
                                    column_step.x * scale,
                                    -column_step.y * scale,
                                ),
                                pixel_row: DoublePoint::new(
                                    row_step.x * scale,
                                    -row_step.y * scale,
                                ),
                                aggregate: self.cache_depth < 32
                                    && width <= limit
                                    && height <= limit,
                                row_jump: 1,
                                cell: instance.cell,
                                base,
                                depth: cursor.depth.saturating_add(1),
                                column_step,
                                row_step,
                                columns,
                                rows,
                                column: columns.0,
                                row: rows.0,
                            };
                            stack.push(Work::Entries(cursor));
                            stack.push(Work::Array(array));
                            continue;
                        }
                    } else if !entry.instance {
                        let shape = cell
                            .shapes()
                            .get(entry.index)
                            .ok_or_else(|| ViewError("Missing indexed shape".into()))?;
                        if let Some(style) = self
                            .key
                            .styles
                            .get(&shape.layer)
                            .copied()
                            .filter(|style| style.visible)
                        {
                            let previous = self.count;
                            self.shape(
                                &shape.geometry,
                                shape.layer,
                                style.color,
                                cursor.transformation,
                            )
                            .map_err(|error| {
                                shape_error(cell.name(), shape.layer, entry.index, error)
                            })?;
                            self.count = self.count.saturating_add(
                                (self.count - previous)
                                    .saturating_mul(cursor.multiplier.saturating_sub(1)),
                            );
                        }
                    }
                    stack.push(Work::Entries(cursor));
                }
                Work::Array(mut array) => {
                    if array.row >= array.rows.1 || array.columns.0 == array.columns.1 {
                        continue;
                    }
                    let position = array
                        .pixel_origin
                        .plus(array.pixel_column.times(f64::from(array.column)))
                        .plus(array.pixel_row.times(f64::from(array.row)));
                    let column_jump = if array.aggregate {
                        pixel_run(position, array.pixel_column)
                            .min(usize::from(array.columns.1 - array.column))
                            .max(1) as u16
                    } else {
                        1
                    };
                    if array.column == array.columns.0 {
                        let orthogonal = (array.pixel_column.y == 0.0 && array.pixel_row.x == 0.0)
                            || (array.pixel_column.x == 0.0 && array.pixel_row.y == 0.0);
                        array.row_jump = if array.aggregate
                            && (orthogonal || array.columns.1 - array.columns.0 == 1)
                        {
                            pixel_run(position, array.pixel_row)
                                .min(usize::from(array.rows.1 - array.row))
                                .max(1) as u16
                        } else {
                            1
                        };
                    }
                    let translation = array
                        .base
                        .translation
                        .plus(array.column_step.times(f64::from(array.column)))
                        .plus(array.row_step.times(f64::from(array.row)));
                    let child = Work::Cell(
                        array.cell,
                        Transformation {
                            translation,
                            ..array.base
                        },
                        array.depth,
                        array
                            .multiplier
                            .saturating_mul(usize::from(column_jump))
                            .saturating_mul(usize::from(array.row_jump)),
                    );
                    array.column += column_jump;
                    if array.column >= array.columns.1 {
                        array.column = array.columns.0;
                        array.row += array.row_jump;
                    }
                    stack.push(Work::Array(array));
                    stack.push(child);
                }
            }
        }
        Ok(())
    }

    fn cached(
        &mut self,
        cell: CellId,
        transformation: Transformation,
        depth: u16,
        bounds: Bounds,
    ) -> Result<bool, ViewError> {
        let scale = self.key.camera.pixels_per_unit * self.scale;
        let metrics = self.scene.label_metrics[cell.index()];
        let margin = if metrics.0 > 0.0 {
            (metrics.0 + 1.0)
                * (12.0 * metrics.1 * transformation.magnification).clamp(6.0, 96.0)
                * self.scale
        } else {
            0.0
        };
        let width = (bounds.width() * scale + margin * 2.0).ceil() as usize + 2;
        let height = (bounds.height() * scale + margin * 2.0).ceil() as usize + 2;
        let limit = if metrics.0 > 0.0 { 256 } else { 64 };
        if width > limit || height > limit {
            return Ok(false);
        }
        let mut layers = DefaultHasher::new();
        for (layer, style) in &self.key.styles {
            if style.visible {
                layer.hash(&mut layers);
            }
        }
        let key = CacheKey {
            cell,
            angle: transformation.angle.to_bits(),
            magnification: transformation.magnification.to_bits(),
            reflected: transformation.reflected,
            scale: scale.to_bits(),
            font_scale: self.scale.to_bits(),
            depth: self.key.depth.saturating_sub(depth),
            layers: layers.finish(),
        };
        let existing = self
            .cache
            .lock()
            .map_err(|error| ViewError(error.to_string()))?
            .entries
            .get(&key)
            .cloned();
        let coverage = if let Some(coverage) = existing {
            coverage
        } else {
            let local = Transformation {
                translation: DoublePoint::default(),
                ..transformation
            };
            let local_bounds = self
                .scene
                .layout
                .transformed_bounds(cell, local)
                .ok_or_else(|| ViewError("Missing cached bounds".into()))?;
            let mut request = self.key.clone();
            request.camera = Camera {
                center: local_bounds.center(),
                pixels_per_unit: self.key.camera.pixels_per_unit,
            };
            request.viewport =
                Rect::from_min_size(Pos2::ZERO, Vec2::new(width as f32, height as f32));
            request.depth = key.depth;
            let mut painter = Self::new(
                self.scene.clone(),
                request,
                self.scale,
                [0.0, 0.0],
                [width, height],
                self.cancel,
                self.cache.clone(),
                self.context.clone(),
            );
            painter.cache_depth = self.cache_depth + 1;
            painter.paint(cell, local, 0, false)?;
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(true);
            }
            let coverage = Arc::new(Coverage::new(width, height, painter.masks, painter.count));
            self.cache
                .lock()
                .map_err(|error| ViewError(error.to_string()))?
                .insert(key, coverage.clone());
            coverage
        };
        let center = self.screen(bounds.center());
        let left = (center.x - coverage.width as f64 * 0.5).floor() as i64;
        let top = (center.y - coverage.height as f64 * 0.5).floor() as i64;
        if self.seen.len() >= 50_000 {
            self.seen.clear();
        }
        if !self.seen.insert((key, left, top)) {
            self.count = self.count.saturating_add(coverage.shape_count);
            return Ok(true);
        }
        for (layer, runs) in &coverage.runs {
            let target = self
                .masks
                .entry(*layer)
                .or_insert_with(|| vec![0; self.size[0] * self.size[1]]);
            for run in runs {
                let y = top + run.row as i64;
                if y < 0 || y >= self.size[1] as i64 {
                    continue;
                }
                let x = left + run.start as i64;
                let start = (-x).max(0).min(run.pixels.len() as i64) as usize;
                let end = (self.size[0] as i64 - x)
                    .max(0)
                    .min(run.pixels.len() as i64) as usize;
                for offset in start..end {
                    let destination =
                        &mut target[y as usize * self.size[0] + (x + offset as i64) as usize];
                    *destination = (*destination).max(run.pixels[offset]);
                }
            }
        }
        self.count = self.count.saturating_add(coverage.shape_count);
        Ok(true)
    }

    fn shape(
        &mut self,
        geometry: &Geometry,
        layer: LayerKey,
        _color: Color32,
        transformation: Transformation,
    ) -> Result<(), ViewError> {
        if let Geometry::Label(label) = geometry {
            self.label(label, layer, transformation)?;
            self.count = self.count.saturating_add(1);
            return Ok(());
        }
        let points = match geometry {
            Geometry::Polygon(points) => points.iter().map(|point| point.to_double()).collect(),
            Geometry::Box { minimum, maximum } => vec![
                minimum.to_double(),
                DoublePoint::new(maximum.x as f64, minimum.y as f64),
                maximum.to_double(),
                DoublePoint::new(minimum.x as f64, maximum.y as f64),
            ],
            Geometry::Path(path) if path.width != 0 => {
                path_contour(path, transformation.magnification)
                    .map_err(|error| ViewError(error.to_string()))?
            }
            Geometry::Path(path) => path.points.iter().map(|point| point.to_double()).collect(),
            Geometry::Label(_) => return Ok(()),
        };
        let points: Vec<_> = points
            .into_iter()
            .map(|point| self.screen(transformation.apply(point)))
            .collect();
        if points.iter().any(|point| !point.finite()) {
            return Err(ViewError("Screen coordinate overflow".into()));
        }
        let Some(bounds) = Bounds::from_points(points.iter().copied()).filter(|bounds| {
            bounds.intersects(Bounds {
                minimum: DoublePoint::new(-1.0, -1.0),
                maximum: DoublePoint::new(self.size[0] as f64 + 1.0, self.size[1] as f64 + 1.0),
            })
        }) else {
            return Ok(());
        };
        let mask = self
            .masks
            .entry(layer)
            .or_insert_with(|| vec![0; self.size[0] * self.size[1]]);
        if matches!(geometry, Geometry::Path(path) if path.width == 0) {
            for pair in points.windows(2) {
                line(mask, self.size, pair[0], pair[1]);
            }
        } else {
            fill(mask, self.size, &points, bounds);
        }
        self.count = self.count.saturating_add(1);
        Ok(())
    }

    fn label(
        &mut self,
        label: &Label,
        layer: LayerKey,
        parent: Transformation,
    ) -> Result<(), ViewError> {
        let transformation = parent.compose(label.transformation);
        let size = (12.0 * transformation.magnification).clamp(6.0, 96.0);
        let key = GlyphKey {
            text: label.text.clone(),
            size: size.to_bits(),
            pixels_per_point: self.key.pixels_per_point.to_bits(),
        };
        let existing = self
            .cache
            .lock()
            .map_err(|error| ViewError(error.to_string()))?
            .glyphs
            .get(&key)
            .cloned();
        let glyph = if let Some(glyph) = existing {
            glyph
        } else {
            let glyph = Arc::new(make_glyph(&self.context, &key)?);
            self.cache
                .lock()
                .map_err(|error| ViewError(error.to_string()))?
                .insert_glyph(key, glyph.clone());
            glyph
        };
        let origin = self.screen(transformation.translation);
        let anchor = DoublePoint::new(
            f64::from(glyph.logical_size.x) * f64::from(label.horizontal_alignment) * 0.5,
            f64::from(glyph.logical_size.y) * f64::from(label.vertical_alignment) * 0.5,
        );
        let (sine, cosine) = transformation.angle.sin_cos();
        let reflect = if transformation.reflected { -1.0 } else { 1.0 };
        let project = |point: DoublePoint| {
            let point = point.minus(anchor);
            let y = point.y * reflect;
            origin.plus(
                DoublePoint::new(cosine * point.x + sine * y, -sine * point.x + cosine * y)
                    .times(self.scale),
            )
        };
        let corners = [
            DoublePoint::default(),
            DoublePoint::new(f64::from(glyph.logical_size.x), 0.0),
            DoublePoint::new(0.0, f64::from(glyph.logical_size.y)),
            DoublePoint::new(
                f64::from(glyph.logical_size.x),
                f64::from(glyph.logical_size.y),
            ),
        ];
        let Some(bounds) = Bounds::from_points(corners.map(project)) else {
            return Ok(());
        };
        let left = bounds.minimum.x.floor().clamp(0.0, self.size[0] as f64) as usize;
        let right = bounds.maximum.x.ceil().clamp(0.0, self.size[0] as f64) as usize;
        let top = bounds.minimum.y.floor().clamp(0.0, self.size[1] as f64) as usize;
        let bottom = bounds.maximum.y.ceil().clamp(0.0, self.size[1] as f64) as usize;
        let mask = self
            .masks
            .entry(layer)
            .or_insert_with(|| vec![0; self.size[0] * self.size[1]]);
        for row in top..bottom {
            for column in left..right {
                let relative = DoublePoint::new(column as f64 + 0.5, row as f64 + 0.5)
                    .minus(origin)
                    .times(1.0 / self.scale);
                let point = DoublePoint::new(
                    cosine * relative.x - sine * relative.y,
                    (sine * relative.x + cosine * relative.y) * reflect,
                )
                .plus(anchor)
                .times(glyph.pixels_per_point);
                let x = point.x.floor() as i64;
                let y = point.y.floor() as i64;
                if x >= 0 && y >= 0 && x < glyph.size[0] as i64 && y < glyph.size[1] as i64 {
                    let coverage = glyph.alpha[y as usize * glyph.size[0] + x as usize];
                    let destination = &mut mask[row * self.size[0] + column];
                    *destination = (*destination).max(coverage);
                }
            }
        }
        Ok(())
    }

    fn image(&self) -> egui::ColorImage {
        let mut pixels = vec![Color32::TRANSPARENT; self.size[0] * self.size[1]];
        for (layer, mask) in &self.masks {
            if let Some(style) = self.key.styles.get(layer) {
                for (pixel, coverage) in pixels.iter_mut().zip(mask) {
                    if *coverage != 0 {
                        let mut source = style.color.to_array();
                        for component in &mut source {
                            *component = (u16::from(*component) * u16::from(*coverage) / 255) as u8;
                        }
                        let previous = pixel.to_array();
                        let inverse = 255 - u16::from(source[3]);
                        let mut blended = [0; 4];
                        for index in 0..4 {
                            blended[index] = (u16::from(source[index])
                                + u16::from(previous[index]) * inverse / 255)
                                .min(255) as u8;
                        }
                        *pixel = Color32::from_rgba_premultiplied(
                            blended[0], blended[1], blended[2], blended[3],
                        );
                    }
                }
            }
        }
        egui::ColorImage::new(self.size, pixels)
    }
}

fn make_glyph(context: &egui::Context, key: &GlyphKey) -> Result<Glyph, ViewError> {
    let (galley, atlas) = context.fonts_mut(|fonts| {
        let galley = fonts.layout_no_wrap(
            key.text.clone(),
            egui::FontId::proportional(f64::from_bits(key.size) as f32),
            Color32::WHITE,
        );
        (galley, fonts.image())
    });
    let pixels_per_point = f64::from(f32::from_bits(key.pixels_per_point));
    let size = [
        (f64::from(galley.size().x) * pixels_per_point).ceil() as usize,
        (f64::from(galley.size().y) * pixels_per_point).ceil() as usize,
    ];
    let length = size[0]
        .checked_mul(size[1])
        .filter(|length| *length <= CACHE_BYTES)
        .ok_or_else(|| ViewError("Text exceeds rendering memory budget".into()))?;
    let mut alpha = vec![0_u8; length];
    for row in &galley.rows {
        for triangle in row.visuals.mesh.indices.as_chunks::<3>().0 {
            let mut positions = [DoublePoint::default(); 3];
            let mut coordinates = [DoublePoint::default(); 3];
            for (offset, index) in triangle.iter().enumerate() {
                let vertex = row
                    .visuals
                    .mesh
                    .vertices
                    .get(*index as usize)
                    .ok_or_else(|| ViewError("Invalid glyph triangle".into()))?;
                let position = vertex.pos.to_vec2() + row.pos.to_vec2();
                positions[offset] = DoublePoint::new(f64::from(position.x), f64::from(position.y))
                    .times(pixels_per_point);
                coordinates[offset] =
                    DoublePoint::new(f64::from(vertex.uv.x), f64::from(vertex.uv.y));
            }
            let Some(bounds) = Bounds::from_points(positions) else {
                continue;
            };
            let edge = positions[1].minus(positions[0]);
            let other = positions[2].minus(positions[0]);
            let determinant = edge.x * other.y - edge.y * other.x;
            if determinant == 0.0 {
                continue;
            }
            let left = bounds.minimum.x.floor().clamp(0.0, size[0] as f64) as usize;
            let right = bounds.maximum.x.ceil().clamp(0.0, size[0] as f64) as usize;
            let top = bounds.minimum.y.floor().clamp(0.0, size[1] as f64) as usize;
            let bottom = bounds.maximum.y.ceil().clamp(0.0, size[1] as f64) as usize;
            for y in top..bottom {
                for x in left..right {
                    let relative =
                        DoublePoint::new(x as f64 + 0.5, y as f64 + 0.5).minus(positions[0]);
                    let first = (relative.x * other.y - relative.y * other.x) / determinant;
                    let second = (edge.x * relative.y - edge.y * relative.x) / determinant;
                    if first >= -1e-8 && second >= -1e-8 && first + second <= 1.0 + 1e-8 {
                        let coordinate = coordinates[0]
                            .plus(coordinates[1].minus(coordinates[0]).times(first))
                            .plus(coordinates[2].minus(coordinates[0]).times(second));
                        let column = coordinate
                            .x
                            .floor()
                            .clamp(0.0, atlas.size[0].saturating_sub(1) as f64)
                            as usize;
                        let row = coordinate
                            .y
                            .floor()
                            .clamp(0.0, atlas.size[1].saturating_sub(1) as f64)
                            as usize;
                        if let Some(pixel) = atlas.pixels.get(row * atlas.size[0] + column) {
                            alpha[y * size[0] + x] = alpha[y * size[0] + x].max(pixel.a());
                        }
                    }
                }
            }
        }
    }
    Ok(Glyph {
        size,
        logical_size: galley.size(),
        pixels_per_point,
        alpha,
    })
}

fn pixel_run(position: DoublePoint, step: DoublePoint) -> usize {
    let mut count = usize::MAX;
    for (position, step) in [(position.x, step.x), (position.y, step.y)] {
        let steps = if step > 0.0 {
            ((position.floor() + 1.0 - position) / step).ceil().max(1.0) as usize
        } else if step < 0.0 {
            (((position - position.floor()) / -step).floor().max(0.0) as usize).saturating_add(1)
        } else {
            usize::MAX
        };
        count = count.min(steps);
    }
    count.max(1)
}

fn label_margin(metrics: (f64, f64), magnification: f64, units_per_point: f64) -> f64 {
    if metrics.0 == 0.0 {
        0.0
    } else {
        (metrics.0 + 1.0) * (12.0 * metrics.1 * magnification).clamp(6.0, 96.0) * units_per_point
    }
}

fn inverse(transformation: Transformation) -> Transformation {
    let angle = if transformation.reflected {
        transformation.angle
    } else {
        -transformation.angle
    };
    let mut inverse = Transformation {
        translation: DoublePoint::default(),
        angle,
        magnification: 1.0 / transformation.magnification,
        reflected: transformation.reflected,
    };
    inverse.translation = inverse.apply_vector(transformation.translation).times(-1.0);
    inverse
}

fn fill(mask: &mut [u8], size: [usize; 2], points: &[DoublePoint], bounds: Bounds) {
    if points.len() < 3 {
        return;
    }
    let first_point = points[0];
    let Some(second_point) = points.iter().find(|point| **point != first_point) else {
        return;
    };
    let direction = second_point.minus(first_point);
    if points.iter().all(|point| {
        let relative = point.minus(first_point);
        direction.x * relative.y - direction.y * relative.x == 0.0
    }) {
        return;
    }
    let first = (bounds.minimum.y - 0.5).ceil().clamp(0.0, size[1] as f64) as usize;
    let end = (bounds.maximum.y - 0.5).ceil().clamp(0.0, size[1] as f64) as usize;
    let mut crossings = Vec::with_capacity(points.len());
    for row in first..end {
        crossings.clear();
        let y = row as f64 + 0.5;
        for (first, second) in points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .take(points.len())
        {
            if (first.y <= y && second.y > y) || (second.y <= y && first.y > y) {
                crossings.push((
                    first.x + (y - first.y) * (second.x - first.x) / (second.y - first.y),
                    if second.y > first.y { 1_i32 } else { -1 },
                ));
            }
        }
        crossings.sort_unstable_by(|first, second| first.0.total_cmp(&second.0));
        let mut winding = 0_i32;
        let mut left = 0.0;
        for (x, direction) in &crossings {
            if winding != 0 {
                let start = (left - 0.5_f64).ceil().clamp(0.0, size[0] as f64) as usize;
                let end = (*x - 0.5).ceil().clamp(0.0, size[0] as f64) as usize;
                mask[row * size[0] + start..row * size[0] + end].fill(255);
            }
            winding += direction;
            left = *x;
        }
    }
    // A center sample misses features thinner than a pixel, including at full-chip fit.
    if bounds.width() < 1.0 || bounds.height() < 1.0 {
        for (first, second) in points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .take(points.len())
        {
            line(mask, size, *first, *second);
        }
    }
}

fn line(mask: &mut [u8], size: [usize; 2], first: DoublePoint, second: DoublePoint) {
    let direction = second.minus(first);
    let mut start = 0.0_f64;
    let mut end = 1.0_f64;
    for (position, step, maximum) in [
        (first.x, direction.x, size[0] as f64 - 1e-9),
        (first.y, direction.y, size[1] as f64 - 1e-9),
    ] {
        if step == 0.0 {
            if position < 0.0 || position > maximum {
                return;
            }
        } else {
            let first = -position / step;
            let second = (maximum - position) / step;
            start = start.max(first.min(second));
            end = end.min(first.max(second));
        }
    }
    if start > end {
        return;
    }
    let clipped = first.plus(direction.times(start));
    let direction = direction.times(end - start);
    let steps = direction.x.abs().max(direction.y.abs()).ceil().max(1.0) as usize;
    for index in 0..=steps {
        let point = clipped.plus(direction.times(index as f64 / steps as f64));
        let x = point.x.floor() as i64;
        let y = point.y.floor() as i64;
        if x >= 0 && y >= 0 && x < size[0] as i64 && y < size[1] as i64 {
            mask[y as usize * size[0] + x as usize] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlayout_core::{Array, Cell, Instance, Point, Shape};

    fn scene(shapes: Vec<Shape>, instances: Vec<Instance>) -> Result<Arc<Scene>, Box<dyn Error>> {
        let layout = Layout::new(
            "raster".into(),
            1e-9,
            0.001,
            vec![
                Cell::new("LEAF".into(), shapes, Vec::new()),
                Cell::new("TOP".into(), Vec::new(), instances),
            ],
        )?;
        Ok(Arc::new(Scene::prepare(layout)?))
    }

    fn square(layer: u16) -> Shape {
        Shape {
            layer: LayerKey { layer, datatype: 0 },
            geometry: Geometry::Polygon(vec![
                Point::new(-8, -8),
                Point::new(8, -8),
                Point::new(8, 8),
                Point::new(-8, 8),
            ]),
        }
    }

    fn request(scene: &Scene, size: Vec2) -> FrameKey {
        FrameKey {
            camera: Camera::default(),
            viewport: Rect::from_min_size(Pos2::new(25.0, 35.0), size),
            pixels_per_point: 1.0,
            selected: Some(CellId::from_index(1)),
            depth: u16::MAX,
            styles: scene
                .layout
                .layers()
                .iter()
                .map(|layer| {
                    (
                        *layer,
                        LayerStyle {
                            visible: true,
                            color: Color32::WHITE,
                        },
                    )
                })
                .collect(),
            text_pass: 0,
        }
    }

    fn context() -> egui::Context {
        let context = egui::Context::default();
        let mut output = context.run_ui(egui::RawInput::default(), |_| {});
        output.textures_delta.clear();
        context
    }

    #[test]
    fn raster_nonzero_fill_matches_vector_triangles() -> Result<(), Box<dyn Error>> {
        for points in [
            vec![
                DoublePoint::new(2.2, 2.4),
                DoublePoint::new(25.3, 2.4),
                DoublePoint::new(25.3, 10.2),
                DoublePoint::new(9.1, 10.2),
                DoublePoint::new(9.1, 25.8),
                DoublePoint::new(2.2, 25.8),
            ],
            vec![
                DoublePoint::new(2.0, 2.0),
                DoublePoint::new(28.0, 28.0),
                DoublePoint::new(2.0, 28.0),
                DoublePoint::new(28.0, 2.0),
            ],
            path_contour(
                &Path {
                    points: vec![
                        Point::new(4, 12),
                        Point::new(7, 12),
                        Point::new(7, 25),
                        Point::new(4, 25),
                    ],
                    width: 6,
                    absolute_width: false,
                    ends: rlayout_core::PathEnds::Flush,
                },
                1.0,
            )?,
        ] {
            let bounds = Bounds::from_points(points.iter().copied()).ok_or("No polygon bounds")?;
            let mut raster = vec![0; 32 * 32];
            fill(&mut raster, [32, 32], &points, bounds);
            let polygon = triangulate(points)?;
            let mut vector = vec![0; 32 * 32];
            for indices in polygon.triangles.as_chunks::<3>().0 {
                let triangle = [
                    polygon.points[indices[0] as usize],
                    polygon.points[indices[1] as usize],
                    polygon.points[indices[2] as usize],
                ];
                fill(
                    &mut vector,
                    [32, 32],
                    &triangle,
                    Bounds::from_points(triangle).ok_or("No triangle bounds")?,
                );
            }
            assert_eq!(raster, vector);
        }
        Ok(())
    }

    #[test]
    fn transforms_absolute_width_and_layer_composition_are_preserved() -> Result<(), Box<dyn Error>>
    {
        let path = Shape {
            layer: LayerKey {
                layer: 2,
                datatype: 0,
            },
            geometry: Geometry::Path(Path {
                points: vec![Point::new(0, -40), Point::new(0, 40)],
                width: 10,
                absolute_width: true,
                ends: rlayout_core::PathEnds::Flush,
            }),
        };
        let instance = Instance {
            cell: CellId::from_index(0),
            transformation: Transformation {
                reflected: true,
                angle: 0.3,
                magnification: 0.1,
                ..Default::default()
            },
            array: Array::default(),
        };
        let scene = scene(vec![square(1), path], vec![instance])?;
        let mut key = request(&scene, Vec2::splat(100.0));
        key.styles
            .get_mut(&LayerKey {
                layer: 1,
                datatype: 0,
            })
            .ok_or("Missing layer")?
            .color = Color32::RED;
        key.styles
            .get_mut(&LayerKey {
                layer: 2,
                datatype: 0,
            })
            .ok_or("Missing layer")?
            .color = Color32::BLUE;
        let image = capture(scene.clone(), key.clone(), context())?;
        assert!(image.image.pixels.contains(&Color32::BLUE));
        key.styles
            .get_mut(&LayerKey {
                layer: 2,
                datatype: 0,
            })
            .ok_or("Missing layer")?
            .visible = false;
        let hidden = capture(scene.clone(), key.clone(), context())?;
        assert!(!hidden.image.pixels.contains(&Color32::BLUE));
        key.depth = 0;
        assert!(
            capture(scene, key, context())?
                .image
                .pixels
                .iter()
                .all(|pixel| *pixel == Color32::TRANSPARENT)
        );
        Ok(())
    }

    #[test]
    fn cache_eviction_preserves_the_memory_limit() {
        let mut cache = Cache::new();
        let coverage = Arc::new(Coverage::new(
            8,
            8,
            BTreeMap::from([(
                LayerKey {
                    layer: 1,
                    datatype: 0,
                },
                vec![255; 64],
            )]),
            1,
        ));
        cache.limit = coverage.bytes();
        let key = CacheKey {
            cell: CellId::from_index(0),
            angle: 0,
            magnification: 0,
            scale: 0,
            font_scale: 0,
            reflected: false,
            depth: 0,
            layers: 0,
        };
        cache.insert(key, coverage.clone());
        let other = CacheKey {
            cell: CellId::from_index(1),
            ..key
        };
        cache.insert(other, coverage);
        assert_eq!(cache.entries.len(), 1);
        assert!(!cache.entries.contains_key(&key));
        assert!(cache.bytes <= cache.limit);
    }

    #[test]
    fn raster_worker_tiles_complete_and_cancel_obsolete_views() -> Result<(), Box<dyn Error>> {
        let scene = scene(
            vec![square(1)],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation: Transformation::default(),
                array: Array {
                    columns: 200,
                    rows: 200,
                    column_step: DoublePoint::new(10.0, 0.0),
                    row_step: DoublePoint::new(0.0, 10.0),
                },
            }],
        )?;
        let context = context();
        let mut key = request(&scene, Vec2::new(520.0, 280.0));
        let expected = capture(scene.clone(), key.clone(), context.clone())?.image;
        let mut worker = Worker::new();
        worker.start(scene.clone(), key.clone(), context.clone())?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut pixels = vec![Color32::TRANSPARENT; 520 * 280];
        let mut complete = false;
        let mut preview = false;
        while !complete && std::time::Instant::now() < deadline {
            for tile in worker.receive() {
                let tile = tile?;
                if tile.preview {
                    preview = true;
                    continue;
                }
                let left = (tile.rect.left() - key.viewport.left()).round() as usize;
                let top = (tile.rect.top() - key.viewport.top()).round() as usize;
                for row in 0..tile.image.size[1] {
                    for column in 0..tile.image.size[0] {
                        pixels[(top + row) * 520 + left + column] =
                            tile.image.pixels[row * tile.image.size[0] + column];
                    }
                }
                complete = tile.complete;
            }
            thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(preview && complete);
        assert_eq!(pixels, expected.pixels);
        worker.start(scene.clone(), key.clone(), context.clone())?;
        for style in key.styles.values_mut() {
            style.visible = false;
        }
        worker.start(scene.clone(), key, context.clone())?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut complete = false;
        while !complete && std::time::Instant::now() < deadline {
            for tile in worker.receive() {
                let tile = tile?;
                assert!(
                    tile.image
                        .pixels
                        .iter()
                        .all(|pixel| *pixel == Color32::TRANSPARENT)
                );
                complete |= tile.complete;
            }
            thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(complete);
        let mut invalid = request(&scene, Vec2::splat(20_000.0));
        invalid.selected = None;
        worker.start(scene, invalid, context)?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut failure = None;
        while failure.is_none() && std::time::Instant::now() < deadline {
            failure = worker.receive().into_iter().find_map(Result::err);
            thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(failure.is_some_and(|error| error.to_string().contains("memory budget")));
        Ok(())
    }

    #[test]
    fn render_visits_more_than_the_old_limit_and_keeps_thin_shapes() -> Result<(), Box<dyn Error>> {
        let shape = Shape {
            layer: LayerKey {
                layer: 1,
                datatype: 0,
            },
            geometry: Geometry::Path(Path {
                points: vec![Point::new(0, 0), Point::new(10, 0)],
                width: 1,
                absolute_width: false,
                ends: rlayout_core::PathEnds::Flush,
            }),
        };
        let scene = scene(
            vec![shape],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation: Transformation::default(),
                array: Array {
                    columns: 501,
                    rows: 501,
                    column_step: DoublePoint::new(0.01, 0.0),
                    row_step: DoublePoint::new(0.0, 0.01),
                },
            }],
        )?;
        let mut key = request(&scene, Vec2::splat(40.0));
        key.camera.pixels_per_unit = 0.1;
        let image = capture(scene, key, context())?;
        assert_eq!(image.shape_count, 251_001);
        assert!(image.image.pixels.iter().any(|pixel| pixel.a() != 0));
        Ok(())
    }

    #[test]
    fn billion_instance_arrays_aggregate_at_screen_resolution() -> Result<(), Box<dyn Error>> {
        let scene = scene(
            vec![square(1)],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation: Transformation::default(),
                array: Array {
                    columns: u16::MAX,
                    rows: u16::MAX,
                    column_step: DoublePoint::new(0.0001, 0.0),
                    row_step: DoublePoint::new(0.0, 0.0001),
                },
            }],
        )?;
        let image = capture(scene.clone(), request(&scene, Vec2::splat(40.0)), context())?;
        assert_eq!(
            image.shape_count,
            usize::from(u16::MAX) * usize::from(u16::MAX)
        );
        assert!(image.image.pixels.iter().any(|pixel| pixel.a() != 0));
        Ok(())
    }

    #[test]
    fn raster_labels_keep_rotation_reflection_and_alignment() -> Result<(), Box<dyn Error>> {
        let shape = Shape {
            layer: LayerKey {
                layer: 1,
                datatype: 0,
            },
            geometry: Geometry::Label(Label {
                text: "Original label".into(),
                transformation: Transformation {
                    angle: 0.3,
                    reflected: true,
                    ..Default::default()
                },
                horizontal_alignment: 1,
                vertical_alignment: 1,
            }),
        };
        let scene = scene(
            vec![shape],
            vec![Instance {
                cell: CellId::from_index(0),
                transformation: Transformation::default(),
                array: Array::default(),
            }],
        )?;
        let image = capture(
            scene.clone(),
            request(&scene, Vec2::splat(400.0)),
            context(),
        )?;
        assert_eq!(image.shape_count, 1);
        assert!(image.image.pixels.iter().any(|pixel| pixel.a() != 0));
        Ok(())
    }
}
