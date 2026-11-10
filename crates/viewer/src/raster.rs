use super::*;
use crate::spatial::*;

const REFERENCE_MEMORY_LIMIT: usize = 128 * 1024 * 1024;

struct GlyphRequest {
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

pub(super) fn capture(
    scene: Arc<Scene>,
    key: FrameKey,
    context: egui::Context,
) -> Result<RenderedImage, ViewError> {
    if !key.viewport.min.is_finite()
        || !key.viewport.max.is_finite()
        || !key.viewport.width().is_finite()
        || !key.viewport.height().is_finite()
        || !key.pixels_per_point.is_finite()
        || key.viewport.width() <= 0.0
        || key.viewport.height() <= 0.0
        || key.pixels_per_point <= 0.0
    {
        return Err(ViewError("Invalid reference viewport".into()));
    }
    if !key.camera.center.finite()
        || !key.camera.pixels_per_unit.is_finite()
        || key.camera.pixels_per_unit <= 0.0
    {
        return Err(ViewError("Invalid reference camera".into()));
    }
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
        .is_none_or(|pixels| pixels > REFERENCE_MEMORY_LIMIT / 8)
    {
        return Err(ViewError("Viewport exceeds rendering memory budget".into()));
    }

    let layers: Vec<_> = key
        .styles
        .iter()
        .filter_map(|(layer, style)| style.visible.then_some(*layer))
        .collect();
    let group_size = (48 * 1024 * 1024 / (size[0] * size[1])).max(1);
    let mut image = egui::ColorImage::new(size, vec![Color32::TRANSPARENT; size[0] * size[1]]);
    let mut shape_count = 0_usize;
    for group in layers.chunks(group_size) {
        let mut request = key.clone();
        for (layer, style) in &mut request.styles {
            style.visible = group.binary_search(layer).is_ok();
        }
        let mut painter = Rasterizer::new(scene.clone(), request, size, context.clone());
        if let Some(selected) = key.selected {
            painter.paint(selected)?;
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
    Ok(RenderedImage {
        image,
        shape_count,
        cache_bytes: 0,
        statistics: RenderStatus::default(),
    })
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

struct Rasterizer {
    scene: Arc<Scene>,
    key: FrameKey,
    scale: f64,
    size: [usize; 2],
    masks: BTreeMap<LayerKey, Vec<u8>>,
    context: egui::Context,
    count: usize,
}

impl Rasterizer {
    fn new(scene: Arc<Scene>, key: FrameKey, size: [usize; 2], context: egui::Context) -> Self {
        Self {
            scene,
            scale: f64::from(key.pixels_per_point),
            key,
            size,
            masks: BTreeMap::new(),
            context,
            count: 0,
        }
    }

    fn screen(&self, point: DoublePoint) -> DoublePoint {
        let displacement = point
            .minus(self.key.camera.center)
            .times(self.key.camera.pixels_per_unit * self.scale);
        DoublePoint::new(
            f64::from(self.key.viewport.width()) * self.scale * 0.5 + displacement.x,
            f64::from(self.key.viewport.height()) * self.scale * 0.5 - displacement.y,
        )
    }

    fn visible(&self) -> Bounds {
        let minimum =
            self.key.viewport.min + Vec2::new(0.0, (self.size[1] as f64 / self.scale) as f32);
        let maximum =
            self.key.viewport.min + Vec2::new((self.size[0] as f64 / self.scale) as f32, 0.0);
        Bounds {
            minimum: self.key.camera.screen_to_world(minimum, self.key.viewport),
            maximum: self.key.camera.screen_to_world(maximum, self.key.viewport),
        }
    }

    fn paint(&mut self, cell: CellId) -> Result<(), ViewError> {
        let mut stack = vec![Work::Cell(cell, Transformation::default(), 0, 1)];
        let visible = self.visible();
        while let Some(work) = stack.pop() {
            match work {
                Work::Cell(cell, transformation, depth, multiplier) => {
                    if !transformation.valid() {
                        return Err(ViewError("Hierarchy transformation overflow".into()));
                    }
                    let metrics = self
                        .scene
                        .label_metrics
                        .get(cell.index())
                        .copied()
                        .ok_or_else(|| ViewError("Missing reference cell label metrics".into()))?;
                    let culling = visible.expanded(label_margin(
                        metrics,
                        transformation.magnification,
                        1.0 / self.key.camera.pixels_per_unit,
                    ));
                    let Some(_) = self
                        .scene
                        .layout
                        .transformed_bounds(cell, transformation)
                        .filter(|bounds| bounds.intersects(culling))
                    else {
                        continue;
                    };
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
                                    scene
                                        .label_metrics
                                        .get(instance.cell.index())
                                        .copied()
                                        .ok_or_else(|| {
                                            ViewError(
                                                "Missing reference child label metrics".into(),
                                            )
                                        })?,
                                    base.magnification,
                                    1.0 / self.key.camera.pixels_per_unit,
                                )),
                                column_step,
                                row_step,
                                instance.array.columns,
                                instance.array.rows,
                            );
                            let mut multiplier = cursor.multiplier;
                            if column_step == DoublePoint::default() && columns.0 < columns.1 {
                                multiplier =
                                    multiplier.saturating_mul(usize::from(columns.1 - columns.0));
                                columns.1 = columns.0 + 1;
                            }
                            if row_step == DoublePoint::default() && rows.0 < rows.1 {
                                multiplier =
                                    multiplier.saturating_mul(usize::from(rows.1 - rows.0));
                                rows.1 = rows.0 + 1;
                            }
                            let array = ArrayCursor {
                                multiplier,
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
                        if self
                            .key
                            .styles
                            .get(&shape.layer)
                            .copied()
                            .is_some_and(|style| style.visible)
                        {
                            let previous = self.count;
                            self.shape(&shape.geometry, shape.layer, cursor.transformation)
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
                        array.multiplier,
                    );
                    array.column += 1;
                    if array.column >= array.columns.1 {
                        array.column = array.columns.0;
                        array.row += 1;
                    }
                    stack.push(Work::Array(array));
                    stack.push(child);
                }
            }
        }
        Ok(())
    }

    fn shape(
        &mut self,
        geometry: &Geometry,
        layer: LayerKey,
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
        let request = GlyphRequest {
            text: label.text.clone(),
            size: size.to_bits(),
            pixels_per_point: self.key.pixels_per_point.to_bits(),
        };
        let glyph = make_glyph(&self.context, &request)?;
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

fn make_glyph(context: &egui::Context, request: &GlyphRequest) -> Result<Glyph, ViewError> {
    let (galley, atlas) = context.fonts_mut(|fonts| {
        let galley = fonts.layout_no_wrap(
            request.text.clone(),
            egui::FontId::proportional(f64::from_bits(request.size) as f32),
            Color32::WHITE,
        );
        (galley, fonts.image())
    });
    let pixels_per_point = f64::from(f32::from_bits(request.pixels_per_point));
    let size = [
        (f64::from(galley.size().x) * pixels_per_point).ceil() as usize,
        (f64::from(galley.size().y) * pixels_per_point).ceil() as usize,
    ];
    let length = size[0]
        .checked_mul(size[1])
        .filter(|length| *length <= REFERENCE_MEMORY_LIMIT)
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
    fn uncached_reference_visits_all_placements_and_keeps_thin_shapes() -> Result<(), Box<dyn Error>>
    {
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
    fn fractional_arrays_and_coincident_counts_keep_layer_union() -> Result<(), Box<dyn Error>> {
        let color = Color32::from_rgba_unmultiplied(200, 100, 50, 128);
        for (array, count, right) in [
            (
                Array {
                    columns: 3,
                    rows: 1,
                    column_step: DoublePoint::new(0.4, 0.0),
                    row_step: DoublePoint::default(),
                },
                3,
                10,
            ),
            (
                Array {
                    columns: 7,
                    rows: 11,
                    column_step: DoublePoint::default(),
                    row_step: DoublePoint::default(),
                },
                77,
                9,
            ),
        ] {
            let scene = scene(
                vec![Shape {
                    layer: LayerKey {
                        layer: 1,
                        datatype: 0,
                    },
                    geometry: Geometry::Box {
                        minimum: Point::new(0, 0),
                        maximum: Point::new(1, 4),
                    },
                }],
                vec![Instance {
                    cell: CellId::from_index(0),
                    transformation: Transformation::default(),
                    array,
                }],
            )?;
            let mut key = request(&scene, Vec2::splat(16.0));
            for style in key.styles.values_mut() {
                style.color = color;
            }
            let rendered = capture(scene, key, context())?;
            assert_eq!(rendered.shape_count, count);
            assert_eq!(rendered.cache_bytes, 0);
            for (index, pixel) in rendered.image.pixels.iter().enumerate() {
                let x = index % 16;
                let y = index / 16;
                let expected = if (8..right).contains(&x) && (4..8).contains(&y) {
                    color
                } else {
                    Color32::TRANSPARENT
                };
                assert_eq!(*pixel, expected, "Unexpected array pixel at ({x}, {y})");
            }
        }
        Ok(())
    }

    #[test]
    fn uncached_reference_rejects_invalid_viewports() -> Result<(), Box<dyn Error>> {
        let scene = scene(vec![square(1)], vec![])?;
        for size in [Vec2::splat(20_000.0), Vec2::ZERO] {
            assert!(capture(scene.clone(), request(&scene, size), context()).is_err());
        }
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
