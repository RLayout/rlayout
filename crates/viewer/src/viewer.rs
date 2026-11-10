pub use egui;

mod gpu;
mod prepare;
mod spatial;

#[cfg(test)]
mod gpu_tests;
#[cfg(test)]
mod raster;

pub use gpu::{GpuRendererHandle, RenderStatus};

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;
#[cfg(test)]
use std::sync::OnceLock;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[cfg(test)]
use egui::epaint::Mesh;
use egui::{Color32, Pos2, Rect, Vec2};
#[cfg(test)]
use rlayout_core::Path;
use rlayout_core::{
    Bounds, CellId, DoublePoint, Geometry, Label, LayerKey, Layout, Transformation, path_contour,
    shape_bounds,
};
use tess2_rust::{ElementType, TessStatus, Tessellator, WindingRule};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub center: DoublePoint,
    pub pixels_per_unit: f64,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            center: DoublePoint::default(),
            pixels_per_unit: 1.0,
        }
    }
}

impl Camera {
    pub fn screen_to_world(self, point: Pos2, viewport: Rect) -> DoublePoint {
        self.center.plus(DoublePoint::new(
            f64::from(point.x - viewport.center().x) / self.pixels_per_unit,
            -f64::from(point.y - viewport.center().y) / self.pixels_per_unit,
        ))
    }

    pub fn world_to_screen(self, point: DoublePoint, viewport: Rect) -> Pos2 {
        let displacement = point.minus(self.center).times(self.pixels_per_unit);
        Pos2::new(
            viewport.center().x + displacement.x as f32,
            viewport.center().y - displacement.y as f32,
        )
    }

    pub fn fit(&mut self, bounds: Option<Bounds>, viewport: Rect) {
        if let Some(bounds) = bounds {
            self.center = bounds.center();
            self.pixels_per_unit = ((f64::from(viewport.width()) / bounds.width().max(1.0))
                .min(f64::from(viewport.height()) / bounds.height().max(1.0))
                * 0.9)
                .clamp(1e-12, 1e12);
        } else {
            *self = Self::default();
        }
    }

    pub fn zoom_at(&mut self, point: Pos2, viewport: Rect, factor: f64) {
        if !factor.is_finite() || factor <= 0.0 {
            return;
        }
        let before = self.screen_to_world(point, viewport);
        self.pixels_per_unit = (self.pixels_per_unit * factor).clamp(1e-12, 1e12);
        let after = self.screen_to_world(point, viewport);
        self.center = self.center.plus(before.minus(after));
    }

    pub fn pan(&mut self, displacement: Vec2) {
        self.center = self.center.plus(DoublePoint::new(
            -f64::from(displacement.x) / self.pixels_per_unit,
            f64::from(displacement.y) / self.pixels_per_unit,
        ));
    }

    pub fn world_bounds(self, viewport: Rect) -> Bounds {
        Bounds {
            minimum: self.screen_to_world(viewport.left_bottom(), viewport),
            maximum: self.screen_to_world(viewport.right_top(), viewport),
        }
    }
}

#[derive(Debug)]
pub struct ViewError(pub String);
impl fmt::Display for ViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
impl Error for ViewError {}

#[derive(Debug)]
struct Polygon {
    points: Vec<DoublePoint>,
    triangles: Vec<u32>,
    bounds: Bounds,
}

#[cfg(test)]
#[derive(Debug)]
enum PreparedGeometry {
    Polygon(Polygon),
    AbsolutePath(Path),
    Line(Vec<DoublePoint>),
    Label(Label),
}

#[cfg(test)]
#[derive(Debug)]
struct PreparedShape {
    layer: LayerKey,
    geometry: PreparedGeometry,
}

#[derive(Debug)]
pub struct Scene {
    layout: Layout,
    revision: u64,
    #[cfg(test)]
    shapes: HashMap<CellId, OnceLock<Vec<PreparedShape>>>,
    indexes: HashMap<CellId, spatial::SpatialIndex>,
    label_metrics: Vec<(f64, f64)>,
    #[cfg(test)]
    label_margin: f64,
}

impl Scene {
    pub fn prepare(layout: Layout) -> Result<Self, ViewError> {
        static NEXT_REVISION: AtomicU64 = AtomicU64::new(1);
        let revision = NEXT_REVISION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |revision| {
                revision.checked_add(1)
            })
            .map_err(|_| ViewError("Too many document revisions".into()))?;
        #[cfg(test)]
        let mut shapes = HashMap::new();
        let mut indexes = HashMap::new();
        let mut label_metrics = vec![(0.0_f64, 0.0_f64); layout.cells().count()];
        #[cfg(test)]
        let mut label_margin = 0.0_f64;
        for (id, cell) in layout.cells() {
            #[cfg(test)]
            shapes.insert(id, OnceLock::new());
            for shape in cell.shapes() {
                if let Geometry::Label(label) = &shape.geometry {
                    label_metrics[id.index()].0 = label_metrics[id.index()]
                        .0
                        .max(label.text.chars().count() as f64);
                    label_metrics[id.index()].1 = label_metrics[id.index()]
                        .1
                        .max(label.transformation.magnification);
                    #[cfg(test)]
                    {
                        label_margin =
                            label_margin.max(label.text.chars().count() as f64 * 96.0 + 96.0);
                    }
                }
            }
        }
        loop {
            let mut changed = false;
            for (id, cell) in layout.cells() {
                for instance in cell.instances() {
                    let child = label_metrics[instance.cell.index()];
                    let previous = label_metrics[id.index()];
                    let updated = (
                        previous.0.max(child.0),
                        previous
                            .1
                            .max(child.1 * instance.transformation.magnification),
                    );
                    if updated != previous {
                        label_metrics[id.index()] = updated;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for (id, _) in layout.cells() {
            indexes.insert(id, spatial::SpatialIndex::new(&layout, id, &label_metrics)?);
        }
        Ok(Self {
            layout,
            revision,
            #[cfg(test)]
            shapes,
            indexes,
            label_metrics,
            #[cfg(test)]
            label_margin,
        })
    }

    #[cfg(test)]
    fn prepared(&self, id: CellId) -> Result<&[PreparedShape], ViewError> {
        let destination = self
            .shapes
            .get(&id)
            .ok_or_else(|| ViewError("Missing prepared cell".into()))?;
        if destination.get().is_none() {
            let cell = self
                .layout
                .cell(id)
                .ok_or_else(|| ViewError("Missing cell".into()))?;
            let mut prepared = Vec::with_capacity(cell.shapes().len());
            for (index, shape) in cell.shapes().iter().enumerate() {
                let geometry = match &shape.geometry {
                    Geometry::Polygon(points) => {
                        triangulate(points.iter().map(|point| point.to_double()).collect())
                            .map(PreparedGeometry::Polygon)
                    }
                    Geometry::Box { minimum, maximum } => triangulate(vec![
                        minimum.to_double(),
                        DoublePoint::new(maximum.x as f64, minimum.y as f64),
                        maximum.to_double(),
                        DoublePoint::new(minimum.x as f64, maximum.y as f64),
                    ])
                    .map(PreparedGeometry::Polygon),
                    Geometry::Path(path) if path.width == 0 => Ok(PreparedGeometry::Line(
                        path.points.iter().map(|point| point.to_double()).collect(),
                    )),
                    Geometry::Path(path) if path.absolute_width => {
                        Ok(PreparedGeometry::AbsolutePath(path.clone()))
                    }
                    Geometry::Path(path) => {
                        triangulate_path(path, 1.0).map(PreparedGeometry::Polygon)
                    }
                    Geometry::Label(label) => Ok(PreparedGeometry::Label(label.clone())),
                }
                .map_err(|error| shape_error(cell.name(), shape.layer, index, error))?;
                prepared.push(PreparedShape {
                    layer: shape.layer,
                    geometry,
                });
            }
            if destination.set(prepared).is_err() && destination.get().is_none() {
                return Err(ViewError("Preparing cell failed".into()));
            }
        }
        destination
            .get()
            .map(|shapes| shapes.as_slice())
            .ok_or_else(|| ViewError("Missing prepared shapes".into()))
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

fn triangulate(points: Vec<DoublePoint>) -> Result<Polygon, ViewError> {
    if points.len() < 3 || points.iter().any(|point| !point.finite()) {
        return Err(ViewError(
            "Polygon requires at least three finite vertices".into(),
        ));
    }
    let bounds = Bounds::from_points(points.iter().copied())
        .ok_or_else(|| ViewError("Empty polygon bounds".into()))?;
    let origin = points
        .first()
        .copied()
        .ok_or_else(|| ViewError("Empty polygon".into()))?;
    let vertices: Vec<_> = points
        .iter()
        .flat_map(|point| [point.x - origin.x, point.y - origin.y])
        .collect();
    if vertices.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(ViewError("Polygon coordinate overflow".into()));
    }
    let mut tessellator = Tessellator::new();
    tessellator.add_contour(2, &vertices);
    if tessellator.get_status() != TessStatus::Ok
        || !tessellator.tessellate(
            WindingRule::NonZero,
            ElementType::Polygons,
            3,
            2,
            Some([0.0, 0.0, 1.0]),
        )
    {
        return Err(ViewError(format!(
            "Polygon tessellation failed: {:?}",
            tessellator.get_status()
        )));
    }
    let (vertices, remainder) = tessellator.vertices().as_chunks::<2>();
    if !remainder.is_empty() || !tessellator.elements().len().is_multiple_of(3) {
        return Err(ViewError(
            "Polygon tessellation produced incomplete vertices or triangles".into(),
        ));
    }
    let points = vertices
        .iter()
        .map(|coordinates| {
            let point = DoublePoint::new(coordinates[0], coordinates[1]).plus(origin);
            if !point.finite() {
                return Err(ViewError("Non-finite tessellated polygon vertex".into()));
            }
            Ok(point)
        })
        .collect::<Result<Vec<_>, ViewError>>()?;
    let triangles = tessellator.elements().to_vec();
    for index in &triangles {
        let index = usize::try_from(*index)
            .map_err(|_| ViewError("Polygon tessellation index overflow".into()))?;
        if points.get(index).is_none() {
            return Err(ViewError("Invalid polygon tessellation index".into()));
        }
    }
    Ok(Polygon {
        points,
        triangles,
        bounds,
    })
}

#[cfg(test)]
fn triangulate_path(path: &Path, magnification: f64) -> Result<Polygon, ViewError> {
    let points = path_contour(path, magnification).map_err(|error| ViewError(error.to_string()))?;
    if points.len() < 3 {
        return Ok(Polygon {
            points: Vec::new(),
            triangles: Vec::new(),
            bounds: Bounds::from_points(path.points.iter().map(|point| point.to_double()))
                .ok_or_else(|| ViewError("Empty path".into()))?,
        });
    }
    triangulate(points)
}

fn shape_error(cell: &str, layer: LayerKey, index: usize, error: ViewError) -> ViewError {
    ViewError(format!(
        "Cell {cell}, layer {}/{}, shape {}: {error}",
        layer.layer,
        layer.datatype,
        index + 1
    ))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerStyle {
    pub visible: bool,
    pub color: Color32,
}

#[cfg(test)]
pub struct RenderedTile {
    pub rect: Rect,
    pub texture: egui::TextureId,
    pub preview: bool,
}

pub struct RenderedImage {
    pub image: egui::ColorImage,
    pub shape_count: usize,
    pub cache_bytes: usize,
    pub statistics: RenderStatus,
}

#[cfg(test)]
#[derive(Default)]
pub struct RenderedFrame {
    pub meshes: Vec<Arc<Mesh>>,
    pub lines: Vec<egui::Shape>,
    pub shape_count: usize,
    pub truncated: bool,
    pub complete: bool,
    pub tiles: Vec<RenderedTile>,
}

#[derive(Clone, PartialEq)]
struct FrameKey {
    camera: Camera,
    viewport: Rect,
    pixels_per_point: f32,
    selected: Option<CellId>,
    depth: u16,
    styles: BTreeMap<LayerKey, LayerStyle>,
    text_pass: u64,
}

pub struct Viewer {
    scene: Arc<Scene>,
    pub camera: Camera,
    pub selected: Option<CellId>,
    pub hierarchy_depth: u16,
    pub styles: BTreeMap<LayerKey, LayerStyle>,
    fit_pending: bool,
    #[cfg(test)]
    absolute_paths: HashMap<(CellId, usize, u64), Polygon>,
    pub cursor: Option<DoublePoint>,
    pub error: Option<String>,
    renderer: gpu::RenderSession,
    requested: Option<FrameKey>,
    changed_at: Instant,
    exact_requested: bool,
}

impl Viewer {
    pub fn new(scene: Scene, renderer: GpuRendererHandle) -> Result<Self, ViewError> {
        let selected = scene.layout.top_cells().first().copied();
        let palette = [
            Color32::from_rgb(91, 174, 229),
            Color32::from_rgb(231, 139, 96),
            Color32::from_rgb(119, 198, 136),
            Color32::from_rgb(194, 145, 229),
            Color32::from_rgb(230, 210, 112),
            Color32::from_rgb(98, 201, 199),
        ];
        let styles = scene
            .layout
            .layers()
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let color = palette
                    .get(index % palette.len())
                    .copied()
                    .unwrap_or(Color32::WHITE);
                (
                    *layer,
                    LayerStyle {
                        visible: true,
                        color,
                    },
                )
            })
            .collect();
        let scene = Arc::new(scene);
        let renderer = renderer.create_session(scene.clone())?;
        Ok(Self {
            scene,
            camera: Camera::default(),
            selected,
            hierarchy_depth: u16::MAX,
            styles,
            fit_pending: true,
            #[cfg(test)]
            absolute_paths: HashMap::new(),
            cursor: None,
            error: None,
            renderer,
            requested: None,
            changed_at: Instant::now(),
            exact_requested: false,
        })
    }
    pub fn layout(&self) -> &Layout {
        &self.scene.layout
    }
    pub fn request_fit(&mut self) {
        self.fit_pending = true;
    }
    pub fn activate(&mut self, id: CellId) {
        if self.scene.layout.cell(id).is_some() {
            self.selected = Some(id);
            self.request_fit();
        }
    }

    pub fn panels(&mut self, ui: &mut egui::Ui) {
        let mut activated = None;
        egui::Panel::left("cells")
            .default_size(220.0)
            .show(ui, |ui| {
                ui.heading("Cells");
                ui.label(format!(
                    "{} cells · {} shapes",
                    self.scene.layout.cells().count(),
                    self.scene.layout.shape_count()
                ));
                ui.horizontal(|ui| {
                    let mut all = self.hierarchy_depth == u16::MAX;
                    if ui.checkbox(&mut all, "All levels").changed() {
                        self.hierarchy_depth = if all { u16::MAX } else { 1 };
                    }
                    if !all {
                        ui.add(egui::DragValue::new(&mut self.hierarchy_depth).range(0..=64));
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut count = 0;
                    for id in self.scene.layout.top_cells() {
                        hierarchy_row(
                            ui,
                            &self.scene.layout,
                            *id,
                            self.selected,
                            &mut activated,
                            &mut count,
                            0,
                            id.index() as u64,
                        );
                    }
                    if count >= 2000 {
                        ui.label("Expand fewer branches to see more cells.");
                    }
                });
            });
        if let Some(id) = activated {
            self.activate(id);
        }
        egui::Panel::right("layers")
            .default_size(190.0)
            .show(ui, |ui| {
                ui.heading("Layers");
                ui.horizontal(|ui| {
                    if ui.button("Show all").clicked() {
                        for style in self.styles.values_mut() {
                            style.visible = true;
                        }
                    }
                    if ui.button("Hide all").clicked() {
                        for style in self.styles.values_mut() {
                            style.visible = false;
                        }
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (layer, style) in &mut self.styles {
                        ui.horizontal(|ui| {
                            ui.checkbox(
                                &mut style.visible,
                                format!("{}/{}", layer.layer, layer.datatype),
                            );
                            ui.color_edit_button_srgba(&mut style.color);
                        });
                    }
                });
            });
    }

    pub fn canvas(&mut self, ui: &mut egui::Ui) {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());
        let viewport = response.rect;
        painter.rect_filled(viewport, 0.0, Color32::from_rgb(17, 21, 27));
        if viewport.width() <= 0.0 || viewport.height() <= 0.0 {
            return;
        }
        if self.fit_pending {
            self.camera.fit(
                self.selected.and_then(|id| self.scene.layout.bounds(id)),
                viewport,
            );
            self.fit_pending = false;
        }
        if response.hovered() {
            let (scroll, position, middle_down, displacement) = ui.input(|input| {
                (
                    input.smooth_scroll_delta.y,
                    input.pointer.hover_pos(),
                    input.pointer.button_down(egui::PointerButton::Middle),
                    input.pointer.delta(),
                )
            });
            if middle_down {
                self.camera.pan(displacement);
            }
            if scroll != 0.0
                && let Some(position) = position
            {
                self.camera
                    .zoom_at(position, viewport, (f64::from(scroll) * 0.005).exp());
            }
        }
        self.cursor = response
            .hover_pos()
            .map(|position| self.camera.screen_to_world(position, viewport));
        let key = FrameKey {
            camera: self.camera,
            viewport,
            pixels_per_point: ui.ctx().pixels_per_point(),
            selected: self.selected,
            depth: self.hierarchy_depth,
            styles: self.styles.clone(),
            text_pass: 0,
        };
        if self.requested.as_ref() != Some(&key) {
            self.requested = Some(key.clone());
            self.changed_at = Instant::now();
            self.exact_requested = false;
            self.error = self
                .renderer
                .request(key.clone(), ui.ctx().clone(), true)
                .err()
                .map(|error| error.to_string());
        }
        let quiet = self.changed_at.elapsed();
        if !self.exact_requested {
            ui.ctx()
                .request_repaint_after(Duration::from_millis(150).saturating_sub(quiet));
        }
        if !self.exact_requested && quiet >= Duration::from_millis(150) {
            self.exact_requested = true;
            self.error = self
                .renderer
                .request(key, ui.ctx().clone(), false)
                .err()
                .map(|error| error.to_string());
        }
        painter.add(self.renderer.callback(viewport, self.camera));
        let status = self.renderer.status();
        if let Some(error) = status.error {
            self.error = Some(error);
        }
        if !status.complete && self.error.is_none() {
            ui.ctx().request_repaint_after(Duration::from_millis(16));
        }
        if self.selected.is_none() {
            painter.text(
                viewport.center(),
                egui::Align2::CENTER_CENTER,
                "This library has no cells",
                egui::FontId::proportional(16.0),
                Color32::GRAY,
            );
        }
        if let Some(error) = &self.error {
            painter.text(
                viewport.left_top() + Vec2::splat(12.0),
                egui::Align2::LEFT_TOP,
                error,
                egui::FontId::proportional(14.0),
                Color32::LIGHT_RED,
            );
        }
    }
    pub fn rendering_complete(&self) -> bool {
        self.exact_requested && self.renderer.status().complete
    }

    pub fn render_status(&self) -> RenderStatus {
        self.renderer.status()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn render_image(
        &mut self,
        context: &egui::Context,
        viewport: Rect,
    ) -> Result<RenderedImage, ViewError> {
        self.renderer
            .capture(
                FrameKey {
                    camera: self.camera,
                    viewport,
                    pixels_per_point: context.pixels_per_point(),
                    selected: self.selected,
                    depth: self.hierarchy_depth,
                    styles: self.styles.clone(),
                    text_pass: 0,
                },
                context.clone(),
            )
            .await
    }

    pub fn visible_shape_count(&self) -> usize {
        self.renderer.status().shape_count
    }

    #[cfg(test)]
    pub fn render(
        &mut self,
        context: &egui::Context,
        viewport: Rect,
    ) -> Result<RenderedFrame, ViewError> {
        let mut result = RenderedFrame {
            complete: true,
            ..Default::default()
        };
        let Some(selected) = self.selected else {
            return Ok(result);
        };
        let visible = self.camera.world_bounds(viewport);
        let culling_bounds =
            visible.expanded(self.scene.label_margin / self.camera.pixels_per_unit);
        let scene = self.scene.clone();
        let mut stack = vec![(selected, Transformation::default(), 0_u16)];
        let mut batches: BTreeMap<LayerKey, Mesh> = BTreeMap::new();
        let mut text_batches: BTreeMap<LayerKey, Mesh> = BTreeMap::new();
        while let Some((id, transformation, depth)) = stack.pop() {
            if !scene
                .layout
                .transformed_bounds(id, transformation)
                .is_some_and(|bounds| bounds.intersects(culling_bounds))
            {
                continue;
            }
            if !transformation.valid() {
                return Err(ViewError("Hierarchy transformation overflow".into()));
            }
            {
                let shapes = scene.prepared(id)?;
                for (index, shape) in shapes.iter().enumerate() {
                    let Some(style) = self.styles.get(&shape.layer).filter(|style| style.visible)
                    else {
                        continue;
                    };
                    let mesh = batches.entry(shape.layer).or_default();
                    let drawn = match &shape.geometry {
                        PreparedGeometry::Polygon(polygon) => paint_polygon(
                            mesh,
                            polygon,
                            transformation,
                            self.camera,
                            viewport,
                            visible,
                            style.color,
                        )?,
                        PreparedGeometry::AbsolutePath(path) => {
                            let cache_key = (id, index, transformation.magnification.to_bits());
                            if let std::collections::hash_map::Entry::Vacant(entry) =
                                self.absolute_paths.entry(cache_key)
                            {
                                entry.insert(
                                    triangulate_path(path, transformation.magnification).map_err(
                                        |error| {
                                            shape_error(
                                                scene
                                                    .layout
                                                    .cell(id)
                                                    .map_or("<unknown>", |cell| cell.name()),
                                                shape.layer,
                                                index,
                                                error,
                                            )
                                        },
                                    )?,
                                );
                            }
                            let polygon = self
                                .absolute_paths
                                .get(&cache_key)
                                .ok_or_else(|| ViewError("Missing path tessellation".into()))?;
                            paint_polygon(
                                mesh,
                                polygon,
                                transformation,
                                self.camera,
                                viewport,
                                visible,
                                style.color,
                            )?
                        }
                        PreparedGeometry::Line(points) => {
                            if Bounds::from_points(
                                points.iter().map(|point| transformation.apply(*point)),
                            )
                            .is_some_and(|bounds| bounds.intersects(visible))
                            {
                                result.lines.push(egui::Shape::line(
                                    points
                                        .iter()
                                        .map(|point| {
                                            self.camera.world_to_screen(
                                                transformation.apply(*point),
                                                viewport,
                                            )
                                        })
                                        .collect(),
                                    egui::Stroke::new(1.0, style.color),
                                ));
                                true
                            } else {
                                false
                            }
                        }
                        PreparedGeometry::Label(label) => paint_label(
                            text_batches.entry(shape.layer).or_default(),
                            context,
                            label,
                            transformation,
                            self.camera,
                            viewport,
                            style.color,
                        )?,
                    };
                    if drawn {
                        result.shape_count += 1;
                    }
                }
            }
            if result.truncated {
                break;
            }
            if depth < self.hierarchy_depth
                && let Some(cell) = scene.layout.cell(id)
            {
                for instance in cell.instances().iter().rev() {
                    let base = transformation.compose(instance.transformation);
                    let Some(bounds) = scene.layout.transformed_bounds(instance.cell, base) else {
                        continue;
                    };
                    let column_step = transformation.apply_vector(instance.array.column_step);
                    let row_step = transformation.apply_vector(instance.array.row_step);
                    let (columns, rows) = visible_array_range(
                        bounds,
                        culling_bounds,
                        column_step,
                        row_step,
                        instance.array.columns,
                        instance.array.rows,
                    );
                    for row in rows.0..rows.1 {
                        for column in columns.0..columns.1 {
                            let translation = base
                                .translation
                                .plus(column_step.times(f64::from(column)))
                                .plus(row_step.times(f64::from(row)));
                            let child = Transformation {
                                translation,
                                ..base
                            };
                            if bounds
                                .translated(translation.minus(base.translation))
                                .intersects(culling_bounds)
                            {
                                stack.push((instance.cell, child, depth.saturating_add(1)));
                            }
                        }
                        if result.truncated {
                            break;
                        }
                    }
                    if result.truncated {
                        break;
                    }
                }
            }
        }
        result.meshes = batches
            .into_values()
            .filter(|mesh| !mesh.vertices.is_empty())
            .map(Arc::new)
            .collect();
        if !text_batches.is_empty() {
            let size = context.fonts(|fonts| fonts.font_image_size());
            for mut mesh in text_batches.into_values() {
                for vertex in &mut mesh.vertices {
                    // Galley UVs are texels; normalize after all labels can grow the font atlas.
                    vertex.uv.x /= size[0] as f32;
                    vertex.uv.y /= size[1] as f32;
                }
                if !mesh.vertices.is_empty() {
                    result.meshes.push(Arc::new(mesh));
                }
            }
        }
        Ok(result)
    }
}

#[allow(clippy::too_many_arguments)]
fn hierarchy_row(
    ui: &mut egui::Ui,
    layout: &Layout,
    id: CellId,
    selected: Option<CellId>,
    activated: &mut Option<CellId>,
    count: &mut usize,
    depth: usize,
    path: u64,
) {
    if *count >= 2000 {
        return;
    }
    *count += 1;
    let Some(cell) = layout.cell(id) else {
        return;
    };
    let name = egui::RichText::new(cell.name()).color(if selected == Some(id) {
        Color32::LIGHT_BLUE
    } else {
        ui.visuals().text_color()
    });
    if cell.instances().is_empty() {
        if ui.selectable_label(selected == Some(id), name).clicked() {
            *activated = Some(id);
        }
    } else {
        let response = egui::CollapsingHeader::new(name)
            .id_salt(path)
            .show(ui, |ui| {
                if ui.small_button("View this cell").clicked() {
                    *activated = Some(id);
                }
                if depth >= 64 {
                    ui.label("Hierarchy display depth limit reached.");
                    return;
                }
                for (index, instance) in cell.instances().iter().enumerate() {
                    if *count >= 2000 {
                        break;
                    }
                    if instance.array.columns > 1 || instance.array.rows > 1 {
                        ui.label(format!(
                            "Array {} × {}",
                            instance.array.columns, instance.array.rows
                        ));
                    }
                    hierarchy_row(
                        ui,
                        layout,
                        instance.cell,
                        selected,
                        activated,
                        count,
                        depth + 1,
                        path.wrapping_mul(31).wrapping_add(index as u64 + 1),
                    );
                }
            });
        if response.header_response.double_clicked() {
            *activated = Some(id);
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn paint_polygon(
    mesh: &mut Mesh,
    polygon: &Polygon,
    transformation: Transformation,
    camera: Camera,
    viewport: Rect,
    visible: Bounds,
    color: Color32,
) -> Result<bool, ViewError> {
    if polygon.triangles.is_empty()
        || !polygon
            .bounds
            .transformed(transformation)
            .intersects(visible)
    {
        return Ok(false);
    }
    let offset = u32::try_from(mesh.vertices.len())
        .map_err(|_| ViewError("Too many displayed vertices".into()))?;
    for point in &polygon.points {
        let position = camera.world_to_screen(transformation.apply(*point), viewport);
        if !position.is_finite() {
            return Err(ViewError("Screen coordinate overflow".into()));
        }
        mesh.vertices.push(egui::epaint::Vertex {
            pos: position,
            uv: egui::epaint::WHITE_UV,
            color,
        });
    }
    for index in &polygon.triangles {
        mesh.indices.push(
            offset
                .checked_add(*index)
                .ok_or_else(|| ViewError("Display index overflow".into()))?,
        );
    }
    Ok(true)
}

#[cfg(test)]
fn paint_label(
    mesh: &mut Mesh,
    context: &egui::Context,
    label: &Label,
    parent: Transformation,
    camera: Camera,
    viewport: Rect,
    color: Color32,
) -> Result<bool, ViewError> {
    let transformation = parent.compose(label.transformation);
    let origin = camera.world_to_screen(transformation.translation, viewport);
    let font_size = (12.0 * transformation.magnification).clamp(6.0, 96.0) as f32;
    let galley = context.fonts_mut(|fonts| {
        fonts.layout_no_wrap(
            label.text.clone(),
            egui::FontId::proportional(font_size),
            color,
        )
    });
    let anchor = Vec2::new(
        galley.size().x * f32::from(label.horizontal_alignment) * 0.5,
        galley.size().y * f32::from(label.vertical_alignment) * 0.5,
    );
    let (sine, cosine) = transformation.angle.sin_cos();
    let mut label_mesh = Mesh::default();
    for row in &galley.rows {
        let mut row_mesh = row.visuals.mesh.clone();
        for vertex in &mut row_mesh.vertices {
            let point = vertex.pos.to_vec2() + row.pos.to_vec2() - anchor;
            let x = f64::from(point.x);
            let y = f64::from(point.y) * if transformation.reflected { -1.0 } else { 1.0 };
            vertex.pos = origin
                + Vec2::new(
                    (cosine * x + sine * y) as f32,
                    (-sine * x + cosine * y) as f32,
                );
        }
        label_mesh.append(row_mesh);
    }
    if !label_mesh.calc_bounds().intersects(viewport) {
        return Ok(false);
    }
    if mesh
        .vertices
        .len()
        .checked_add(label_mesh.vertices.len())
        .is_none_or(|length| length > u32::MAX as usize)
    {
        return Err(ViewError("Too many text vertices".into()));
    }
    mesh.append(label_mesh);
    Ok(true)
}

pub fn visible_array_range(
    element: Bounds,
    viewport: Bounds,
    column: DoublePoint,
    row: DoublePoint,
    columns: u16,
    rows: u16,
) -> ((u16, u16), (u16, u16)) {
    let target = Bounds {
        minimum: viewport.minimum.minus(element.maximum),
        maximum: viewport.maximum.minus(element.minimum),
    };
    let determinant = column.x * row.y - column.y * row.x;
    let corners = [
        target.minimum,
        target.maximum,
        DoublePoint::new(target.minimum.x, target.maximum.y),
        DoublePoint::new(target.maximum.x, target.minimum.y),
    ];
    if determinant.abs() > column.x.hypot(column.y) * row.x.hypot(row.y) * 1e-12 {
        let indices = Bounds::from_points(corners.into_iter().map(|point| {
            DoublePoint::new(
                (point.x * row.y - point.y * row.x) / determinant,
                (column.x * point.y - column.y * point.x) / determinant,
            )
        }));
        if let Some(indices) = indices {
            return (
                index_range(indices.minimum.x, indices.maximum.x, columns),
                index_range(indices.minimum.y, indices.maximum.y, rows),
            );
        }
    }
    (
        projected_range(target, column, row, columns, rows),
        projected_range(target, row, column, rows, columns),
    )
}

fn index_range(minimum: f64, maximum: f64, count: u16) -> (u16, u16) {
    let start = (minimum - 1e-8).ceil().clamp(0.0, f64::from(count)) as u16;
    let end = ((maximum + 1e-8).floor() + 1.0).clamp(0.0, f64::from(count)) as u16;
    (start.min(end), end)
}

fn projected_range(
    target: Bounds,
    step: DoublePoint,
    other: DoublePoint,
    count: u16,
    other_count: u16,
) -> (u16, u16) {
    if step.x == 0.0 && step.y == 0.0 {
        return (0, count);
    }
    let (minimum, maximum, component, other_component) = if step.x.abs() >= step.y.abs() {
        (target.minimum.x, target.maximum.x, step.x, other.x)
    } else {
        (target.minimum.y, target.maximum.y, step.y, other.y)
    };
    let other_extent = other_component * f64::from(other_count.saturating_sub(1));
    let first = (minimum - other_extent.max(0.0)) / component;
    let last = (maximum - other_extent.min(0.0)) / component;
    index_range(first.min(last), first.max(last), count)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
fn test_renderer() -> Result<GpuRendererHandle, ViewError> {
    static RENDERER: std::sync::OnceLock<Result<GpuRendererHandle, String>> =
        std::sync::OnceLock::new();
    RENDERER
        .get_or_init(|| {
            pollster::block_on(GpuRendererHandle::headless()).map_err(|error| error.to_string())
        })
        .as_ref()
        .cloned()
        .map_err(|error| ViewError(error.clone()))
}
