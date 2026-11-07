use std::collections::{BTreeSet, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CellId(usize);

impl CellId {
    pub fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LayerKey {
    pub layer: u16,
    pub datatype: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Point {
    pub x: i64,
    pub y: i64,
}

impl Point {
    pub fn new(x: i64, y: i64) -> Self {
        Self { x, y }
    }

    pub fn to_double(self) -> DoublePoint {
        DoublePoint::new(self.x as f64, self.y as f64)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DoublePoint {
    pub x: f64,
    pub y: f64,
}

impl DoublePoint {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    pub fn plus(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    pub fn minus(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    pub fn times(self, factor: f64) -> Self {
        Self::new(self.x * factor, self.y * factor)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub minimum: DoublePoint,
    pub maximum: DoublePoint,
}

impl Bounds {
    pub fn from_points(points: impl IntoIterator<Item = DoublePoint>) -> Option<Self> {
        let mut points = points.into_iter();
        let first = points.next()?;
        let mut bounds = Self {
            minimum: first,
            maximum: first,
        };
        for point in points {
            bounds.minimum.x = bounds.minimum.x.min(point.x);
            bounds.minimum.y = bounds.minimum.y.min(point.y);
            bounds.maximum.x = bounds.maximum.x.max(point.x);
            bounds.maximum.y = bounds.maximum.y.max(point.y);
        }
        Some(bounds)
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            minimum: DoublePoint::new(
                self.minimum.x.min(other.minimum.x),
                self.minimum.y.min(other.minimum.y),
            ),
            maximum: DoublePoint::new(
                self.maximum.x.max(other.maximum.x),
                self.maximum.y.max(other.maximum.y),
            ),
        }
    }

    pub fn expanded(self, distance: f64) -> Self {
        Self {
            minimum: self.minimum.minus(DoublePoint::new(distance, distance)),
            maximum: self.maximum.plus(DoublePoint::new(distance, distance)),
        }
    }

    pub fn translated(self, displacement: DoublePoint) -> Self {
        Self {
            minimum: self.minimum.plus(displacement),
            maximum: self.maximum.plus(displacement),
        }
    }

    pub fn intersects(self, other: Self) -> bool {
        self.minimum.x <= other.maximum.x
            && self.maximum.x >= other.minimum.x
            && self.minimum.y <= other.maximum.y
            && self.maximum.y >= other.minimum.y
    }

    pub fn center(self) -> DoublePoint {
        self.minimum.times(0.5).plus(self.maximum.times(0.5))
    }

    pub fn width(self) -> f64 {
        self.maximum.x - self.minimum.x
    }
    pub fn height(self) -> f64 {
        self.maximum.y - self.minimum.y
    }

    pub fn transformed(self, transformation: Transformation) -> Self {
        let points = [
            self.minimum,
            DoublePoint::new(self.minimum.x, self.maximum.y),
            self.maximum,
            DoublePoint::new(self.maximum.x, self.minimum.y),
        ];
        let first = transformation.apply(points[0]);
        points.into_iter().skip(1).fold(
            Self {
                minimum: first,
                maximum: first,
            },
            |bounds, point| {
                let point = transformation.apply(point);
                bounds.union(Self {
                    minimum: point,
                    maximum: point,
                })
            },
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transformation {
    pub translation: DoublePoint,
    pub angle: f64,
    pub magnification: f64,
    pub reflected: bool,
}

impl Default for Transformation {
    fn default() -> Self {
        Self {
            translation: DoublePoint::default(),
            angle: 0.0,
            magnification: 1.0,
            reflected: false,
        }
    }
}

impl Transformation {
    pub fn apply(self, point: DoublePoint) -> DoublePoint {
        self.apply_vector(point).plus(self.translation)
    }

    pub fn apply_vector(self, point: DoublePoint) -> DoublePoint {
        let point = DoublePoint::new(point.x, if self.reflected { -point.y } else { point.y });
        let (sine, cosine) = self.angle.sin_cos();
        DoublePoint::new(
            cosine * point.x - sine * point.y,
            sine * point.x + cosine * point.y,
        )
        .times(self.magnification)
    }

    pub fn compose(self, child: Self) -> Self {
        Self {
            translation: self.apply(child.translation),
            angle: self.angle
                + if self.reflected {
                    -child.angle
                } else {
                    child.angle
                },
            magnification: self.magnification * child.magnification,
            reflected: self.reflected != child.reflected,
        }
    }

    pub fn valid(self) -> bool {
        self.translation.finite()
            && self.angle.is_finite()
            && self.magnification.is_finite()
            && self.magnification > 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Array {
    pub columns: u16,
    pub rows: u16,
    pub column_step: DoublePoint,
    pub row_step: DoublePoint,
}

impl Default for Array {
    fn default() -> Self {
        Self {
            columns: 1,
            rows: 1,
            column_step: DoublePoint::default(),
            row_step: DoublePoint::default(),
        }
    }
}

impl Array {
    pub fn offset(self, column: u16, row: u16) -> DoublePoint {
        self.column_step
            .times(f64::from(column))
            .plus(self.row_step.times(f64::from(row)))
    }

    pub fn bounds(self, element: Bounds) -> Bounds {
        let last_column = self.columns.saturating_sub(1);
        let last_row = self.rows.saturating_sub(1);
        [
            (0, 0),
            (last_column, 0),
            (0, last_row),
            (last_column, last_row),
        ]
        .into_iter()
        .fold(element, |bounds, (column, row)| {
            bounds.union(element.translated(self.offset(column, row)))
        })
    }
}

#[derive(Clone, Debug)]
pub struct Instance {
    pub cell: CellId,
    pub transformation: Transformation,
    pub array: Array,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathEnds {
    Flush,
    Round,
    Square,
    Extended { beginning: i64, end: i64 },
}

#[derive(Clone, Debug)]
pub struct Path {
    pub points: Vec<Point>,
    pub width: u64,
    pub absolute_width: bool,
    pub ends: PathEnds,
}

#[derive(Clone, Debug)]
pub struct Label {
    pub text: String,
    pub transformation: Transformation,
    pub horizontal_alignment: u8,
    pub vertical_alignment: u8,
}

#[derive(Clone, Debug)]
pub enum Geometry {
    Polygon(Vec<Point>),
    Box { minimum: Point, maximum: Point },
    Path(Path),
    Label(Label),
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub layer: LayerKey,
    pub geometry: Geometry,
}

#[derive(Clone, Debug)]
pub struct Cell {
    name: String,
    shapes: Vec<Shape>,
    instances: Vec<Instance>,
}

impl Cell {
    pub fn new(name: String, shapes: Vec<Shape>, instances: Vec<Instance>) -> Self {
        Self {
            name,
            shapes,
            instances,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn shapes(&self) -> &[Shape] {
        &self.shapes
    }
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }
}

#[derive(Debug)]
pub struct Layout {
    name: String,
    meters_per_unit: f64,
    user_units_per_unit: f64,
    cells: Vec<Cell>,
    top_cells: Vec<CellId>,
    layers: BTreeSet<LayerKey>,
    bounds: Vec<Option<Bounds>>,
    absolute_padding: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LayoutError(pub String);

impl fmt::Display for LayoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Error for LayoutError {}

impl Layout {
    pub fn new(
        name: String,
        meters_per_unit: f64,
        user_units_per_unit: f64,
        cells: Vec<Cell>,
    ) -> Result<Self, LayoutError> {
        if !meters_per_unit.is_finite()
            || meters_per_unit <= 0.0
            || !user_units_per_unit.is_finite()
            || user_units_per_unit <= 0.0
        {
            return Err(LayoutError(
                "Database units must be finite and positive".into(),
            ));
        }
        let mut names = HashSet::new();
        let mut referenced = HashSet::new();
        let mut layers = BTreeSet::new();
        for cell in &cells {
            if cell.name.is_empty() || !names.insert(cell.name.as_str()) {
                return Err(LayoutError(format!(
                    "Empty or duplicate cell name: {}",
                    cell.name
                )));
            }
            for shape in &cell.shapes {
                validate_shape(shape)
                    .map_err(|error| LayoutError(format!("{}: {error}", cell.name)))?;
                layers.insert(shape.layer);
            }
            for instance in &cell.instances {
                if cells.get(instance.cell.index()).is_none() {
                    return Err(LayoutError(format!(
                        "{}: missing referenced cell",
                        cell.name
                    )));
                }
                if !instance.transformation.valid()
                    || instance.array.columns == 0
                    || instance.array.rows == 0
                    || !instance.array.column_step.finite()
                    || !instance.array.row_step.finite()
                {
                    return Err(LayoutError(format!(
                        "{}: invalid instance transformation or array",
                        cell.name
                    )));
                }
                referenced.insert(instance.cell);
            }
        }
        let mut states = vec![0_u8; cells.len()];
        let mut order = Vec::with_capacity(cells.len());
        for start in 0..cells.len() {
            if states.get(start) == Some(&2) {
                continue;
            }
            let mut stack = vec![(start, false)];
            while let Some((index, exiting)) = stack.pop() {
                let state = states
                    .get_mut(index)
                    .ok_or_else(|| LayoutError("Invalid hierarchy index".into()))?;
                if exiting {
                    *state = 2;
                    order.push(index);
                    continue;
                }
                if *state == 2 {
                    continue;
                }
                let cell = cells
                    .get(index)
                    .ok_or_else(|| LayoutError("Invalid cell index".into()))?;
                if *state == 1 {
                    return Err(LayoutError(format!(
                        "Cyclic hierarchy at cell {}",
                        cell.name
                    )));
                }
                *state = 1;
                stack.push((index, true));
                for instance in cell.instances.iter().rev() {
                    stack.push((instance.cell.index(), false));
                }
            }
        }
        let mut bounds: Vec<Option<Bounds>> = vec![None; cells.len()];
        let mut absolute_padding = vec![0.0_f64; cells.len()];
        for index in order {
            let cell = cells
                .get(index)
                .ok_or_else(|| LayoutError("Invalid bounds index".into()))?;
            let mut cell_bounds: Option<Bounds> = None;
            let mut padding = 0.0_f64;
            for shape in &cell.shapes {
                let shape_bounds = shape_bounds(&shape.geometry)?;
                if let Geometry::Path(path) = &shape.geometry
                    && path.absolute_width
                {
                    let center_bounds =
                        Bounds::from_points(path.points.iter().map(|point| point.to_double()));
                    if let (Some(center), Some(outline)) = (center_bounds, shape_bounds) {
                        padding = padding
                            .max((center.minimum.x - outline.minimum.x).abs())
                            .max((center.minimum.y - outline.minimum.y).abs())
                            .max((center.maximum.x - outline.maximum.x).abs())
                            .max((center.maximum.y - outline.maximum.y).abs());
                    }
                }
                merge_bounds(&mut cell_bounds, shape_bounds);
            }
            for instance in &cell.instances {
                let child_padding = *absolute_padding
                    .get(instance.cell.index())
                    .ok_or_else(|| LayoutError("Invalid padding index".into()))?;
                padding = padding.max(child_padding);
                if let Some(child_bounds) = bounds.get(instance.cell.index()).copied().flatten() {
                    let transformed = child_bounds.transformed(instance.transformation).expanded(
                        child_padding
                            * (1.0 - instance.transformation.magnification).max(0.0)
                            * 2.0_f64.sqrt(),
                    );
                    merge_bounds(&mut cell_bounds, Some(instance.array.bounds(transformed)));
                }
            }
            if cell_bounds
                .is_some_and(|bounds| !bounds.minimum.finite() || !bounds.maximum.finite())
            {
                return Err(LayoutError(format!(
                    "{}: hierarchy bounds overflow",
                    cell.name
                )));
            }
            if let Some(destination) = bounds.get_mut(index) {
                *destination = cell_bounds;
            }
            if let Some(destination) = absolute_padding.get_mut(index) {
                *destination = padding;
            }
        }
        let mut top_cells: Vec<_> = cells
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                let id = CellId::from_index(index);
                (!referenced.contains(&id)).then_some(id)
            })
            .collect();
        top_cells.sort_by_key(|id| cells.get(id.index()).map(|cell| cell.name.as_str()));
        Ok(Self {
            name,
            meters_per_unit,
            user_units_per_unit,
            cells,
            top_cells,
            layers,
            bounds,
            absolute_padding,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn meters_per_unit(&self) -> f64 {
        self.meters_per_unit
    }
    pub fn user_units_per_unit(&self) -> f64 {
        self.user_units_per_unit
    }
    pub fn cells(&self) -> impl Iterator<Item = (CellId, &Cell)> {
        self.cells
            .iter()
            .enumerate()
            .map(|(index, cell)| (CellId::from_index(index), cell))
    }
    pub fn cell(&self, id: CellId) -> Option<&Cell> {
        self.cells.get(id.index())
    }
    pub fn top_cells(&self) -> &[CellId] {
        &self.top_cells
    }
    pub fn layers(&self) -> &BTreeSet<LayerKey> {
        &self.layers
    }
    pub fn bounds(&self, id: CellId) -> Option<Bounds> {
        self.bounds.get(id.index()).copied().flatten()
    }
    pub fn transformed_bounds(&self, id: CellId, transformation: Transformation) -> Option<Bounds> {
        let padding = self
            .absolute_padding
            .get(id.index())
            .copied()
            .unwrap_or_default();
        self.bounds(id).map(|bounds| {
            bounds
                .transformed(transformation)
                .expanded(padding * (1.0 - transformation.magnification).max(0.0) * 2.0_f64.sqrt())
        })
    }
    pub fn shape_count(&self) -> usize {
        self.cells.iter().map(|cell| cell.shapes.len()).sum()
    }
}

fn merge_bounds(destination: &mut Option<Bounds>, bounds: Option<Bounds>) {
    if let Some(bounds) = bounds {
        *destination = Some(destination.map_or(bounds, |previous| previous.union(bounds)));
    }
}

pub fn shape_bounds(geometry: &Geometry) -> Result<Option<Bounds>, LayoutError> {
    Ok(match geometry {
        Geometry::Polygon(points) => {
            Bounds::from_points(points.iter().map(|point| point.to_double()))
        }
        Geometry::Box { minimum, maximum } => {
            Bounds::from_points([minimum.to_double(), maximum.to_double()])
        }
        Geometry::Path(path) => {
            let outline = path_contour(path, 1.0)?;
            Bounds::from_points(outline)
                .or_else(|| Bounds::from_points(path.points.iter().map(|point| point.to_double())))
        }
        Geometry::Label(label) => Bounds::from_points([label.transformation.translation]),
    })
}

fn cross(first: DoublePoint, second: DoublePoint) -> f64 {
    first.x * second.y - first.y * second.x
}

fn validate_shape(shape: &Shape) -> Result<(), LayoutError> {
    match &shape.geometry {
        Geometry::Polygon(points) => validate_polygon(points),
        Geometry::Box { minimum, maximum } if minimum.x >= maximum.x || minimum.y >= maximum.y => {
            Err(LayoutError("Box must have positive area".into()))
        }
        Geometry::Path(path) => {
            if path.points.is_empty() || (path.points.len() < 2 && path.ends != PathEnds::Round) {
                return Err(LayoutError("Path has too few points".into()));
            }
            if path
                .points
                .windows(2)
                .any(|pair| pair.first() == pair.get(1))
            {
                return Err(LayoutError("Path has repeated consecutive points".into()));
            }
            let contour = path_contour(path, 1.0)?;
            if !contour.is_empty() {
                validate_contour(&contour)?;
            }
            Ok(())
        }
        Geometry::Label(label)
            if !label.transformation.valid()
                || label.horizontal_alignment > 2
                || label.vertical_alignment > 2 =>
        {
            Err(LayoutError(
                "Invalid label transformation or alignment".into(),
            ))
        }
        _ => Ok(()),
    }
}

pub fn validate_polygon(points: &[Point]) -> Result<(), LayoutError> {
    if points.len() < 3 {
        return Err(LayoutError("Polygon has fewer than three vertices".into()));
    }
    Ok(())
}

pub fn validate_contour(points: &[DoublePoint]) -> Result<(), LayoutError> {
    if points.len() < 3 {
        return Err(LayoutError("Polygon has fewer than three vertices".into()));
    }
    let edges: Vec<_> = points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
        .map(|(first, second)| (*first, *second))
        .collect();
    if points.iter().any(|point| !point.finite()) {
        return Err(LayoutError("Non-finite polygon coordinate".into()));
    }
    let origin = *points
        .first()
        .ok_or_else(|| LayoutError("Empty polygon".into()))?;
    let area: f64 = edges
        .iter()
        .map(|(first, second)| cross(first.minus(origin), second.minus(origin)))
        .sum();
    if area == 0.0 {
        return Err(LayoutError("Polygon has zero area".into()));
    }
    for (index, (first, second)) in edges.iter().enumerate() {
        if first == second {
            return Err(LayoutError(
                "Polygon has repeated consecutive vertices".into(),
            ));
        }
        for (other_index, (other_first, other_second)) in edges.iter().enumerate().skip(index + 1) {
            if other_index == index + 1 || (index == 0 && other_index == edges.len() - 1) {
                continue;
            }
            if first.x.min(second.x) > other_first.x.max(other_second.x)
                || first.x.max(second.x) < other_first.x.min(other_second.x)
                || first.y.min(second.y) > other_first.y.max(other_second.y)
                || first.y.max(second.y) < other_first.y.min(other_second.y)
            {
                continue;
            }
            let direction = second.minus(*first);
            let other_direction = other_second.minus(*other_first);
            let first_side = cross(direction, other_first.minus(*first));
            let second_side = cross(direction, other_second.minus(*first));
            let other_first_side = cross(other_direction, first.minus(*other_first));
            let other_second_side = cross(other_direction, second.minus(*other_first));
            if first_side * second_side <= 0.0 && other_first_side * other_second_side <= 0.0 {
                return Err(LayoutError(
                    "Intersecting or touching polygon contours are unsupported".into(),
                ));
            }
        }
    }
    Ok(())
}

pub fn path_contour(path: &Path, magnification: f64) -> Result<Vec<DoublePoint>, LayoutError> {
    let width = path.width as f64
        / if path.absolute_width {
            magnification
        } else {
            1.0
        };
    if !width.is_finite() || !magnification.is_finite() || magnification <= 0.0 {
        return Err(LayoutError("Path width overflow".into()));
    }
    if width == 0.0 {
        return Ok(Vec::new());
    }
    let half_width = width * 0.5;
    let first = path
        .points
        .first()
        .ok_or_else(|| LayoutError("Empty path".into()))?
        .to_double();
    if path.points.len() == 1 {
        if path.ends != PathEnds::Round {
            return Err(LayoutError("Single-point path requires round ends".into()));
        }
        return Ok((0..64)
            .map(|index| {
                let angle = f64::from(index) * std::f64::consts::TAU / 64.0;
                first.plus(DoublePoint::new(angle.cos(), angle.sin()).times(half_width))
            })
            .collect());
    }
    let points: Vec<_> = path.points.iter().map(|point| point.to_double()).collect();
    let directions: Vec<_> = points
        .windows(2)
        .map(|pair| {
            let first = pair
                .first()
                .copied()
                .ok_or_else(|| LayoutError("Missing path point".into()))?;
            let second = pair
                .get(1)
                .copied()
                .ok_or_else(|| LayoutError("Missing path point".into()))?;
            let direction = second.minus(first);
            let length = direction.x.hypot(direction.y);
            if length == 0.0 {
                return Err(LayoutError("Repeated path point".into()));
            }
            Ok(direction.times(1.0 / length))
        })
        .collect::<Result<Vec<_>, LayoutError>>()?;
    let beginning_direction = *directions
        .first()
        .ok_or_else(|| LayoutError("Missing path direction".into()))?;
    let end_direction = *directions
        .last()
        .ok_or_else(|| LayoutError("Missing path direction".into()))?;
    let (beginning_extension, end_extension) = match path.ends {
        PathEnds::Flush | PathEnds::Round => (0.0, 0.0),
        PathEnds::Square => (half_width, half_width),
        PathEnds::Extended { beginning, end } => (beginning as f64, end as f64),
    };
    let mut left = Vec::with_capacity(points.len());
    let mut right = Vec::with_capacity(points.len());
    for (index, point) in points.iter().copied().enumerate() {
        let (center, offset) = if index == 0 {
            (
                point.minus(beginning_direction.times(beginning_extension)),
                DoublePoint::new(-beginning_direction.y, beginning_direction.x).times(half_width),
            )
        } else if index == points.len() - 1 {
            (
                point.plus(end_direction.times(end_extension)),
                DoublePoint::new(-end_direction.y, end_direction.x).times(half_width),
            )
        } else {
            let before = *directions
                .get(index - 1)
                .ok_or_else(|| LayoutError("Missing path segment".into()))?;
            let after = *directions
                .get(index)
                .ok_or_else(|| LayoutError("Missing path segment".into()))?;
            let normal_before = DoublePoint::new(-before.y, before.x);
            let normal_after = DoublePoint::new(-after.y, after.x);
            let denominator = 1.0 + before.x * after.x + before.y * after.y;
            if denominator < 1e-12 {
                return Err(LayoutError(
                    "Reversing path segments are unsupported".into(),
                ));
            }
            (
                point,
                normal_before
                    .plus(normal_after)
                    .times(half_width / denominator),
            )
        };
        left.push(center.plus(offset));
        right.push(center.minus(offset));
    }
    let mut outline = left;
    if path.ends == PathEnds::Round {
        let center = *points
            .last()
            .ok_or_else(|| LayoutError("Missing path endpoint".into()))?;
        let angle = end_direction.y.atan2(end_direction.x) + std::f64::consts::FRAC_PI_2;
        for index in 1..32 {
            let angle = angle - f64::from(index) * std::f64::consts::PI / 32.0;
            outline.push(center.plus(DoublePoint::new(angle.cos(), angle.sin()).times(half_width)));
        }
    }
    outline.extend(right.into_iter().rev());
    if path.ends == PathEnds::Round {
        let angle =
            beginning_direction.y.atan2(beginning_direction.x) - std::f64::consts::FRAC_PI_2;
        for index in 1..32 {
            let angle = angle - f64::from(index) * std::f64::consts::PI / 32.0;
            outline.push(first.plus(DoublePoint::new(angle.cos(), angle.sin()).times(half_width)));
        }
    }
    if outline.iter().any(|point| !point.finite()) {
        return Err(LayoutError("Path contour overflow".into()));
    }
    Ok(outline)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(first: DoublePoint, second: DoublePoint) {
        assert!((first.x - second.x).abs() < 1e-8, "{first:?} != {second:?}");
        assert!((first.y - second.y).abs() < 1e-8, "{first:?} != {second:?}");
    }

    #[test]
    fn nested_reflection_changes_rotation_direction() {
        let parent = Transformation {
            translation: DoublePoint::new(10.0, 20.0),
            angle: std::f64::consts::FRAC_PI_2,
            magnification: 2.0,
            reflected: true,
        };
        let child = Transformation {
            translation: DoublePoint::new(3.0, 5.0),
            angle: 0.3,
            magnification: 0.5,
            reflected: true,
        };
        let point = DoublePoint::new(7.0, 11.0);
        close(
            parent.compose(child).apply(point),
            parent.apply(child.apply(point)),
        );
    }

    #[test]
    fn rejects_cycles_and_missing_cells() {
        let instance = Instance {
            cell: CellId::from_index(0),
            transformation: Transformation::default(),
            array: Array::default(),
        };
        assert!(
            Layout::new(
                "test".into(),
                1e-9,
                0.001,
                vec![Cell::new("cycle".into(), vec![], vec![instance.clone()])]
            )
            .is_err()
        );
        let instance = Instance {
            cell: CellId::from_index(1),
            ..instance
        };
        assert!(
            Layout::new(
                "test".into(),
                1e-9,
                0.001,
                vec![Cell::new("missing".into(), vec![], vec![instance])]
            )
            .is_err()
        );
    }

    #[test]
    fn bounds_preserve_skew_array_steps_and_top_cell_order() -> Result<(), LayoutError> {
        let shape = Shape {
            layer: LayerKey {
                layer: 1,
                datatype: 0,
            },
            geometry: Geometry::Box {
                minimum: Point::new(0, 0),
                maximum: Point::new(10, 20),
            },
        };
        let instance = Instance {
            cell: CellId::from_index(0),
            transformation: Transformation::default(),
            array: Array {
                columns: 3,
                rows: 2,
                column_step: DoublePoint::new(12.5, -5.0),
                row_step: DoublePoint::new(-7.0, 30.0),
            },
        };
        let layout = Layout::new(
            "test".into(),
            1e-9,
            0.001,
            vec![
                Cell::new("leaf".into(), vec![shape], vec![]),
                Cell::new("Z".into(), vec![], vec![instance]),
                Cell::new("A".into(), vec![], vec![]),
            ],
        )?;
        let bounds = layout
            .bounds(CellId::from_index(1))
            .ok_or_else(|| LayoutError("Missing bounds".into()))?;
        close(bounds.minimum, DoublePoint::new(-7.0, -10.0));
        close(bounds.maximum, DoublePoint::new(35.0, 50.0));
        assert_eq!(
            layout.top_cells(),
            &[CellId::from_index(2), CellId::from_index(1)]
        );
        assert!(layout.cell(CellId::from_index(100)).is_none());
        Ok(())
    }

    #[test]
    fn path_end_styles_and_absolute_width() -> Result<(), LayoutError> {
        for (ends, minimum, maximum) in [
            (PathEnds::Flush, 0.0, 100.0),
            (PathEnds::Square, -5.0, 105.0),
            (PathEnds::Round, -5.0, 105.0),
            (
                PathEnds::Extended {
                    beginning: 3,
                    end: 7,
                },
                -3.0,
                107.0,
            ),
        ] {
            let path = Path {
                points: vec![Point::new(0, 0), Point::new(100, 0)],
                width: 10,
                absolute_width: false,
                ends,
            };
            let bounds = Bounds::from_points(path_contour(&path, 1.0)?)
                .ok_or_else(|| LayoutError("Missing path bounds".into()))?;
            close(bounds.minimum, DoublePoint::new(minimum, -5.0));
            close(bounds.maximum, DoublePoint::new(maximum, 5.0));
        }
        let path = Path {
            points: vec![Point::new(0, 0), Point::new(100, 0)],
            width: 10,
            absolute_width: true,
            ends: PathEnds::Flush,
        };
        let bounds = Bounds::from_points(path_contour(&path, 2.0)?)
            .ok_or_else(|| LayoutError("Missing path bounds".into()))?;
        assert_eq!(bounds.height(), 5.0);
        Ok(())
    }

    #[test]
    fn polygons_accept_crossings_while_path_contours_remain_strict() -> Result<(), LayoutError> {
        validate_polygon(&[
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 4),
            Point::new(4, 4),
            Point::new(4, 10),
            Point::new(0, 10),
        ])?;
        let crossing = [
            Point::new(0, 0),
            Point::new(10, 10),
            Point::new(0, 10),
            Point::new(10, 0),
        ];
        validate_polygon(&crossing)?;
        validate_polygon(&[Point::new(0, 0); 3])?;
        assert!(validate_polygon(&crossing[..2]).is_err());
        assert!(validate_contour(&crossing.map(Point::to_double)).is_err());
        Ok(())
    }
}
