use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Entry {
    pub(super) bounds: Bounds,
    pub(super) index: usize,
    pub(super) instance: bool,
    pub(super) labels: (f64, f64),
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

    pub(super) fn query(&self) -> Query {
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

pub(super) struct Query {
    stack: Vec<usize>,
    range: std::ops::Range<usize>,
}

impl Query {
    pub(super) fn next(
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

fn label_margin(metrics: (f64, f64), magnification: f64, units_per_point: f64) -> f64 {
    if metrics.0 == 0.0 {
        0.0
    } else {
        (metrics.0 + 1.0) * (12.0 * metrics.1 * magnification).clamp(6.0, 96.0) * units_per_point
    }
}
