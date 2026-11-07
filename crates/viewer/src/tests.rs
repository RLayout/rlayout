use super::*;
use rlayout_core::{Array, Cell, Instance, LayerKey, Point, Shape};

#[test]
fn zoom_preserves_cursor_location() {
    let viewport = Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(800.0, 600.0));
    let cursor = Pos2::new(230.0, 420.0);
    let mut camera = Camera {
        center: DoublePoint::new(2e9, -2e9),
        pixels_per_unit: 2.0,
    };
    let before = camera.screen_to_world(cursor, viewport);
    camera.zoom_at(cursor, viewport, 3.0);
    assert_eq!(camera.screen_to_world(cursor, viewport), before);
    assert!((camera.world_to_screen(before, viewport) - cursor).length() < 0.01);
}

fn scene_with_instances() -> Result<Scene, Box<dyn Error>> {
    let shape = Shape {
        layer: LayerKey {
            layer: 1,
            datatype: 0,
        },
        geometry: Geometry::Polygon(vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ]),
    };
    let instances = [0.0, 10_000.0]
        .into_iter()
        .map(|x| Instance {
            cell: CellId::from_index(0),
            transformation: Transformation {
                translation: DoublePoint::new(x, 0.0),
                ..Default::default()
            },
            array: Array::default(),
        })
        .collect();
    let layout = Layout::new(
        "test".into(),
        1e-9,
        0.001,
        vec![
            Cell::new("LEAF".into(), vec![shape], vec![]),
            Cell::new("TOP".into(), vec![], instances),
        ],
    )?;
    Ok(Scene::prepare(layout)?)
}

#[test]
fn culls_offscreen_hierarchy_and_respects_depth_and_layers() -> Result<(), Box<dyn Error>> {
    let mut viewer = Viewer::new(scene_with_instances()?);
    let context = egui::Context::default();
    let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::splat(100.0));
    let rendered = viewer.render(&context, viewport)?;
    assert_eq!(rendered.shape_count, 1);
    assert!(!rendered.truncated);
    viewer.hierarchy_depth = 0;
    assert_eq!(viewer.render(&context, viewport)?.shape_count, 0);
    viewer.hierarchy_depth = 1;
    for style in viewer.styles.values_mut() {
        style.visible = false;
    }
    assert_eq!(viewer.render(&context, viewport)?.shape_count, 0);
    Ok(())
}

#[test]
fn array_culling_never_omits_a_visible_element() {
    let element = Bounds {
        minimum: DoublePoint::new(-2.0, -3.0),
        maximum: DoublePoint::new(4.0, 5.0),
    };
    let viewport = Bounds {
        minimum: DoublePoint::new(5.0, -10.0),
        maximum: DoublePoint::new(30.0, 20.0),
    };
    for (column, row) in [
        (DoublePoint::new(8.0, 2.0), DoublePoint::new(-3.0, 7.0)),
        (DoublePoint::new(-8.0, 2.0), DoublePoint::new(3.0, 7.0)),
        (DoublePoint::new(4.0, 0.0), DoublePoint::new(-2.0, 0.0)),
        (DoublePoint::default(), DoublePoint::new(0.0, 5.0)),
    ] {
        let (columns, rows) = visible_array_range(element, viewport, column, row, 20, 20);
        for column_index in 0..20_u16 {
            for row_index in 0..20_u16 {
                let offset = column
                    .times(f64::from(column_index))
                    .plus(row.times(f64::from(row_index)));
                if element.translated(offset).intersects(viewport) {
                    assert!((columns.0..columns.1).contains(&column_index));
                    assert!((rows.0..rows.1).contains(&row_index));
                }
            }
        }
    }
}

#[test]
fn concave_triangles_preserve_area_at_large_coordinates() -> Result<(), Box<dyn Error>> {
    let origin = 2e9;
    let polygon = triangulate(vec![
        DoublePoint::new(origin, origin),
        DoublePoint::new(origin + 10.0, origin),
        DoublePoint::new(origin + 10.0, origin + 4.0),
        DoublePoint::new(origin + 4.0, origin + 4.0),
        DoublePoint::new(origin + 4.0, origin + 10.0),
        DoublePoint::new(origin, origin + 10.0),
    ])?;
    assert!((triangle_area(&polygon)? - 64.0).abs() < 1e-9);
    Ok(())
}

fn triangle_points(polygon: &Polygon) -> Result<Vec<[DoublePoint; 3]>, Box<dyn Error>> {
    polygon
        .triangles
        .as_chunks::<3>()
        .0
        .iter()
        .map(|indices| {
            let [first, second, third] = indices.map(|index| {
                polygon
                    .points
                    .get(index as usize)
                    .copied()
                    .ok_or("Missing triangle vertex")
            });
            Ok([first?, second?, third?])
        })
        .collect()
}

fn triangle_area(polygon: &Polygon) -> Result<f64, Box<dyn Error>> {
    Ok(triangle_points(polygon)?
        .into_iter()
        .map(|[first, second, third]| {
            let first_edge = second.minus(first);
            let second_edge = third.minus(first);
            (first_edge.x * second_edge.y - first_edge.y * second_edge.x).abs() * 0.5
        })
        .sum())
}

fn covers(triangles: &[[DoublePoint; 3]], point: DoublePoint) -> bool {
    triangles.iter().any(|triangle| {
        let sides = triangle
            .iter()
            .zip(triangle.iter().cycle().skip(1))
            .take(3)
            .map(|(first, second)| {
                let edge = second.minus(*first);
                let displacement = point.minus(*first);
                edge.x * displacement.y - edge.y * displacement.x
            })
            .collect::<Vec<_>>();
        sides.iter().all(|side| *side >= 0.0) || sides.iter().all(|side| *side <= 0.0)
    })
}

fn winding_number(points: &[DoublePoint], point: DoublePoint) -> i32 {
    let mut winding = 0;
    for (first, second) in points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
    {
        let edge = second.minus(*first);
        let displacement = point.minus(*first);
        let side = edge.x * displacement.y - edge.y * displacement.x;
        if first.y <= point.y && second.y > point.y && side > 0.0 {
            winding += 1;
        } else if first.y > point.y && second.y <= point.y && side < 0.0 {
            winding -= 1;
        }
    }
    winding
}

#[test]
fn complex_polygons_preserve_nonzero_fill_and_area() -> Result<(), Box<dyn Error>> {
    let cases = [
        (
            vec![
                (0, 1355),
                (0, 2910),
                (2730, 2910),
                (2730, 2530),
                (1945, 2530),
                (1945, 2035),
                (2730, 2035),
                (2730, 2910),
                (3220, 2910),
                (3220, 1355),
            ],
            4_618_525.0,
        ),
        (
            vec![
                (0, 0),
                (20, 0),
                (20, 20),
                (0, 20),
                (0, 0),
                (3, 3),
                (3, 7),
                (7, 7),
                (7, 3),
                (3, 3),
                (0, 0),
                (12, 12),
                (12, 16),
                (16, 16),
                (16, 12),
                (12, 12),
                (0, 0),
            ],
            368.0,
        ),
        (
            vec![
                (0, 0),
                (5, 0),
                (5, 5),
                (0, 5),
                (0, 0),
                (5, 5),
                (10, 5),
                (10, 10),
                (5, 10),
                (5, 5),
                (0, 0),
            ],
            50.0,
        ),
        (vec![(0, 0), (10, 10), (0, 10), (10, 0)], 50.0),
        (
            vec![
                (0, 0),
                (10, 0),
                (10, 10),
                (0, 10),
                (0, 0),
                (5, 0),
                (15, 0),
                (15, 10),
                (5, 10),
                (5, 0),
                (0, 0),
            ],
            150.0,
        ),
        (
            vec![
                (0, 0),
                (10, 0),
                (10, 10),
                (0, 10),
                (0, 0),
                (5, 0),
                (5, 10),
                (15, 10),
                (15, 0),
                (5, 0),
                (0, 0),
            ],
            100.0,
        ),
        (
            vec![
                (0, 0),
                (10, 0),
                (10, 10),
                (0, 10),
                (0, 0),
                (0, 10),
                (10, 10),
                (10, 0),
            ],
            0.0,
        ),
        (
            vec![(0, 0), (0, 0), (10, 0), (10, 10), (10, 10), (0, 10)],
            100.0,
        ),
        (vec![(0, 0), (5, 0), (10, 0), (5, 0)], 0.0),
        (vec![(0, 0); 3], 0.0),
    ];
    for (coordinates, area) in cases {
        let points: Vec<_> = coordinates
            .into_iter()
            .map(|(x, y)| DoublePoint::new(f64::from(x), f64::from(y)))
            .collect();
        for mut contour in [points.clone(), points.into_iter().rev().collect()] {
            let bounds =
                Bounds::from_points(contour.iter().copied()).ok_or("Missing contour bounds")?;
            let polygon = triangulate(contour.clone())?;
            assert_eq!(polygon.bounds, bounds);
            assert!(
                (triangle_area(&polygon)? - area).abs() < 1e-6,
                "Area differs for {contour:?}"
            );
            let triangles = triangle_points(&polygon)?;
            for column in 0..21 {
                for row in 0..21 {
                    let point = DoublePoint::new(
                        bounds.minimum.x + bounds.width() * (f64::from(column) + 0.314) / 21.0,
                        bounds.minimum.y + bounds.height() * (f64::from(row) + 0.271) / 21.0,
                    );
                    if bounds.width() > 0.0 && bounds.height() > 0.0 {
                        assert_eq!(
                            covers(&triangles, point),
                            winding_number(&contour, point) != 0,
                            "Coverage differs at {point:?} for {contour:?}"
                        );
                    }
                }
            }
            contour.rotate_left(1);
            assert!((triangle_area(&triangulate(contour)?)? - area).abs() < 1e-6);
        }
    }
    Ok(())
}

#[test]
fn crossings_create_vertices_and_preserve_narrow_geometry() -> Result<(), Box<dyn Error>> {
    let polygon = triangulate(vec![
        DoublePoint::new(0.0, 0.0),
        DoublePoint::new(10.0, 10.0),
        DoublePoint::new(0.0, 10.0),
        DoublePoint::new(10.0, 0.0),
    ])?;
    assert!(polygon.points.contains(&DoublePoint::new(5.0, 5.0)));
    for origin in [f64::from(i32::MIN), f64::from(i32::MAX) - 1000.0] {
        let polygon = triangulate(vec![
            DoublePoint::new(origin, origin),
            DoublePoint::new(origin + 1000.0, origin + 1.0),
            DoublePoint::new(origin + 1000.0, origin + 2.0),
            DoublePoint::new(origin, origin + 1.0),
        ])?;
        assert!((triangle_area(&polygon)? - 1000.0).abs() < 1e-9);
    }
    assert!(triangulate(vec![DoublePoint::new(f64::NAN, 0.0); 3]).is_err());
    assert!(triangulate(vec![DoublePoint::new(0.0, 0.0); 2]).is_err());
    Ok(())
}

#[test]
fn polygon_fill_matches_nonzero_winding_at_fractional_coordinates() -> Result<(), Box<dyn Error>> {
    let points = [
        (0.25, 1.75),
        (9.5, 8.25),
        (1.5, 9.75),
        (7.25, -0.5),
        (8.75, 10.5),
        (-1.25, 5.25),
        (10.5, 3.5),
    ]
    .map(|(x, y)| DoublePoint::new(x, y));
    let triangles = triangle_points(&triangulate(points.to_vec())?)?;
    let mut interior_points = 0;
    let mut exterior_points = 0;
    for row in -8..49 {
        for column in -8..49 {
            let point = DoublePoint::new(
                f64::from(column) * 0.25 + 0.137,
                f64::from(row) * 0.25 + 0.263,
            );
            let on_edge = points
                .iter()
                .zip(points.iter().cycle().skip(1))
                .take(points.len())
                .any(|(first, second)| {
                    let edge = second.minus(*first);
                    let displacement = point.minus(*first);
                    let projection = ((displacement.x * edge.x + displacement.y * edge.y)
                        / (edge.x * edge.x + edge.y * edge.y))
                        .clamp(0.0, 1.0);
                    let nearest = first.plus(edge.times(projection));
                    let distance = point.minus(nearest);
                    distance.x * distance.x + distance.y * distance.y < 1e-14
                });
            if on_edge {
                continue;
            }
            let inside = winding_number(&points, point) != 0;
            if inside {
                interior_points += 1;
            } else {
                exterior_points += 1;
            }
            assert_eq!(
                covers(&triangles, point),
                inside,
                "Coverage differs at {point:?}"
            );
        }
    }
    assert!(interior_points > 0);
    assert!(exterior_points > 0);
    Ok(())
}

#[test]
fn nonzero_winding_fills_the_center_of_a_self_intersecting_star() -> Result<(), Box<dyn Error>> {
    let points = [
        (0.0, 6.0),
        (4.0, -6.0),
        (-6.0, 2.0),
        (6.0, 2.0),
        (-4.0, -6.0),
    ]
    .map(|(x, y)| DoublePoint::new(x, y));
    let point = DoublePoint::default();
    assert_eq!(winding_number(&points, point).abs(), 2);
    assert!(covers(
        &triangle_points(&triangulate(points.to_vec())?)?,
        point
    ));
    Ok(())
}

#[test]
fn absolute_path_revalidation_reports_shape_context() -> Result<(), Box<dyn Error>> {
    let path = Path {
        points: vec![
            Point::new(0, 0),
            Point::new(20, 0),
            Point::new(20, 20),
            Point::new(0, 20),
        ],
        width: 2,
        absolute_width: true,
        ends: rlayout_core::PathEnds::Flush,
    };
    let layout = Layout::new(
        "paths".into(),
        1e-9,
        0.001,
        vec![
            Cell::new(
                "LEAF".into(),
                vec![Shape {
                    layer: LayerKey {
                        layer: 7,
                        datatype: 2,
                    },
                    geometry: Geometry::Path(path),
                }],
                vec![],
            ),
            Cell::new(
                "TOP".into(),
                vec![],
                vec![Instance {
                    cell: CellId::from_index(0),
                    transformation: Transformation {
                        magnification: 0.02,
                        ..Default::default()
                    },
                    array: Array::default(),
                }],
            ),
        ],
    )?;
    let mut viewer = Viewer::new(Scene::prepare(layout)?);
    let error = viewer
        .render(
            &egui::Context::default(),
            Rect::from_min_size(Pos2::ZERO, Vec2::splat(400.0)),
        )
        .err()
        .ok_or("Expected an invalid enlarged path contour")?;
    assert!(error.0.contains("Cell LEAF, layer 7/2, shape 1"));
    assert!(error.0.contains("contours"));
    Ok(())
}

#[test]
fn fit_and_pan_use_the_canvas_coordinate_system() {
    let viewport = Rect::from_min_size(Pos2::new(100.0, 50.0), Vec2::new(800.0, 400.0));
    let bounds = Bounds {
        minimum: DoublePoint::new(1000.0, 2000.0),
        maximum: DoublePoint::new(1200.0, 2100.0),
    };
    let mut camera = Camera::default();
    camera.fit(Some(bounds), viewport);
    assert_eq!(camera.center, bounds.center());
    assert_eq!(camera.pixels_per_unit, 3.6);
    let before = camera.world_to_screen(bounds.minimum, viewport);
    camera.pan(Vec2::new(36.0, 18.0));
    assert!(
        (camera.world_to_screen(bounds.minimum, viewport) - before - Vec2::new(36.0, 18.0))
            .length()
            < 0.001
    );
}

#[test]
fn canvas_middle_drag_pans_without_changing_scale() -> Result<(), Box<dyn Error>> {
    let mut viewer = Viewer::new(scene_with_instances()?);
    let context = egui::Context::default();
    let mut draw = |events| {
        let input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::splat(400.0))),
            events,
            ..Default::default()
        };
        let mut output = context.run_ui(input, |ui| viewer.canvas(ui));
        output.textures_delta.clear();
        viewer.camera
    };
    let initial = draw(vec![]);
    draw(vec![
        egui::Event::PointerMoved(Pos2::new(200.0, 200.0)),
        egui::Event::PointerButton {
            pos: Pos2::new(200.0, 200.0),
            button: egui::PointerButton::Middle,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
    ]);
    let moved = draw(vec![egui::Event::PointerMoved(Pos2::new(240.0, 220.0))]);
    assert_eq!(moved.pixels_per_unit, initial.pixels_per_unit);
    assert!((moved.center.x - initial.center.x + 40.0 / initial.pixels_per_unit).abs() < 1e-8);
    assert!((moved.center.y - initial.center.y - 20.0 / initial.pixels_per_unit).abs() < 1e-8);
    Ok(())
}

#[test]
fn label_meshes_have_normalized_font_coordinates() -> Result<(), Box<dyn Error>> {
    let layer = LayerKey {
        layer: 10,
        datatype: 0,
    };
    let shape = Shape {
        layer,
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
    let layout = Layout::new(
        "text".into(),
        1e-9,
        0.001,
        vec![Cell::new("TOP".into(), vec![shape], vec![])],
    )?;
    let mut viewer = Viewer::new(Scene::prepare(layout)?);
    let context = egui::Context::default();
    let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::splat(400.0));
    let mut rendered = None;
    let mut output = context.run_ui(egui::RawInput::default(), |ui| {
        rendered = Some(viewer.render(ui.ctx(), viewport));
    });
    output.textures_delta.clear();
    let frame = rendered.ok_or("No rendered text frame")??;
    assert_eq!(frame.shape_count, 1);
    let mesh = frame.meshes.first().ok_or("No text mesh")?;
    assert!(!mesh.vertices.is_empty());
    assert!(
        mesh.vertices
            .iter()
            .all(|vertex| (0.0..=1.0).contains(&vertex.uv.x) && (0.0..=1.0).contains(&vertex.uv.y))
    );
    Ok(())
}

#[test]
#[ignore = "Run with --release --ignored --nocapture for the 100,000-shape render benchmark"]
fn benchmark_navigation() -> Result<(), Box<dyn Error>> {
    let start = std::time::Instant::now();
    let shapes = (0..100_000_i64)
        .map(|index| {
            let x = index % 500 * 12;
            let y = index / 500 * 12;
            Shape {
                layer: LayerKey {
                    layer: (index % 6) as u16,
                    datatype: 0,
                },
                geometry: Geometry::Polygon(vec![
                    Point::new(x, y),
                    Point::new(x + 10, y),
                    Point::new(x + 10, y + 10),
                    Point::new(x, y + 10),
                ]),
            }
        })
        .collect();
    let layout = Layout::new(
        "benchmark".into(),
        1e-9,
        0.001,
        vec![Cell::new("TOP".into(), shapes, vec![])],
    )?;
    let mut viewer = Viewer::new(Scene::prepare(layout)?);
    let preparation = start.elapsed();
    let viewport = Rect::from_min_size(Pos2::ZERO, Vec2::new(1920.0, 1080.0));
    viewer.camera.fit(
        viewer.selected.and_then(|id| viewer.layout().bounds(id)),
        viewport,
    );
    let context = egui::Context::default();
    let start = std::time::Instant::now();
    let frame = viewer.render(&context, viewport)?;
    assert_eq!(frame.shape_count, 100_000);
    assert!(!frame.truncated);
    let mut durations = Vec::new();
    for _ in 0..30 {
        viewer.camera.pan(Vec2::new(0.5, 0.5));
        let start = std::time::Instant::now();
        let frame = viewer.render(&context, viewport)?;
        assert!(!frame.truncated);
        durations.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    durations.sort_by(f64::total_cmp);
    let median = durations
        .get(15)
        .copied()
        .ok_or("Missing benchmark median")?;
    let maximum = durations
        .last()
        .copied()
        .ok_or("Missing benchmark maximum")?;
    println!(
        "100k shapes: preparation {:.2} ms; initial and navigation total {:.2} ms; median {:.2} ms, maximum {:.2} ms; CPU {:.1} fps",
        preparation.as_secs_f64() * 1000.0,
        start.elapsed().as_secs_f64() * 1000.0,
        median,
        maximum,
        1000.0 / median
    );
    assert!(
        median < 33.333,
        "CPU mesh preparation missed the 30 FPS target: {median:.2} ms"
    );
    Ok(())
}
