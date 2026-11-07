use super::*;

#[path = "../examples/fixture.rs"]
mod fixtures;

use fixtures::Fixture;
use rlayout_core::DoublePoint;

#[test]
fn real_numbers_use_base_sixteen() {
    assert_eq!(real8(&[0x41, 0x10, 0, 0, 0, 0, 0, 0]), Some(1.0));
    assert_eq!(real8(&[0xc1, 0x20, 0, 0, 0, 0, 0, 0]), Some(-2.0));
    assert_eq!(real8(&[0; 8]), Some(0.0));
}

#[test]
fn original_demo_loads_common_geometry_and_units() -> Result<(), Box<dyn Error>> {
    let bytes = fixtures::demo()?;
    let loaded = read(bytes.as_slice())?;
    assert!((loaded.layout.meters_per_unit() - 1e-9).abs() < 1e-20);
    assert!((loaded.layout.user_units_per_unit() - 0.001).abs() < 1e-12);
    assert_eq!(loaded.layout.cells().count(), 2);
    assert_eq!(loaded.layout.shape_count(), 7);
    assert!(loaded.warnings.is_empty());
    let top = loaded
        .layout
        .cell(CellId::from_index(0))
        .ok_or("Missing top cell")?;
    assert_eq!(top.instances().len(), 3);
    let instance = top
        .instances()
        .get(1)
        .ok_or("Missing transformed instance")?;
    assert!(instance.transformation.reflected);
    assert!((instance.transformation.angle - std::f64::consts::PI / 6.0).abs() < 1e-12);
    assert_eq!(instance.transformation.magnification, 0.75);
    Ok(())
}

#[test]
fn array_endpoints_are_divided_by_counts_in_parent_coordinates() -> Result<(), Box<dyn Error>> {
    let mut fixture = Fixture::new()?;
    fixture.cell("TOP")?;
    fixture.array("LEAF", 3, 2, &[(10, 20), (35, 14), (4, 50)])?;
    fixture.end_cell()?;
    fixture.cell("LEAF")?;
    fixture.polygon(3, &[(0, 0), (4, 0), (4, 6), (0, 6), (0, 0)])?;
    fixture.end_cell()?;
    fixture.end()?;
    let result = read(fixture.bytes.as_slice())?;
    let top = result
        .layout
        .cell(CellId::from_index(0))
        .ok_or("Missing top")?;
    let array = top.instances().first().ok_or("Missing array")?.array;
    assert!((array.column_step.x - 25.0 / 3.0).abs() < 1e-12);
    assert_eq!(array.column_step.y, -2.0);
    assert_eq!(array.row_step, DoublePoint::new(-3.0, 15.0));
    let bounds = result
        .layout
        .bounds(CellId::from_index(0))
        .ok_or("Missing bounds")?;
    assert_eq!(bounds.minimum, DoublePoint::new(7.0, 16.0));
    assert!((bounds.maximum.x - (14.0 + 50.0 / 3.0)).abs() < 1e-12);
    assert_eq!(bounds.maximum.y, 41.0);
    Ok(())
}

#[test]
fn nested_transforms_preserve_known_coordinates() -> Result<(), Box<dyn Error>> {
    let mut fixture = Fixture::new()?;
    fixture.cell("ROOT")?;
    fixture.reference("MIDDLE", (100, 200), true, 90.0, 2.0)?;
    fixture.end_cell()?;
    fixture.cell("MIDDLE")?;
    fixture.reference("LEAF", (10, 20), false, 90.0, 0.5)?;
    fixture.end_cell()?;
    fixture.cell("LEAF")?;
    fixture.polygon(1, &[(0, 0), (4, 0), (4, 6), (0, 6), (0, 0)])?;
    fixture.end_cell()?;
    fixture.end()?;
    let loaded = read(fixture.bytes.as_slice())?;
    let bounds = loaded
        .layout
        .bounds(CellId::from_index(0))
        .ok_or("Missing bounds")?;
    assert!((bounds.minimum.x - 140.0).abs() < 1e-10);
    assert!((bounds.maximum.x - 144.0).abs() < 1e-10);
    assert!((bounds.minimum.y - 214.0).abs() < 1e-10);
    assert!((bounds.maximum.y - 220.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn missing_duplicate_and_cyclic_cells_fail() -> Result<(), Box<dyn Error>> {
    for referenced in ["MISSING", "TOP"] {
        let mut fixture = Fixture::new()?;
        fixture.cell("TOP")?;
        fixture.reference(referenced, (0, 0), false, 0.0, 1.0)?;
        fixture.end_cell()?;
        fixture.end()?;
        let error = read(fixture.bytes.as_slice())
            .err()
            .ok_or("Expected reference error")?;
        assert!(error.message.contains(if referenced == "TOP" {
            "Cyclic"
        } else {
            "Unresolved"
        }));
    }
    let mut fixture = Fixture::new()?;
    for _ in 0..2 {
        fixture.cell("DUPLICATE")?;
        fixture.end_cell()?;
    }
    fixture.end()?;
    assert!(read(fixture.bytes.as_slice()).is_err());
    Ok(())
}

#[test]
fn malformed_and_truncated_records_report_offsets() -> Result<(), Box<dyn Error>> {
    let bytes = fixtures::demo()?;
    for length in 0..bytes.len() {
        assert!(
            read(bytes.get(..length).ok_or("Missing prefix")?).is_err(),
            "Accepted truncated file at {length}"
        );
    }
    for header in [[0, 2, 0, 2], [0, 5, 0, 2], [0, 6, 0, 3]] {
        let error = read(header.as_slice())
            .err()
            .ok_or("Expected malformed record")?;
        assert_eq!(error.offset, 0);
    }
    assert!(read([0x1f, 0x8b, 0x08, 0].as_slice()).is_err());
    Ok(())
}

#[test]
fn unsupported_geometry_is_an_error_with_element_context() -> Result<(), Box<dyn Error>> {
    let mut fixture = Fixture::new()?;
    fixture.cell("TOP")?;
    fixture.record(0x0a, 0, &[])?;
    fixture.string(0x12, "TOP")?;
    fixture.record(0x1a, 1, &4_u16.to_be_bytes())?;
    let error = read(fixture.bytes.as_slice())
        .err()
        .ok_or("Expected STRANS error")?;
    assert_eq!(error.cell.as_deref(), Some("TOP"));
    assert_eq!(error.element, Some("SREF"));
    assert!(error.message.contains("Absolute"));
    Ok(())
}

#[test]
fn invalid_units_are_rejected_at_the_units_record() -> Result<(), Box<dyn Error>> {
    let fixture = Fixture::new()?;
    let offset = fixture
        .bytes
        .windows(4)
        .position(|header| header == [0, 20, 3, 5])
        .ok_or("Missing units record")?;
    for unit in [0, 8] {
        for negative in [false, true] {
            let mut bytes = fixture.bytes.clone();
            let real = bytes
                .get_mut(offset + 4 + unit..offset + 12 + unit)
                .ok_or("Missing unit real")?;
            if negative {
                let sign = real.first_mut().ok_or("Missing sign")?;
                *sign |= 0x80;
            } else {
                real.fill(0);
            }
            let error = read(bytes.as_slice())
                .err()
                .ok_or("Accepted invalid units")?;
            assert_eq!(error.offset, offset as u64);
            assert!(error.message.contains("positive"));
        }
    }
    Ok(())
}

#[test]
fn unsupported_path_contours_keep_cell_and_element_context() -> Result<(), Box<dyn Error>> {
    for points in [
        vec![],
        vec![(0, 0), (0, 0)],
        vec![(0, 0), (20, 0), (20, 20), (0, 20), (0, 10), (30, 10)],
    ] {
        let mut fixture = Fixture::new()?;
        fixture.cell("TOP")?;
        fixture.path(1, 0, 2, &points)?;
        let error = read(fixture.bytes.as_slice())
            .err()
            .ok_or("Accepted invalid path")?;
        assert_eq!(error.cell.as_deref(), Some("TOP"));
        assert_eq!(error.element, Some("PATH"));
    }
    Ok(())
}

#[test]
fn array_counts_reject_zero_and_negative_signed_values() -> Result<(), Box<dyn Error>> {
    for (columns, rows) in [(0, 1), (1, 0), (u16::MAX, 1), (1, u16::MAX)] {
        let mut fixture = Fixture::new()?;
        fixture.cell("TOP")?;
        fixture.array("LEAF", columns, rows, &[(0, 0), (30, 0), (0, 30)])?;
        let error = read(fixture.bytes.as_slice())
            .err()
            .ok_or("Accepted invalid array counts")?;
        assert_eq!(error.element, Some("AREF"));
        assert!(error.message.contains("counts"));
    }
    Ok(())
}

#[test]
fn properties_warn_and_boxes_keep_their_layer_type() -> Result<(), Box<dyn Error>> {
    let mut fixture = Fixture::new()?;
    fixture.cell("BOX")?;
    fixture.record(0x2d, 0, &[])?;
    fixture.short(0x0d, 8)?;
    fixture.short(0x2e, 2)?;
    fixture.points(&[(0, 0), (10, 0), (10, 20), (0, 20), (0, 0)])?;
    fixture.short(0x2b, 1)?;
    fixture.string(0x2c, "original property")?;
    fixture.record(0x11, 0, &[])?;
    fixture.end_cell()?;
    fixture.end()?;
    let loaded = read(fixture.bytes.as_slice())?;
    assert_eq!(loaded.warnings.len(), 1);
    assert!(loaded.layout.layers().contains(&LayerKey {
        layer: 8,
        datatype: 2
    }));
    Ok(())
}

#[test]
fn boundaries_accept_implicit_closure_repeated_vertices_and_crossings() -> Result<(), Box<dyn Error>>
{
    let contours = [
        vec![(0, 0), (10, 0), (10, 10), (0, 10)],
        vec![(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        vec![(0, 0), (0, 0), (10, 0), (10, 10), (0, 10)],
        vec![(0, 0), (10, 10), (0, 10), (10, 0), (0, 0)],
        vec![(0, 0), (5, 0), (10, 0)],
    ];
    let mut fixture = Fixture::new()?;
    fixture.cell("TOP")?;
    for contour in &contours {
        fixture.polygon(94, contour)?;
    }
    fixture.end_cell()?;
    fixture.end()?;
    let loaded = read(fixture.bytes.as_slice())?;
    assert!(loaded.warnings.is_empty());
    assert_eq!(loaded.layout.shape_count(), contours.len());
    let cell = loaded
        .layout
        .cell(CellId::from_index(0))
        .ok_or("Missing top cell")?;
    for (shape, contour) in cell.shapes().iter().zip(&contours) {
        let Geometry::Polygon(points) = &shape.geometry else {
            return Err("Boundary was not retained as a polygon".into());
        };
        let length = contour.len() - usize::from(contour.first() == contour.last());
        assert_eq!(points.len(), length);
    }
    Ok(())
}

#[test]
fn undersized_boundaries_warn_and_do_not_stop_following_elements() -> Result<(), Box<dyn Error>> {
    let mut fixture = Fixture::new()?;
    fixture.cell("TOP")?;
    let mut offsets = Vec::new();
    for points in [
        vec![],
        vec![(0, 0)],
        vec![(0, 0), (10, 0)],
        vec![(0, 0), (10, 0), (0, 0)],
    ] {
        offsets.push(fixture.bytes.len());
        fixture.polygon(94, &points)?;
    }
    fixture.polygon(1, &[(0, 0), (10, 0), (10, 10), (0, 10)])?;
    fixture.end_cell()?;
    fixture.end()?;
    let loaded = read(fixture.bytes.as_slice())?;
    assert_eq!(loaded.layout.shape_count(), 1);
    assert_eq!(loaded.warnings.len(), offsets.len());
    for (warning, offset) in loaded.warnings.iter().zip(offsets) {
        assert_eq!(warning.offset, offset as u64);
        assert_eq!(warning.cell.as_deref(), Some("TOP"));
        assert!(warning.message.contains("BOUNDARY"));
    }
    Ok(())
}

#[test]
fn large_original_fixture_keeps_one_hundred_thousand_shapes() -> Result<(), Box<dyn Error>> {
    let bytes = fixtures::large()?;
    let loaded = read(bytes.as_slice())?;
    assert_eq!(loaded.layout.shape_count(), 100_000);
    Ok(())
}

#[test]
fn labels_and_negative_widths_preserve_presentation() -> Result<(), Box<dyn Error>> {
    let mut fixture = Fixture::new()?;
    fixture.cell("Z")?;
    fixture.label("ORIGINAL", (100, 200))?;
    fixture.path(2, 0, -10, &[(0, 0), (10, 20)])?;
    fixture.end_cell()?;
    fixture.cell("A")?;
    fixture.end_cell()?;
    fixture.end()?;
    let loaded = read(fixture.bytes.as_slice())?;
    let cell = loaded
        .layout
        .cell(CellId::from_index(0))
        .ok_or("Missing cell")?;
    match &cell.shapes().first().ok_or("Missing label")?.geometry {
        Geometry::Label(label) => {
            assert_eq!(label.horizontal_alignment, 1);
            assert_eq!(label.vertical_alignment, 1);
            assert_eq!(
                label.transformation.translation,
                DoublePoint::new(100.0, 200.0)
            );
        }
        _ => return Err("Expected label".into()),
    }
    match &cell.shapes().get(1).ok_or("Missing path")?.geometry {
        Geometry::Path(path) => {
            assert!(path.absolute_width);
            assert_eq!(path.width, 10);
        }
        _ => return Err("Expected path".into()),
    }
    assert_eq!(
        loaded.layout.top_cells().first(),
        Some(&CellId::from_index(1))
    );
    Ok(())
}
