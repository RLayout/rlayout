struct Job {
    size: vec2<f32>,
    columns: u32,
    rows: u32,
    column_step: vec2<f32>,
    row_step: vec2<f32>,
};
struct Instance {
    affine: vec4<f32>,
    translation: vec2<f32>,
    padding: vec2<f32>,
};
@group(0) @binding(0) var<uniform> job: Job;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(1) @binding(0) var glyph: texture_2d<f32>;
@group(1) @binding(1) var glyph_sampler: sampler;

struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) kind: u32,
    @location(2) @interpolate(flat) enabled: u32,
};
fn transform(point: vec2<f32>, instance: Instance) -> vec2<f32> {
    return vec2<f32>(dot(instance.affine.xy, point), dot(instance.affine.zw, point)) + instance.translation;
}
@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) other: vec2<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) bounds: vec4<f32>,
    @location(4) kind: u32,
    @location(5) corner: u32,
    @builtin(instance_index) instance_index: u32,
) -> Output {
    var instance: Instance;
    if job.rows != 0u {
        let array_count = job.columns * job.rows;
        let base_index = instance_index / array_count;
        let lattice_index = instance_index % array_count;
        instance = instances[base_index];
        instance.translation += f32(lattice_index % job.columns) * job.column_step
            + f32(lattice_index / job.columns) * job.row_step;
    } else {
        instance = instances[instance_index];
    }
    var point = transform(position, instance);
    var enabled = 1u;
    if kind == 4u {
        let source_extent = bounds.zw - bounds.xy;
        let affine_length_x = length(instance.affine.xz);
        let affine_length_y = length(instance.affine.yw);
        let length_x = affine_length_x * source_extent.x;
        let length_y = affine_length_y * source_extent.y;
        var direction_x = vec2<f32>(1.0, 0.0);
        var direction_y = vec2<f32>(0.0, 1.0);
        if affine_length_x > 0.0 { direction_x = instance.affine.xz / affine_length_x; }
        if affine_length_y > 0.0 { direction_y = instance.affine.yw / affine_length_y; }
        let sign_x = select(-1.0, 1.0, corner == 1u || corner == 2u);
        let sign_y = select(-1.0, 1.0, corner >= 2u);
        point += direction_x * max(1.0 - length_x, 0.0) * sign_x * 0.5
            + direction_y * max(1.0 - length_y, 0.0) * sign_y * 0.5;
    }
    if kind == 1u || kind == 2u {
        if kind == 1u {
            let source_extent = bounds.zw - bounds.xy;
            let extent = vec2<f32>(dot(abs(instance.affine.xy), source_extent), dot(abs(instance.affine.zw), source_extent));
            let intrinsic_width = uv.x * length(instance.affine.xz);
            let intrinsically_thin = intrinsic_width > 0.0 && intrinsic_width < 1.0;
            if extent.x >= 1.0 && extent.y >= 1.0 && !intrinsically_thin { enabled = 0u; }
        }
        if enabled != 0u {
            let source_difference = other - position;
            let difference = vec2<f32>(dot(instance.affine.xy, source_difference), dot(instance.affine.zw, source_difference));
            let distance = length(difference);
            var direction = vec2<f32>(1.0, 0.0);
            if distance > 0.000001 { direction = difference / distance; }
            let side = f32(corner >> 1u) * 2.0 - 1.0;
            let endpoint = select(0.0, 1.0, corner == 1u || corner == 2u);
            point += difference * endpoint + direction * (endpoint - 0.5)
                + vec2<f32>(-direction.y, direction.x) * side * 0.5;
        } else {
            point = vec2<f32>(-2.0);
        }
    }
    var output: Output;
    output.position = vec4<f32>(2.0 * point.x / job.size.x - 1.0, 1.0 - 2.0 * point.y / job.size.y, 0.0, 1.0);
    output.uv = uv;
    output.kind = kind;
    output.enabled = enabled;
    return output;
}
@fragment
fn fs_main(input: Output) -> @location(0) f32 {
    if input.enabled == 0u { discard; }
    if input.kind == 3u { return textureSampleLevel(glyph, glyph_sampler, input.uv, 0.0).r; }
    return 1.0;
}
