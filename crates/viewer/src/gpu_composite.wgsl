struct Palette {
    colors: array<vec4<u32>, 64>,
    count: vec4<u32>,
};
@group(0) @binding(0) var masks: texture_2d_array<f32>;
@group(0) @binding(1) var<uniform> palette: Palette;
@group(0) @binding(2) var previous: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let coordinate = vec2<i32>(position.xy);
    var color = vec4<u32>(round(textureLoad(previous, coordinate, 0) * 255.0));
    for (var layer = 0u; layer < palette.count.x; layer += 1u) {
        let coverage = u32(round(textureLoad(masks, coordinate, i32(layer), 0).r * 255.0));
        let source = palette.colors[layer] * coverage / 255u;
        color = min(vec4<u32>(255u), source + color * (255u - source.a) / 255u);
    }
    return vec4<f32>(color) / 255.0;
}
