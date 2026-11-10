struct Presentation {
    rectangle: vec4<f32>,
    uv_rectangle: vec4<f32>,
    size: vec2<f32>,
    srgb_target: u32,
    padding: u32,
};
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@group(0) @binding(2) var<uniform> presentation: Presentation;
struct Output {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Output {
    let corners = array<vec2<f32>, 6>(vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0));
    let uv = corners[index];
    let point = mix(presentation.rectangle.xy, presentation.rectangle.zw, uv);
    var output: Output;
    output.position = vec4<f32>(2.0 * point.x / presentation.size.x - 1.0, 1.0 - 2.0 * point.y / presentation.size.y, 0.0, 1.0);
    output.uv = mix(presentation.uv_rectangle.xy, presentation.uv_rectangle.zw, uv);
    return output;
}
fn linear_from_gamma(value: vec3<f32>) -> vec3<f32> {
    return select(pow((value + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)), value / 12.92, value < vec3<f32>(0.04045));
}
@fragment
fn fs_main(input: Output) -> @location(0) vec4<f32> {
    let geometry = textureSample(image, image_sampler, input.uv);
    var color = geometry.rgb + vec3<f32>(17.0, 21.0, 27.0) / 255.0 * (1.0 - geometry.a);
    if presentation.srgb_target != 0u { color = linear_from_gamma(color); }
    return vec4<f32>(color, 1.0);
}
