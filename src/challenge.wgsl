// Vertex shader
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // Generate quad vertices (4 vertices for triangle strip)
    let x = select(-0.5, 0.5, in_vertex_index >= 2);
    let y = select(-0.5, 0.5, (in_vertex_index == 1) || (in_vertex_index == 3));
    let position = vec3<f32>(x, y, 0.0);

    // Apply rotation animation
    let angle = f32(instance_index) * 0.01;
    let rot = mat3x3<f32>(
        cos(angle), -sin(angle), 0.0,
        sin(angle), cos(angle), 0.0,
        0.0, 0.0, 1.0
    );
    let world_pos = rot * position;
    
    var out: VertexOutput;
    out.world_position = world_pos;
    out.clip_position = vec4<f32>(world_pos, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Gradient based on world position
    let r = abs(in.world_position.x) + 0.2;
    let g = abs(in.world_position.y) + 0.2;
    let b = 0.8 - length(in.world_position.xy) * 0.5;
    
    return vec4<f32>(r, g, b, 1.0);
}