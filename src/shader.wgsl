// Vertex shader
// This is where we define the structure of the input and output data for our vertex shader.
struct CameraUniform {
    // This struct holds the view-projection matrix used to transform vertices from object space to screen space.
    view_proj: mat4x4<f32>,
};

@group(1) @binding(0) // Binding 1, group 1
var<uniform> camera: CameraUniform;

struct InstanceInput {
    // This struct holds the model matrices for each instance of a mesh.
    // The location attributes specify where these values should be stored in memory.
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
};

struct VertexInput {
    // This struct holds the position and texture coordinates of a vertex.
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    // This struct holds the final transformed position and texture coordinates of a vertex.
    @builtin(position) clip_position: vec4<f32>, // The output position is built-in to the shader pipeline.
    @location(0) tex_coords: vec2<f32>,
}

// Main entry point for the vertex shader
@vertex
fn vs_main(
    model: VertexInput, // The input data from the mesh vertices.
    instance: InstanceInput, // The instance data for this particular vertex in the mesh.
) -> VertexOutput {
    // Calculate the combined model matrix by concatenating the model matrices.
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    
    var out: VertexOutput;
    // Copy the texture coordinates from the input data to the output data.
    out.tex_coords = model.tex_coords;
    // Transform the vertex position using the combined model matrix and view-projection matrix.
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>; // The diffuse texture to sample from.

@group(0) @binding(1)
var s_diffuse: sampler; // The sampler used to read from the diffuse texture.

// Main entry point for the fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the diffuse texture using the current texture coordinates and sampler.
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}