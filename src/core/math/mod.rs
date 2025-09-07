// src/core/math/mod.rs
//! Core mathematical operations for 3D graphics
//! 
//! This module provides fundamental mathematical types and operations
//! needed for 3D rendering, including transforms, quaternions, and
//! geometric operations.

pub mod transform;
pub mod quaternion;
pub mod frustum;
pub mod interpolation;

// // Re-export commonly used types and functions
// pub use transform::{Transform, TransformBuilder};
// pub use quaternion::{QuaternionExt, slerp, squad};
// pub use frustum::{Frustum, BoundingBox, BoundingSphere};

// Re-export cgmath types for convenience, but with our extensions
pub use cgmath::{
    //  Matrix4, 
     Vector3,
    //   Vector4, Point3,
    // Deg, Rad, perspective, ortho,
    InnerSpace, 
    // SquareMatrix, 
    Transform as CgTransform,
    //Quaternion, Zero, One
};

/// Common mathematical constants for graphics programming
pub mod constants {
    pub const PI: f32 = std::f32::consts::PI;
    pub const TAU: f32 = 2.0 * PI;
    pub const DEG_TO_RAD: f32 = PI / 180.0;
    pub const RAD_TO_DEG: f32 = 180.0 / PI;
    
    /// OpenGL to wgpu coordinate system conversion matrix
    /// Converts from OpenGL's [-1, 1] Z-range to wgpu's [0, 1] Z-range
    pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::from_cols(
        cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
        cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
        cgmath::Vector4::new(0.0, 0.0, 0.5, 0.0),
        cgmath::Vector4::new(0.0, 0.0, 0.5, 1.0),
    );
}

/// Utility functions for common graphics operations
pub fn degrees(radians: f32) -> f32 {
    radians * constants::RAD_TO_DEG
}

pub fn radians(degrees: f32) -> f32 {
    degrees * constants::DEG_TO_RAD
}

/// Safe normalization that handles zero-length vectors
pub fn safe_normalize(v: Vector3<f32>) -> Vector3<f32> {
    let length_sq = v.magnitude2();
    if length_sq > f32::EPSILON {
        v / length_sq.sqrt()
    } else {
        Vector3::new(0.0, 1.0, 0.0) // Default to up vector
    }
}

/// Calculate tangent and bitangent vectors for normal mapping
/// Returns (tangent, bitangent) as a tuple
pub fn calculate_tangent_bitangent(
    pos1: Vector3<f32>, pos2: Vector3<f32>, pos3: Vector3<f32>,
    uv1: cgmath::Vector2<f32>, uv2: cgmath::Vector2<f32>, uv3: cgmath::Vector2<f32>,
    normal: Vector3<f32>
) -> (Vector3<f32>, Vector3<f32>) {
    let edge1 = pos2 - pos1;
    let edge2 = pos3 - pos1;
    let delta_uv1 = uv2 - uv1;
    let delta_uv2 = uv3 - uv1;

    let f = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y);
    
    let tangent = Vector3::new(
        f * (delta_uv2.y * edge1.x - delta_uv1.y * edge2.x),
        f * (delta_uv2.y * edge1.y - delta_uv1.y * edge2.y),
        f * (delta_uv2.y * edge1.z - delta_uv1.y * edge2.z),
    );
    
    let bitangent = Vector3::new(
        f * (-delta_uv2.x * edge1.x + delta_uv1.x * edge2.x),
        f * (-delta_uv2.x * edge1.y + delta_uv1.x * edge2.y),
        f * (-delta_uv2.x * edge1.z + delta_uv1.x * edge2.z),
    );

    // Gram-Schmidt orthogonalize
    let tangent = safe_normalize(tangent - normal * tangent.dot(normal));
    
    // Calculate handedness
    let handedness = if normal.cross(tangent).dot(bitangent) < 0.0 { -1.0 } else { 1.0 };
    let bitangent = normal.cross(tangent) * handedness;

    (tangent, bitangent)
}