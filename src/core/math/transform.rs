// src/core/math/transform.rs
//! Transform operations for 3D objects
//! 
//! Provides a clean interface for managing position, rotation, and scale
//! transformations, replacing the scattered matrix operations in your
//! current codebase.

use cgmath::*;
use std::fmt;

/// A complete 3D transform representing position, rotation, and scale
#[derive(Debug, Clone, PartialEq)]
pub struct GTransform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl GTransform {
    /// Create a new identity transform
    pub fn identity() -> Self {
        Self {
            position: Vector3::zero(),
            rotation: Quaternion::one(),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    /// Create a transform with just position
    pub fn from_position(position: Vector3<f32>) -> Self {
        Self {
            position,
            rotation: Quaternion::one(),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    /// Create a transform from position and rotation
    pub fn from_position_rotation(position: Vector3<f32>, rotation: Quaternion<f32>) -> Self {
        Self {
            position,
            rotation,
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }

    /// Convert transform to a 4x4 transformation matrix
    /// This replaces the manual matrix construction you're doing in Instance::to_raw()
    pub fn to_matrix(&self) -> Matrix4<f32> {
        let translation_matrix = Matrix4::from_translation(self.position);
        let rotation_matrix = Matrix4::from(self.rotation);
        let scale_matrix = Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);
        
        translation_matrix * rotation_matrix * scale_matrix
    }

    /// Get the forward direction vector (local -Z axis in right-handed system)
    pub fn forward(&self) -> Vector3<f32> {
        self.rotation * Vector3::new(0.0, 0.0, -1.0)
    }

    /// Get the up direction vector (local Y axis)
    pub fn up(&self) -> Vector3<f32> {
        self.rotation * Vector3::new(0.0, 1.0, 0.0)
    }

    /// Get the right direction vector (local X axis)
    pub fn right(&self) -> Vector3<f32> {
        self.rotation * Vector3::new(1.0, 0.0, 0.0)
    }

    /// Translate by a vector
    pub fn translate(&mut self, translation: Vector3<f32>) {
        self.position += translation;
    }

    /// Rotate by a quaternion
    pub fn rotate(&mut self, rotation: Quaternion<f32>) {
        self.rotation = rotation * self.rotation;
    }

    /// Rotate around an arbitrary axis
    pub fn rotate_axis_angle(&mut self, axis: Vector3<f32>, angle: Rad<f32>) {
        let rotation = Quaternion::from_axis_angle(axis.normalize(), angle);
        self.rotate(rotation);
    }

    /// Scale uniformly
    pub fn scale_uniform(&mut self, scale: f32) {
        self.scale *= scale;
    }

    /// Scale non-uniformly
    pub fn scale_non_uniform(&mut self, scale: Vector3<f32>) {
        self.scale.x *= scale.x;
        self.scale.y *= scale.y;
        self.scale.z *= scale.z;
    }

    /// Look at a target position (useful for cameras and lights)
    pub fn look_at(&mut self, target: Point3<f32>, up: Vector3<f32>) {
        let forward = (target - Point3::from_vec(self.position)).normalize();
        let right = forward.cross(up.normalize()).normalize();
        let up = right.cross(forward);

        // Create rotation matrix from basis vectors
        let rotation_matrix = Matrix3::from_cols(right, up, -forward);
        self.rotation = Quaternion::from(rotation_matrix);
    }

    // /// Combine two transforms (this * other)
    // pub fn combine(&self, other: &Transform) -> Transform {
    //     Transform {
    //         position: self.position + self.rotation * (other.position * self.scale),
    //         rotation: self.rotation * other.rotation,
    //         scale: Vector3::new(
    //             self.scale.x * other.scale.x,
    //             self.scale.y * other.scale.y,
    //             self.scale.z * other.scale.z,
    //         ),
    //     }
    // }

    // /// Get the inverse transform
    // pub fn inverse(&self) -> Transform {
    //     let inv_rotation = self.rotation.conjugate();
    //     let inv_scale = Vector3::new(1.0 / self.scale.x, 1.0 / self.scale.y, 1.0 / self.scale.z);
    //     let inv_position = inv_rotation * (-self.position * inv_scale);

    //     Transform {
    //         position: inv_position,
    //         rotation: inv_rotation,
    //         scale: inv_scale,
    //     }
    // }

    /// Transform a point from local space to world space
    pub fn transform_point(&self, point: Point3<f32>) -> Point3<f32> {
        let scaled = Vector3::new(
            point.x * self.scale.x,
            point.y * self.scale.y,
            point.z * self.scale.z,
        );
        let rotated = self.rotation * scaled;
        Point3::from_vec(self.position + rotated)
    }

    /// Transform a vector from local space to world space (ignores position)
    pub fn transform_vector(&self, vector: Vector3<f32>) -> Vector3<f32> {
        let scaled = Vector3::new(
            vector.x * self.scale.x,
            vector.y * self.scale.y,
            vector.z * self.scale.z,
        );
        self.rotation * scaled
    }

    /// Transform a normal vector (inverse transpose transformation)
    pub fn transform_normal(&self, normal: Vector3<f32>) -> Vector3<f32> {
        // For normal vectors, we need the inverse transpose of the rotation
        // and the inverse of the scale
        let inv_scale = Vector3::new(1.0 / self.scale.x, 1.0 / self.scale.y, 1.0 / self.scale.z);
        let scaled_normal = Vector3::new(
            normal.x * inv_scale.x,
            normal.y * inv_scale.y,
            normal.z * inv_scale.z,
        );
        (self.rotation * scaled_normal).normalize()
    }
}

impl Default for GTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl fmt::Display for GTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Transform {{ pos: {:?}, rot: {:?}, scale: {:?} }}", 
               self.position, self.rotation, self.scale)
    }
}

/// Builder pattern for constructing transforms
pub struct TransformBuilder {
    transform: GTransform,
}

impl TransformBuilder {
    pub fn new() -> Self {
        Self {
            transform: GTransform::identity(),
        }
    }

    pub fn with_position(mut self, position: Vector3<f32>) -> Self {
        self.transform.position = position;
        self
    }

    pub fn with_rotation(mut self, rotation: Quaternion<f32>) -> Self {
        self.transform.rotation = rotation;
        self
    }

    pub fn with_rotation_euler(mut self, pitch: f32, yaw: f32, roll: f32) -> Self {
        let pitch_quat = Quaternion::from_angle_x(Rad(pitch));
        let yaw_quat = Quaternion::from_angle_y(Rad(yaw));
        let roll_quat = Quaternion::from_angle_z(Rad(roll));
        self.transform.rotation = yaw_quat * pitch_quat * roll_quat;
        self
    }

    pub fn with_scale(mut self, scale: Vector3<f32>) -> Self {
        self.transform.scale = scale;
        self
    }

    pub fn with_uniform_scale(mut self, scale: f32) -> Self {
        self.transform.scale = Vector3::new(scale, scale, scale);
        self
    }

    pub fn looking_at(mut self, target: Point3<f32>, up: Vector3<f32>) -> Self {
        self.transform.look_at(target, up);
        self
    }

    pub fn build(self) -> GTransform {
        self.transform
    }
}

impl Default for TransformBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Camera-specific transform utilities
/// This replaces and extends your current Camera struct
// pub struct CameraTransform {
//     pub transform: Transform,
//     pub fovy: f32,
//     pub aspect: f32,
//     pub znear: f32,
//     pub zfar: f32,
// }

// impl CameraTransform {
//     pub fn new(fovy: f32, aspect: f32, znear: f32, zfar: f32) -> Self {
//         Self {
//             transform: Transform::identity(),
//             fovy,
//             aspect,
//             znear,
//             zfar,
//         }
//     }

//     // /// Create view matrix from camera transform
//     pub fn view_matrix(&self) -> Matrix4<f32> {
//         // Camera view matrix is the inverse of the camera's world transform
//         let target = Point3::from_vec(self.transform.position) + Point3::from_vec(self.transform.forward());
//         Matrix4::look_at_rh(
//             Point3::from_vec(self.transform.position),
//             target,
//             self.transform.up(),
//         )
//     }

//     /// Create projection matrix
//     pub fn projection_matrix(&self) -> Matrix4<f32> {
//         perspective(Deg(self.fovy), self.aspect, self.znear, self.zfar)
//     }

//     /// Create combined view-projection matrix
//     /// This replaces your Camera::build_view_projection_matrix method
//     pub fn view_projection_matrix(&self) -> Matrix4<f32> {
//         let view = self.view_matrix();
//         let proj = self.projection_matrix();
//         proj * view
//     }

//     /// Get camera frustum for culling (we'll implement this in frustum.rs)
//     pub fn frustum(&self) -> crate::core::math::frustum::Frustum {
//         crate::core::math::frustum::Frustum::from_view_projection(self.view_projection_matrix())
//     }

//     /// Move camera forward/backward along its forward vector
//     pub fn move_forward(&mut self, distance: f32) {
//         let forward = self.transform.forward() * distance;
//         self.transform.translate(forward);
//     }

//     /// Move camera right/left along its right vector
//     pub fn move_right(&mut self, distance: f32) {
//         let right = self.transform.right() * distance;
//         self.transform.translate(right);
//     }

//     /// Move camera up/down along its up vector
//     pub fn move_up(&mut self, distance: f32) {
//         let up = self.transform.up() * distance;
//         self.transform.translate(up);
//     }

//     /// Orbit around a target point
//     pub fn orbit(&mut self, target: Point3<f32>, delta_yaw: f32, delta_pitch: f32) {
//         let distance = (Point3::from_vec(self.transform.position) - target).magnitude();
        
//         // Convert to spherical coordinates
//         let offset = Point3::from_vec(self.transform.position) - target;
//         let radius = offset.magnitude();
//         let theta = offset.z.atan2(offset.x); // azimuth
//         let phi = (offset.y / radius).acos(); // polar angle
        
//         // Apply rotation
//         let new_theta = theta + delta_yaw;
//         let new_phi = (phi + delta_pitch).clamp(0.1, std::f32::consts::PI - 0.1);
        
//         // Convert back to Cartesian
//         let new_offset = Vector3::new(
//             radius * new_phi.sin() * new_theta.cos(),
//             radius * new_phi.cos(),
//             radius * new_phi.sin() * new_theta.sin(),
//         );
        
//         self.transform.position = target.to_vec() + new_offset;
//         self.transform.look_at(target, Vector3::unit_y());
//     }
// }

/// Utility functions for common transform operations
pub mod utils {
    use super::*;

    /// Create a transform that represents the transformation from one coordinate space to another
    pub fn change_of_basis(
        from_right: Vector3<f32>,
        from_up: Vector3<f32>,
        from_forward: Vector3<f32>,
        to_right: Vector3<f32>,
        to_up: Vector3<f32>,
        to_forward: Vector3<f32>,
    ) -> Matrix4<f32> {
        let from_matrix = Matrix3::from_cols(from_right, from_up, from_forward);
        let to_matrix = Matrix3::from_cols(to_right, to_up, to_forward);
        Matrix4::from(to_matrix * from_matrix.transpose())
    }

    /// Decompose a transformation matrix into position, rotation, and scale
    pub fn decompose_matrix(matrix: Matrix4<f32>) -> (Vector3<f32>, Quaternion<f32>, Vector3<f32>) {
        // Extract translation
        let position = Vector3::new(matrix.w.x, matrix.w.y, matrix.w.z);

        // Extract scale
        let scale_x = Vector3::new(matrix.x.x, matrix.x.y, matrix.x.z).magnitude();
        let scale_y = Vector3::new(matrix.y.x, matrix.y.y, matrix.y.z).magnitude();
        let scale_z = Vector3::new(matrix.z.x, matrix.z.y, matrix.z.z).magnitude();
        let scale = Vector3::new(scale_x, scale_y, scale_z);

        // Extract rotation (normalize the matrix first)
        let rotation_matrix = Matrix3::new(
            matrix.x.x / scale_x, matrix.x.y / scale_x, matrix.x.z / scale_x,
            matrix.y.x / scale_y, matrix.y.y / scale_y, matrix.y.z / scale_y,
            matrix.z.x / scale_z, matrix.z.y / scale_z, matrix.z.z / scale_z,
        );
        let rotation = Quaternion::from(rotation_matrix);

        (position, rotation, scale)
    }
}