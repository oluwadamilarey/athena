// src/core/math/quaternion.rs
//! Extended quaternion operations for animation and smooth rotations
//! 
//! Provides additional functionality beyond cgmath's basic quaternion support,
//! particularly for animation interpolation and conversion utilities.

use cgmath::*;

/// Extension trait for cgmath::Quaternion with additional utility methods
pub trait QuaternionExt<S: BaseFloat> {
    /// Create a quaternion from Euler angles in ZYX order (yaw, pitch, roll)
    fn from_euler_zyx(yaw: S, pitch: S, roll: S) -> Quaternion<S>;
    
    /// Convert quaternion to Euler angles in ZYX order
    fn to_euler_zyx(&self) -> (S, S, S);
    
    /// Create a quaternion that represents the shortest rotation between two vectors
    fn from_vectors(from: Vector3<S>, to: Vector3<S>) -> Quaternion<S>;
    
    /// Get the angle of rotation represented by this quaternion
    fn angle(&self) -> S;
    
    /// Get the axis of rotation represented by this quaternion
    fn axis(&self) -> Vector3<S>;
    
    /// Ensure the quaternion is in the positive hemisphere (w >= 0)
    fn ensure_positive(&self) -> Quaternion<S>;
    
    /// Check if this quaternion is approximately equal to another
    fn approx_eq(&self, other: &Quaternion<S>, epsilon: S) -> bool;
}

impl QuaternionExt<f32> for Quaternion<f32> {
    fn from_euler_zyx(yaw: f32, pitch: f32, roll: f32) -> Quaternion<f32> {
        // Convert to half angles
        let cy = (yaw * 0.5).cos();
        let sy = (yaw * 0.5).sin();
        let cp = (pitch * 0.5).cos();
        let sp = (pitch * 0.5).sin();
        let cr = (roll * 0.5).cos();
        let sr = (roll * 0.5).sin();

        Quaternion::new(
            cr * cp * cy + sr * sp * sy, // w
            sr * cp * cy - cr * sp * sy, // x
            cr * sp * cy + sr * cp * sy, // y
            cr * cp * sy - sr * sp * cy, // z
        )
    }

    fn to_euler_zyx(&self) -> (f32, f32, f32) {
        let q = self;
        
        // Roll (x-axis rotation)
        let sin_r_cp = 2.0 * (q.s * q.v.x + q.v.y * q.v.z);
        let cos_r_cp = 1.0 - 2.0 * (q.v.x * q.v.x + q.v.y * q.v.y);
        let roll = sin_r_cp.atan2(cos_r_cp);

        // Pitch (y-axis rotation)
        let sin_p = 2.0 * (q.s * q.v.y - q.v.z * q.v.x);
        let pitch = if sin_p.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sin_p)
        } else {
            sin_p.asin()
        };

        // Yaw (z-axis rotation)
        let sin_y_cp = 2.0 * (q.s * q.v.z + q.v.x * q.v.y);
        let cos_y_cp = 1.0 - 2.0 * (q.v.y * q.v.y + q.v.z * q.v.z);
        let yaw = sin_y_cp.atan2(cos_y_cp);

        (yaw, pitch, roll)
    }

    fn from_vectors(from: Vector3<f32>, to: Vector3<f32>) -> Quaternion<f32> {
        let from = from.normalize();
        let to = to.normalize();
        
        let dot = from.dot(to);
        
        // If vectors are parallel
        if dot >= 1.0 - f32::EPSILON {
            return Quaternion::one(); // No rotation needed
        }
        
        // If vectors are opposite
        if dot <= -1.0 + f32::EPSILON {
            // Find a perpendicular axis
            let axis = if from.x.abs() < 0.9 {
                from.cross(Vector3::unit_x()).normalize()
            } else {
                from.cross(Vector3::unit_y()).normalize()
            };
            return Quaternion::from_axis_angle(axis, Rad(std::f32::consts::PI));
        }
        
        // General case
        let cross = from.cross(to);
        let s = ((1.0 + dot) * 2.0).sqrt();
        let inv_s = 1.0 / s;
        
        Quaternion::new(
            s * 0.5,
            cross.x * inv_s,
            cross.y * inv_s,
            cross.z * inv_s,
        )
    }

    fn angle(&self) -> f32 {
        2.0 * self.s.abs().min(1.0).acos()
    }

    fn axis(&self) -> Vector3<f32> {
        let sin_theta_inv = 1.0 / (1.0 - self.s * self.s).max(f32::EPSILON).sqrt();
        Vector3::new(
            self.v.x * sin_theta_inv,
            self.v.y * sin_theta_inv,
            self.v.z * sin_theta_inv,
        )
    }

    fn ensure_positive(&self) -> Quaternion<f32> {
        if self.s < 0.0 {
            Quaternion::new(-self.s, -self.v.x, -self.v.y, -self.v.z)
        } else {
            *self
        }
    }

    fn approx_eq(&self, other: &Quaternion<f32>, epsilon: f32) -> bool {
        // Account for double-cover property of quaternions
        let dot = self.s * other.s + self.v.x * other.v.x + self.v.y * other.v.y + self.v.z * other.v.z;
        dot.abs() >= 1.0 - epsilon
    }
}

/// Spherical linear interpolation between two quaternions
/// This is the standard SLERP algorithm used in animation systems
pub fn slerp(q1: Quaternion<f32>, q2: Quaternion<f32>, t: f32) -> Quaternion<f32> {
    let  q1 = q1.ensure_positive();
    let mut q2 = q2.ensure_positive();
    
    // Compute the cosine of the angle between the quaternions
    let mut dot = q1.s * q2.s + q1.v.dot(q2.v);
    
    // If the dot product is negative, slerp won't take the shorter path.
    // Fix by reversing one quaternion.
    if dot < 0.0 {
        q2 = Quaternion::new(-q2.s, -q2.v.x, -q2.v.y, -q2.v.z);
        dot = -dot;
    }
    
    // If the inputs are too close for comfort, linearly interpolate
    if dot > 0.9995 {
        let result = Quaternion::new(
            q1.s + t * (q2.s - q1.s),
            q1.v.x + t * (q2.v.x - q1.v.x),
            q1.v.y + t * (q2.v.y - q1.v.y),
            q1.v.z + t * (q2.v.z - q1.v.z),
        );
        return result.normalize();
    }
    
    // Calculate the angle between the quaternions
    let theta_0 = dot.acos();
    let sin_theta_0 = theta_0.sin();
    
    let theta = theta_0 * t;
    let sin_theta = theta.sin();
    
    let s0 = (theta_0 - theta).cos() / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;
    
    Quaternion::new(
        s0 * q1.s + s1 * q2.s,
        s0 * q1.v.x + s1 * q2.v.x,
        s0 * q1.v.y + s1 * q2.v.y,
        s0 * q1.v.z + s1 * q2.v.z,
    )
}

/// Spherical quadrangle interpolation for smooth quaternion curves
/// Used for animating through multiple quaternion keyframes
pub fn squad(
    q1: Quaternion<f32>,
    q2: Quaternion<f32>,
    s1: Quaternion<f32>,
    s2: Quaternion<f32>,
    t: f32,
) -> Quaternion<f32> {
    let c = slerp(q1, q2, t);
    let d = slerp(s1, s2, t);
    slerp(c, d, 2.0 * t * (1.0 - t))
}

/// Calculate intermediate quaternions for SQUAD interpolation
pub fn squad_intermediate(
    q_prev: Quaternion<f32>,
    q_curr: Quaternion<f32>,
    q_next: Quaternion<f32>,
) -> Quaternion<f32> {
    let q_curr_inv = q_curr.conjugate();
    let log_prev = quaternion_log(q_curr_inv * q_prev);
    let log_next = quaternion_log(q_curr_inv * q_next);
    let avg_log = Quaternion::new(
        0.0,
        -(log_prev.v.x + log_next.v.x) / 4.0,
        -(log_prev.v.y + log_next.v.y) / 4.0,
        -(log_prev.v.z + log_next.v.z) / 4.0,
    );
    q_curr * quaternion_exp(avg_log)
}

/// Natural logarithm of a quaternion
fn quaternion_log(q: Quaternion<f32>) -> Quaternion<f32> {
    let vec_length = q.v.magnitude();
    if vec_length < f32::EPSILON {
        Quaternion::new(q.s.ln(), 0.0, 0.0, 0.0)
    } else {
        let t = vec_length.atan2(q.s) / vec_length;
        Quaternion::new(
            (q.s * q.s + vec_length * vec_length).sqrt().ln(),
            q.v.x * t,
            q.v.y * t,
            q.v.z * t,
        )
    }
}

/// Exponential of a quaternion
fn quaternion_exp(q: Quaternion<f32>) -> Quaternion<f32> {
    let vec_length = q.v.magnitude();
    if vec_length < f32::EPSILON {
        Quaternion::new(q.s.exp(), 0.0, 0.0, 0.0)
    } else {
        let exp_s = q.s.exp();
        let cos_vec = vec_length.cos();
        let sin_vec = vec_length.sin();
        let factor = exp_s * sin_vec / vec_length;
        
        Quaternion::new(
            exp_s * cos_vec,
            q.v.x * factor,
            q.v.y * factor,
            q.v.z * factor,
        )
    }
}

/// Normalized linear interpolation - faster but less accurate than SLERP
pub fn nlerp(q1: Quaternion<f32>, q2: Quaternion<f32>, t: f32) -> Quaternion<f32> {
    let  q1 = q1;
    let mut q2 = q2;
    
    // Ensure we take the shorter path
    let dot = q1.s * q2.s + q1.v.dot(q2.v);
    if dot < 0.0 {
        q2 = Quaternion::new(-q2.s, -q2.v.x, -q2.v.y, -q2.v.z);
    }
    
    let result = Quaternion::new(
        q1.s + t * (q2.s - q1.s),
        q1.v.x + t * (q2.v.x - q1.v.x),
        q1.v.y + t * (q2.v.y - q1.v.y),
        q1.v.z + t * (q2.v.z - q1.v.z),
    );
    
    result.normalize()
}

/// Quaternion spline interpolation for smooth animation curves
pub struct QuaternionSpline {
    keyframes: Vec<(f32, Quaternion<f32>)>,
    intermediates: Vec<Quaternion<f32>>,
}

impl QuaternionSpline {
    pub fn new(mut keyframes: Vec<(f32, Quaternion<f32>)>) -> Self {
        if keyframes.is_empty() {
            return Self {
                keyframes,
                intermediates: Vec::new(),
            };
        }
        
        // Sort keyframes by time
        keyframes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Ensure all quaternions are in the same hemisphere
        for i in 1..keyframes.len() {
            let prev = keyframes[i - 1].1;
            let curr = keyframes[i].1;
            let dot = prev.s * curr.s + prev.v.dot(curr.v);
            if dot < 0.0 {
                keyframes[i].1 = Quaternion::new(-curr.s, -curr.v.x, -curr.v.y, -curr.v.z);
            }
        }
        
        // Calculate intermediate quaternions for SQUAD
        let mut intermediates = Vec::with_capacity(keyframes.len());
        
        for i in 0..keyframes.len() {
            let q_prev = if i == 0 { keyframes[i].1 } else { keyframes[i - 1].1 };
            let q_curr = keyframes[i].1;
            let q_next = if i == keyframes.len() - 1 { keyframes[i].1 } else { keyframes[i + 1].1 };
            
            let intermediate = squad_intermediate(q_prev, q_curr, q_next);
            intermediates.push(intermediate);
        }
        
        Self {
            keyframes,
            intermediates,
        }
    }
    
    pub fn sample(&self, time: f32) -> Quaternion<f32> {
        if self.keyframes.is_empty() {
            return Quaternion::one();
        }
        
        if self.keyframes.len() == 1 {
            return self.keyframes[0].1;
        }
        
        // Clamp time to valid range
        let time = time.clamp(self.keyframes[0].0, self.keyframes.last().unwrap().0);
        
        // Find the segment containing the time
        let mut i = 0;
        while i < self.keyframes.len() - 1 && time > self.keyframes[i + 1].0 {
            i += 1;
        }
        
        if i >= self.keyframes.len() - 1 {
            return self.keyframes.last().unwrap().1;
        }
        
        // Calculate interpolation parameter
        let t0 = self.keyframes[i].0;
        let t1 = self.keyframes[i + 1].0;
        let t = (time - t0) / (t1 - t0);
        
        // Use SQUAD for smooth interpolation
        squad(
            self.keyframes[i].1,
            self.keyframes[i + 1].1,
            self.intermediates[i],
            self.intermediates[i + 1],
            t,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::assert_abs_diff_eq;

    #[test]
    fn test_quaternion_euler_conversion() {
        let yaw = 1.2;
        let pitch = 0.8;
        let roll = 0.5;
        
        let q = Quaternion::from_euler_zyx(yaw, pitch, roll);
        let (converted_yaw, converted_pitch, converted_roll) = q.to_euler_zyx();
        
        assert_abs_diff_eq!(yaw, converted_yaw, epsilon = 1e-6);
        assert_abs_diff_eq!(pitch, converted_pitch, epsilon = 1e-6);
        assert_abs_diff_eq!(roll, converted_roll, epsilon = 1e-6);
    }
    
    #[test]
    fn test_slerp_identity() {
        let q1 = Quaternion::one();
        let q2 = Quaternion::from_axis_angle(Vector3::unit_y(), Rad(std::f32::consts::PI / 2.0));
        
        let result_start = slerp(q1, q2, 0.0);
        let result_end = slerp(q1, q2, 1.0);
        
        assert!(result_start.approx_eq(&q1, 1e-6));
        assert!(result_end.approx_eq(&q2, 1e-6));
    }
    
    #[test]
    fn test_quaternion_from_vectors() {
        let from = Vector3::unit_x();
        let to = Vector3::unit_y();
        
        let q = Quaternion::from_vectors(from, to);
        let rotated = q * from;
        
        assert_abs_diff_eq!(rotated, to, epsilon = 1e-6);
    }
}