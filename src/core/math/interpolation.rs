// src/core/math/interpolation.rs
//! Mathematical interpolation functions for animation and smooth transitions
//! 
//! Provides various interpolation methods beyond basic linear interpolation,
//! essential for smooth animation curves and easing functions.

use cgmath::*;

/// Linear interpolation between two values
pub fn lerp<T>(a: T, b: T, t: f32) -> T
where
    T: std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<f32, Output = T> + Copy,
{
    a + (b - a) * t
}

/// Spherical linear interpolation between two unit vectors
pub fn slerp_vector(a: Vector3<f32>, b: Vector3<f32>, t: f32) -> Vector3<f32> {
    let dot = a.dot(b).clamp(-1.0, 1.0);
    
    // If vectors are nearly parallel, use linear interpolation
    if dot > 0.9995 {
        return lerp(a, b, t).normalize();
    }
    
    let theta = dot.acos();
    let sin_theta = theta.sin();
    let sin_t_theta = (t * theta).sin();
    let sin_one_minus_t_theta = ((1.0 - t) * theta).sin();
    
    (a * sin_one_minus_t_theta + b * sin_t_theta) / sin_theta
}

/// Cubic Hermite spline interpolation
pub fn cubic_hermite(p0: f32, p1: f32, m0: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

/// Catmull-Rom spline interpolation (4 control points)
pub fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    
    0.5 * ((2.0 * p1) +
           (-p0 + p2) * t +
           (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
           (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Bezier curve interpolation (4 control points)
pub fn cubic_bezier(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let u = 1.0 - t;
    let u2 = u * u;
    let u3 = u2 * u;
    let t2 = t * t;
    let t3 = t2 * t;
    
    u3 * p0 + 3.0 * u2 * t * p1 + 3.0 * u * t2 * p2 + t3 * p3
}

/// Smoothstep function (cubic ease-in-out)
pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smoother step function (quintic ease-in-out)
pub fn smootherstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Collection of common easing functions
pub mod easing {
    /// Quadratic ease-in
    pub fn ease_in_quad(t: f32) -> f32 {
        t * t
    }
    
    /// Quadratic ease-out
    pub fn ease_out_quad(t: f32) -> f32 {
        1.0 - (1.0 - t) * (1.0 - t)
    }
    
    /// Quadratic ease-in-out
    pub fn ease_in_out_quad(t: f32) -> f32 {
        if t < 0.5 {
            2.0 * t * t
        } else {
            1.0 - 2.0 * (1.0 - t) * (1.0 - t)
        }
    }
    
    /// Cubic ease-in
    pub fn ease_in_cubic(t: f32) -> f32 {
        t * t * t
    }
    
    /// Cubic ease-out
    pub fn ease_out_cubic(t: f32) -> f32 {
        let u = 1.0 - t;
        1.0 - u * u * u
    }
    
    /// Cubic ease-in-out
    pub fn ease_in_out_cubic(t: f32) -> f32 {
        if t < 0.5 {
            4.0 * t * t * t
        } else {
            let u = 1.0 - t;
            1.0 - 4.0 * u * u * u
        }
    }
    
    /// Exponential ease-in
    pub fn ease_in_expo(t: f32) -> f32 {
        if t == 0.0 { 0.0 } else { 2.0_f32.powf(10.0 * (t - 1.0)) }
    }
    
    /// Exponential ease-out
    pub fn ease_out_expo(t: f32) -> f32 {
        if t == 1.0 { 1.0 } else { 1.0 - 2.0_f32.powf(-10.0 * t) }
    }
    
    /// Back ease-in (overshoots then returns)
    pub fn ease_in_back(t: f32) -> f32 {
        let c1 = 1.70158;
        let c3 = c1 + 1.0;
        c3 * t * t * t - c1 * t * t
    }
    
    /// Back ease-out
    pub fn ease_out_back(t: f32) -> f32 {
        let c1 = 1.70158;
        let c3 = c1 + 1.0;
        1.0 + c3 * (t - 1.0).powi(3) + c1 * (t - 1.0).powi(2)
    }
    
    /// Elastic ease-out (bouncy effect)
    pub fn ease_out_elastic(t: f32) -> f32 {
        use std::f32::consts::PI;
        if t == 0.0 {
            0.0
        } else if t == 1.0 {
            1.0
        } else {
            let c4 = (2.0 * PI) / 3.0;
            2.0_f32.powf(-10.0 * t) * (t * 10.0 - 0.75) * c4.sin() + 1.0
        }
    }
    
    /// Bounce ease-out
    pub fn ease_out_bounce(t: f32) -> f32 {
        let n1 = 7.5625;
        let d1 = 2.75;
        
        if t < 1.0 / d1 {
            n1 * t * t
        } else if t < 2.0 / d1 {
            let t = t - 1.5 / d1;
            n1 * t * t + 0.75
        } else if t < 2.5 / d1 {
            let t = t - 2.25 / d1;
            n1 * t * t + 0.9375
        } else {
            let t = t - 2.625 / d1;
            n1 * t * t + 0.984375
        }
    }
}

/// Multi-dimensional vector interpolation
pub trait VectorInterpolation<T> {
    fn lerp(&self, other: &Self, t: f32) -> Self;
    fn cubic_bezier(&self, control1: &Self, control2: &Self, other: &Self, t: f32) -> Self;
}

impl VectorInterpolation<f32> for Vector2<f32> {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        *self + (*other - *self) * t
    }
    
    fn cubic_bezier(&self, control1: &Self, control2: &Self, other: &Self, t: f32) -> Self {
        let u = 1.0 - t;
        let u2 = u * u;
        let u3 = u2 * u;
        let t2 = t * t;
        let t3 = t2 * t;
        
        *self * u3 + *control1 * (3.0 * u2 * t) + *control2 * (3.0 * u * t2) + *other * t3
    }
}

impl VectorInterpolation<f32> for Vector3<f32> {
    fn lerp(&self, other: &Self, t: f32) -> Self {
        *self + (*other - *self) * t
    }
    
    fn cubic_bezier(&self, control1: &Self, control2: &Self, other: &Self, t: f32) -> Self {
        let u = 1.0 - t;
        let u2 = u * u;
        let u3 = u2 * u;
        let t2 = t * t;
        let t3 = t2 * t;
        
        *self * u3 + *control1 * (3.0 * u2 * t) + *control2 * (3.0 * u * t2) + *other * t3
    }
}

/// Animation curve types for keyframe animation
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMode {
    Linear,
    Cubic,
    CubicBezier { control1: f32, control2: f32 },
    Step,
    Smoothstep,
    Smootherstep,
}

/// A keyframe for animation
#[derive(Debug, Clone)]
pub struct Keyframe<T> {
    pub time: f32,
    pub value: T,
    pub interpolation: InterpolationMode,
    pub tangent_in: Option<T>,
    pub tangent_out: Option<T>,
}

/// Animation curve for interpolating between keyframes
#[derive(Debug, Clone)]
pub struct AnimationCurve<T> {
    keyframes: Vec<Keyframe<T>>,
}

impl<T> AnimationCurve<T>
where
    T: VectorInterpolation<f32> + Copy,
{
    pub fn new(keyframes: Vec<Keyframe<T>>) -> Self {
        let mut keyframes = keyframes;
        keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
        Self { keyframes }
    }
    
    pub fn sample(&self, time: f32) -> Option<T> {
        if self.keyframes.is_empty() {
            return None;
        }
        
        if self.keyframes.len() == 1 {
            return Some(self.keyframes[0].value);
        }
        
        // Find the keyframes to interpolate between
        let mut i = 0;
        while i < self.keyframes.len() - 1 && time > self.keyframes[i + 1].time {
            i += 1;
        }
        
        if i >= self.keyframes.len() - 1 {
            return Some(self.keyframes.last().unwrap().value);
        }
        
        let k0 = &self.keyframes[i];
        let k1 = &self.keyframes[i + 1];
        
        let t = (time - k0.time) / (k1.time - k0.time);
        
        let result = match k0.interpolation {
            InterpolationMode::Linear => k0.value.lerp(&k1.value, t),
            InterpolationMode::Step => k0.value,
            InterpolationMode::Smoothstep => k0.value.lerp(&k1.value, smoothstep(t)),
            InterpolationMode::Smootherstep => k0.value.lerp(&k1.value, smootherstep(t)),
            InterpolationMode::Cubic => {
                // Use tangents if available, otherwise estimate
                if let (Some(tan_out), Some(tan_in)) = (&k0.tangent_out, &k1.tangent_in) {
                    k0.value.cubic_bezier(tan_out, tan_in, &k1.value, t)
                } else {
                    k0.value.lerp(&k1.value, smoothstep(t))
                }
            }
            InterpolationMode::CubicBezier { control1, control2 } => {
                // For vectors, this is a simplification
                k0.value.lerp(&k1.value, cubic_bezier(0.0, control1, control2, 1.0, t))
            }
        };
        
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::assert_abs_diff_eq;

    #[test]
    fn test_lerp() {
        assert_abs_diff_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_abs_diff_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_abs_diff_eq!(lerp(0.0, 10.0, 1.0), 10.0);
    }

    #[test]
    fn test_smoothstep() {
        assert_abs_diff_eq!(smoothstep(0.0), 0.0);
        assert_abs_diff_eq!(smoothstep(1.0), 1.0);
        assert_abs_diff_eq!(smoothstep(0.5), 0.5);
        
        // Should be smooth at endpoints
        let epsilon = 0.01;
        let start_slope = (smoothstep(epsilon) - smoothstep(0.0)) / epsilon;
        let end_slope = (smoothstep(1.0) - smoothstep(1.0 - epsilon)) / epsilon;
        
        assert!(start_slope.abs() < 0.1); // Nearly flat at start
        assert!(end_slope.abs() < 0.1);   // Nearly flat at end
    }

    #[test]
    fn test_vector_lerp() {
        let v1 = Vector3::new(0.0, 0.0, 0.0);
        let v2 = Vector3::new(1.0, 2.0, 3.0);
        let result = v1.lerp(&v2, 0.5);
        
        assert_abs_diff_eq!(result, Vector3::new(0.5, 1.0, 1.5));
    }

    #[test]
    fn test_animation_curve() {
        let keyframes = vec![
            Keyframe {
                time: 0.0,
                value: Vector3::new(0.0, 0.0, 0.0),
                interpolation: InterpolationMode::Linear,
                tangent_in: None,
                tangent_out: None,
            },
            Keyframe {
                time: 1.0,
                value: Vector3::new(1.0, 1.0, 1.0),
                interpolation: InterpolationMode::Linear,
                tangent_in: None,
                tangent_out: None,
            },
        ];
        
        let curve = AnimationCurve::new(keyframes);
        let result = curve.sample(0.5).unwrap();
        
        assert_abs_diff_eq!(result, Vector3::new(0.5, 0.5, 0.5));
    }
}