// src/core/math/frustum.rs
//! Frustum culling mathematics for efficient 3D rendering
//! 
//! Implements frustum extraction from projection matrices and culling
//! tests against various bounding volumes. Essential for performance
//! when rendering scenes with many objects.

use cgmath::*;

/// A viewing frustum represented as 6 planes
#[derive(Debug, Clone)]
pub struct Frustum {
    /// Frustum planes: [left, right, bottom, top, near, far]
    pub planes: [Plane; 6],
}

/// A plane in 3D space represented as ax + by + cz + d = 0
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vector3<f32>,
    pub distance: f32,
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

/// Bounding sphere
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingSphere {
    pub center: Point3<f32>,
    pub radius: f32,
}

/// Oriented bounding box (OBB)
#[derive(Debug, Clone)]
pub struct OrientedBoundingBox {
    pub center: Point3<f32>,
    pub axes: [Vector3<f32>; 3], // Local coordinate axes
    pub extents: Vector3<f32>,   // Half-widths along each axis
}

/// Result of a frustum culling test
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CullingResult {
    /// Object is completely outside the frustum
    Outside,
    /// Object is completely inside the frustum
    Inside,
    /// Object intersects the frustum boundary
    Intersecting,
}

impl Plane {
    /// Create a plane from a point and normal vector
    pub fn from_point_normal(point: Point3<f32>, normal: Vector3<f32>) -> Self {
        let normal = normal.normalize();
        let distance = -normal.dot(point.to_vec());
        Self { normal, distance }
    }

    /// Create a plane from three points (counter-clockwise)
    pub fn from_points(p1: Point3<f32>, p2: Point3<f32>, p3: Point3<f32>) -> Self {
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let normal = v1.cross(v2).normalize();
        let distance = -normal.dot(p1.to_vec());
        Self { normal, distance }
    }

    /// Calculate signed distance from point to plane
    /// Positive = in front of plane, negative = behind plane
    pub fn distance_to_point(&self, point: Point3<f32>) -> f32 {
        self.normal.dot(point.to_vec()) + self.distance
    }

    /// Test if a point is in front of the plane (positive side)
    pub fn is_point_in_front(&self, point: Point3<f32>) -> bool {
        self.distance_to_point(point) >= 0.0
    }

    /// Normalize the plane equation
    pub fn normalize(&mut self) {
        let length = self.normal.magnitude();
        if length > f32::EPSILON {
            self.normal /= length;
            self.distance /= length;
        }
    }
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix
    /// Uses the Gribb-Hartmann method
    pub fn from_view_projection(view_proj: Matrix4<f32>) -> Self {
        let m = view_proj;
        
        // Extract the planes using the standard method
        let mut planes = [
            // Left plane: m[3] + m[0]
            Plane {
                normal: Vector3::new(m.w.x + m.x.x, m.w.y + m.x.y, m.w.z + m.x.z),
                distance: m.w.w + m.x.w,
            },
            // Right plane: m[3] - m[0]
            Plane {
                normal: Vector3::new(m.w.x - m.x.x, m.w.y - m.x.y, m.w.z - m.x.z),
                distance: m.w.w - m.x.w,
            },
            // Bottom plane: m[3] + m[1]
            Plane {
                normal: Vector3::new(m.w.x + m.y.x, m.w.y + m.y.y, m.w.z + m.y.z),
                distance: m.w.w + m.y.w,
            },
            // Top plane: m[3] - m[1]
            Plane {
                normal: Vector3::new(m.w.x - m.y.x, m.w.y - m.y.y, m.w.z - m.y.z),
                distance: m.w.w - m.y.w,
            },
            // Near plane: m[3] + m[2]
            Plane {
                normal: Vector3::new(m.w.x + m.z.x, m.w.y + m.z.y, m.w.z + m.z.z),
                distance: m.w.w + m.z.w,
            },
            // Far plane: m[3] - m[2]
            Plane {
                normal: Vector3::new(m.w.x - m.z.x, m.w.y - m.z.y, m.w.z - m.z.z),
                distance: m.w.w - m.z.w,
            },
        ];

        // Normalize all planes
        for plane in &mut planes {
            plane.normalize();
        }

        Self { planes }
    }

    /// Test if a point is inside the frustum
    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        for plane in &self.planes {
            if !plane.is_point_in_front(point) {
                return false;
            }
        }
        true
    }

    /// Test a bounding sphere against the frustum
    pub fn cull_sphere(&self, sphere: &BoundingSphere) -> CullingResult {
        let mut inside_count = 0;

        for plane in &self.planes {
            let distance = plane.distance_to_point(sphere.center);
            
            if distance < -sphere.radius {
                // Sphere is completely behind this plane
                return CullingResult::Outside;
            } else if distance > sphere.radius {
                // Sphere is completely in front of this plane
                inside_count += 1;
            }
            // Otherwise, sphere intersects this plane
        }

        if inside_count == 6 {
            CullingResult::Inside
        } else {
            CullingResult::Intersecting
        }
    }

    /// Test an axis-aligned bounding box against the frustum
    pub fn cull_aabb(&self, aabb: &BoundingBox) -> CullingResult {
        let mut inside_count = 0;

        for plane in &self.planes {
            // Find the positive vertex (furthest along plane normal)
            let positive_vertex = Point3::new(
                if plane.normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if plane.normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if plane.normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );

            // Find the negative vertex (furthest against plane normal)
            let negative_vertex = Point3::new(
                if plane.normal.x >= 0.0 { aabb.min.x } else { aabb.max.x },
                if plane.normal.y >= 0.0 { aabb.min.y } else { aabb.max.y },
                if plane.normal.z >= 0.0 { aabb.min.z } else { aabb.max.z },
            );

            // Test negative vertex first (early rejection)
            if plane.distance_to_point(negative_vertex) < 0.0 {
                return CullingResult::Outside;
            }
            
            // Test positive vertex
            if plane.distance_to_point(positive_vertex) >= 0.0 {
                inside_count += 1;
            }
        }

        if inside_count == 6 {
            CullingResult::Inside
        } else {
            CullingResult::Intersecting
        }
    }

    /// Test an oriented bounding box against the frustum
    pub fn cull_obb(&self, obb: &OrientedBoundingBox) -> CullingResult {
        let mut inside_count = 0;

        for plane in &self.planes {
            // Project OBB onto plane normal
            let radius = obb.extents.x * plane.normal.dot(obb.axes[0]).abs() +
                        obb.extents.y * plane.normal.dot(obb.axes[1]).abs() +
                        obb.extents.z * plane.normal.dot(obb.axes[2]).abs();

            let distance = plane.distance_to_point(obb.center);

            if distance < -radius {
                return CullingResult::Outside;
            } else if distance > radius {
                inside_count += 1;
            }
        }

        if inside_count == 6 {
            CullingResult::Inside
        } else {
            CullingResult::Intersecting
        }
    }

    /// Get the corners of the frustum in world space
    pub fn get_corners(&self) -> [Point3<f32>; 8] {
        // This is complex to implement correctly, so for now return placeholder
        // In a production system, you'd calculate the intersection points of the planes
        [Point3::origin(); 8]
    }
}

impl BoundingBox {
    /// Create a bounding box from min and max points
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }

    /// Create a bounding box from a center point and extents
    pub fn from_center_extents(center: Point3<f32>, extents: Vector3<f32>) -> Self {
        Self {
            min: center - extents,
            max: center + extents,
        }
    }

    /// Create an empty bounding box (invalid state, useful for incremental construction)
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    /// Create a bounding box that encompasses all given points
    pub fn from_points(points: &[Point3<f32>]) -> Self {
        if points.is_empty() {
            return Self::empty();
        }

        let mut bbox = Self::empty();
        for &point in points {
            bbox.expand_to_include_point(point);
        }
        bbox
    }

    /// Check if the bounding box is valid (min <= max)
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
    }

    /// Get the center point of the bounding box
    pub fn center(&self) -> Point3<f32> {
        Point3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    /// Get the extents (half-sizes) of the bounding box
    pub fn extents(&self) -> Vector3<f32> {
        Vector3::new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        )
    }

    /// Get the full size of the bounding box
    pub fn size(&self) -> Vector3<f32> {
        Vector3::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }

    /// Get the surface area of the bounding box
    pub fn surface_area(&self) -> f32 {
        let size = self.size();
        2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)
    }

    /// Get the volume of the bounding box
    pub fn volume(&self) -> f32 {
        let size = self.size();
        size.x * size.y * size.z
    }

    /// Expand the bounding box to include a point
    pub fn expand_to_include_point(&mut self, point: Point3<f32>) {
        self.min.x = self.min.x.min(point.x);
        self.min.y = self.min.y.min(point.y);
        self.min.z = self.min.z.min(point.z);
        self.max.x = self.max.x.max(point.x);
        self.max.y = self.max.y.max(point.y);
        self.max.z = self.max.z.max(point.z);
    }

    /// Expand the bounding box to include another bounding box
    pub fn expand_to_include_box(&mut self, other: &BoundingBox) {
        self.expand_to_include_point(other.min);
        self.expand_to_include_point(other.max);
    }

    /// Test if this bounding box intersects another
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        self.max.x >= other.min.x && self.min.x <= other.max.x &&
        self.max.y >= other.min.y && self.min.y <= other.max.y &&
        self.max.z >= other.min.z && self.min.z <= other.max.z
    }

    /// Test if this bounding box contains a point
    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }

    /// Transform the bounding box by a transformation matrix
    pub fn transform(&self, matrix: Matrix4<f32>) -> Self {
        if !self.is_valid() {
            return *self;
        }

        let corners = [
            Point3::new(self.min.x, self.min.y, self.min.z),
            Point3::new(self.max.x, self.min.y, self.min.z),
            Point3::new(self.min.x, self.max.y, self.min.z),
            Point3::new(self.max.x, self.max.y, self.min.z),
            Point3::new(self.min.x, self.min.y, self.max.z),
            Point3::new(self.max.x, self.min.y, self.max.z),
            Point3::new(self.min.x, self.max.y, self.max.z),
            Point3::new(self.max.x, self.max.y, self.max.z),
        ];

        let transformed_corners: Vec<Point3<f32>> = corners
            .iter()
            .map(|&corner| matrix.transform_point(corner))
            .collect();

        Self::from_points(&transformed_corners)
    }

    /// Get the bounding sphere that encompasses this box
    pub fn to_bounding_sphere(&self) -> BoundingSphere {
        let center = self.center();
        let radius = (self.max - center).magnitude();
        BoundingSphere { center, radius }
    }
}

impl BoundingSphere {
    /// Create a new bounding sphere
    pub fn new(center: Point3<f32>, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Create a bounding sphere from a set of points
    pub fn from_points(points: &[Point3<f32>]) -> Self {
        if points.is_empty() {
            return Self::new(Point3::origin(), 0.0);
        }

        // Use Ritter's algorithm for a fast (but not optimal) bounding sphere
        let mut center = points[0];
        let mut radius = 0.0;

        // Find initial diameter
        let mut max_dist = 0.0;
        let mut p1 = points[0];
        let mut p2 = points[0];

        for &point in points {
            for &other_point in points {
                let dist = (point - other_point).magnitude();
                if dist > max_dist {
                    max_dist = dist;
                    p1 = point;
                    p2 = other_point;
                }
            }
        }

        center = Point3::from_vec((p1.to_vec() + p2.to_vec()) * 0.5);
        radius = max_dist * 0.5;

        // Expand to include all points
        for &point in points {
            let dist = (point - center).magnitude();
            if dist > radius {
                let old_radius = radius;
                radius = (old_radius + dist) * 0.5;
                let k = (radius - old_radius) / dist;
                center = Point3::from_vec(center.to_vec() + (point - center) * k);
            }
        }

        Self { center, radius }
    }

    /// Test if this sphere intersects another sphere
    pub fn intersects(&self, other: &BoundingSphere) -> bool {
        let distance = (self.center - other.center).magnitude();
        distance <= self.radius + other.radius
    }

    /// Test if this sphere contains a point
    pub fn contains_point(&self, point: Point3<f32>) -> bool {
        (point - self.center).magnitude() <= self.radius
    }

    /// Transform the sphere (only uniform scaling is preserved)
    pub fn transform(&self, matrix: Matrix4<f32>) -> Self {
        let transformed_center = matrix.transform_point(self.center);
        
        // For radius, we need to account for scaling
        // This is an approximation - it assumes uniform scaling
        let scale_vector = matrix.transform_vector(Vector3::new(1.0, 0.0, 0.0));
        let scale = scale_vector.magnitude();
        
        Self {
            center: transformed_center,
            radius: self.radius * scale,
        }
    }
}

/// Utility functions for culling operations
pub mod culling_utils {
    use super::*;

    /// Hierarchical culling: test bounding sphere first, then AABB if needed
    /// This is faster for objects that are likely to be culled
    pub fn hierarchical_cull(
        frustum: &Frustum,
        sphere: &BoundingSphere,
        aabb: &BoundingBox,
    ) -> CullingResult {
        // Test sphere first (cheaper)
        match frustum.cull_sphere(sphere) {
            CullingResult::Outside => CullingResult::Outside,
            CullingResult::Inside => {
                // Sphere is inside, but AABB might be more precise
                frustum.cull_aabb(aabb)
            }
            CullingResult::Intersecting => {
                // Test AABB for better precision
                frustum.cull_aabb(aabb)
            }
        }
    }

    /// Calculate level-of-detail based on distance to camera
    pub fn calculate_lod(
        object_center: Point3<f32>,
        camera_position: Point3<f32>,
        lod_distances: &[f32],
    ) -> usize {
        let distance = (object_center - camera_position).magnitude();
        
        for (i, &lod_distance) in lod_distances.iter().enumerate() {
            if distance <= lod_distance {
                return i;
            }
        }
        
        lod_distances.len() // Return highest LOD level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::assert_abs_diff_eq;

    #[test]
    fn test_bounding_box_creation() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-1.0, -1.0, -1.0),
        ];
        
        let bbox = BoundingBox::from_points(&points);
        assert_abs_diff_eq!(bbox.min, Point3::new(-1.0, -1.0, -1.0));
        assert_abs_diff_eq!(bbox.max, Point3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_bounding_sphere_contains_point() {
        let sphere = BoundingSphere::new(Point3::origin(), 1.0);
        
        assert!(sphere.contains_point(Point3::new(0.5, 0.5, 0.5)));
        assert!(!sphere.contains_point(Point3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn test_plane_distance() {
        let plane = Plane::from_point_normal(Point3::origin(), Vector3::unit_y());
        
        assert_abs_diff_eq!(plane.distance_to_point(Point3::new(0.0, 1.0, 0.0)), 1.0);
        assert_abs_diff_eq!(plane.distance_to_point(Point3::new(0.0, -1.0, 0.0)), -1.0);
    }
}