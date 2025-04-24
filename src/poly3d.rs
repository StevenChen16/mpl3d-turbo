//! 3D Polygon Collections
//!
//! This module implements functionality equivalent to Matplotlib's Poly3DCollection,
//! for efficiently rendering large numbers of 3D polygons.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use crate::proj3d::{ProjectionMatrix, Projection};

/// 深度排序方法
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZSortMethod {
    /// Sort using the average Z-coordinate
    Average,
    /// Sort using the minimum Z-coordinate
    Min,
    /// Sort using the maximum Z-coordinate
    Max,
}

/// 3D Polygon Collection (corresponds to Matplotlib's Poly3DCollection)
#[derive(Debug)]
pub struct Poly3DCollection {
    /// Vertex data (each polygon is an Nx3 matrix)
    pub vertices: Vec<Array2<f64>>,
    /// Face colors (RGBA, one color per polygon)
    pub facecolors: Vec<[f64; 4]>,
    /// Edge colors (RGBA, one color per polygon)
    pub edgecolors: Vec<[f64; 4]>,
    /// Depth sorting method
    pub zsort_method: ZSortMethod,
    /// Transformed 2D vertices
    pub segments_2d: Vec<Array2<f64>>,
    /// Vertex segments
    pub segments_3d: Vec<Array2<f64>>,
    /// Depth sorting position
    pub sort_zpos: Option<f64>,
    /// Projected Z-value cache (for depth sorting)
    pub z_depths: Vec<f64>,
    /// Sorted indices
    pub sorted_indices: Vec<usize>,
}

impl Poly3DCollection {
    /// Create a new 3D polygon collection
    ///
    /// # Parameters
    /// * `verts` - List of polygon vertices
    /// * `facecolors` - Face colors
    /// * `edgecolors` - Edge colors
    pub fn new(
        verts: Vec<Array2<f64>>, 
        facecolors: Vec<[f64; 4]>, 
        edgecolors: Vec<[f64; 4]>
    ) -> Self {
        let n_polys = verts.len();
        let segments_3d = verts.clone();
        
        // Ensure each polygon has a color
        let facecolors = if facecolors.len() < n_polys && !facecolors.is_empty() {
            let default_color = facecolors[0];
            std::iter::repeat(default_color).take(n_polys).collect()
        } else {
            facecolors
        };
        
        let edgecolors = if edgecolors.len() < n_polys && !edgecolors.is_empty() {
            let default_color = edgecolors[0];
            std::iter::repeat(default_color).take(n_polys).collect()
        } else {
            edgecolors
        };
        
        Self {
            vertices: verts,
            facecolors,
            edgecolors,
            zsort_method: ZSortMethod::Average,
            segments_2d: Vec::with_capacity(n_polys),
            segments_3d,
            sort_zpos: None,
            z_depths: Vec::with_capacity(n_polys),
            sorted_indices: Vec::with_capacity(n_polys),
        }
    }
    
    /// Set the depth sorting method
    pub fn set_zsort(&mut self, zsort: ZSortMethod) {
        self.zsort_method = zsort;
        self.sort_zpos = None;
    }
    
    /// Set a custom Z sorting position
    pub fn set_sort_zpos(&mut self, zpos: f64) {
        self.sort_zpos = Some(zpos);
    }
    
    /// Perform 3D projection
    ///
    /// Projects 3D polygons to 2D while calculating depth sorting values.
    /// Returns the overall minimum Z value for Z-sorting in external rendering engines.
    pub fn do_3d_projection(&mut self, proj_matrix: &ProjectionMatrix) -> f64 {
        if self.segments_3d.is_empty() {
            return std::f64::NAN;
        }
        
        let n_polys = self.segments_3d.len();
        self.segments_2d.clear();
        self.segments_2d.resize(n_polys, Array2::zeros((0, 2)));
        
        self.z_depths.clear();
        self.z_depths.resize(n_polys, 0.0);
        
        // Use parallel processing if there are many polygons
        if n_polys > 100 {
            // Process projections in parallel first
            let mut projected_data: Vec<(Array2<f64>, f64)> = vec![(Array2::zeros((0, 2)), 0.0); n_polys];
            
            projected_data.par_iter_mut().enumerate().for_each(|(idx, (segment_2d, z_depth))| {
                if idx < self.segments_3d.len() {
                    let segment = &self.segments_3d[idx];
                    let n_points = segment.shape()[0];
                    
                    // Extract coordinates
                    let mut xs = Array1::zeros(n_points);
                    let mut ys = Array1::zeros(n_points);
                    let mut zs = Array1::zeros(n_points);
                    
                    for i in 0..n_points {
                        xs[i] = segment[[i, 0]];
                        ys[i] = segment[[i, 1]];
                        zs[i] = segment[[i, 2]];
                    }
                    
                    // Perform projection
                    let vec = [xs, ys, zs];
                    let (txs, tys, tzs) = Projection::proj_transform_vec(&vec, proj_matrix);
                    
                    // Calculate Z sorting value
                    *z_depth = match self.zsort_method {
                        ZSortMethod::Average => tzs.mean().unwrap_or(0.0),
                        ZSortMethod::Min => tzs.iter().fold(std::f64::INFINITY, |a, &b| a.min(b)),
                        ZSortMethod::Max => tzs.iter().fold(std::f64::NEG_INFINITY, |a, &b| a.max(b)),
                    };
                    
                    // Create 2D segments
                    let mut s2d = Array2::zeros((n_points, 2));
                    for i in 0..n_points {
                        s2d[[i, 0]] = txs[i];
                        s2d[[i, 1]] = tys[i];
                    }
                    *segment_2d = s2d;
                }
            });
            
            // Collect results
            for (idx, (segment_2d, z_depth)) in projected_data.into_iter().enumerate() {
                if idx < n_polys {
                    self.segments_2d[idx] = segment_2d;
                    self.z_depths[idx] = z_depth;
                }
            }
        } else {
            // Small dataset, process directly
            for (idx, segment) in self.segments_3d.iter().enumerate() {
                let n_points = segment.shape()[0];
                
                // 提取坐标
                let mut xs = Array1::zeros(n_points);
                let mut ys = Array1::zeros(n_points);
                let mut zs = Array1::zeros(n_points);
                
                for i in 0..n_points {
                    xs[i] = segment[[i, 0]];
                    ys[i] = segment[[i, 1]];
                    zs[i] = segment[[i, 2]];
                }
                
                // Perform projection
                let vec = [xs, ys, zs];
                let (txs, tys, tzs) = Projection::proj_transform_vec(&vec, proj_matrix);
                
                // Calculate Z sorting value
                let z_depth = match self.zsort_method {
                    ZSortMethod::Average => tzs.mean().unwrap_or(0.0),
                    ZSortMethod::Min => tzs.iter().fold(std::f64::INFINITY, |a, &b| a.min(b)),
                    ZSortMethod::Max => tzs.iter().fold(std::f64::NEG_INFINITY, |a, &b| a.max(b)),
                };
                
                // Create 2D segments
                let mut segment_2d = Array2::zeros((n_points, 2));
                for i in 0..n_points {
                    segment_2d[[i, 0]] = txs[i];
                    segment_2d[[i, 1]] = tys[i];
                }
                
                self.segments_2d[idx] = segment_2d;
                self.z_depths[idx] = z_depth;
            }
        }
        
        // Generate sorting indices from zfront to zback
        self.sorted_indices = (0..n_polys).collect();
        self.sorted_indices.sort_by(|&a, &b| {
            self.z_depths[b].partial_cmp(&self.z_depths[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Return Z value for overall sorting
        if let Some(sort_zpos) = self.sort_zpos {
            sort_zpos
        } else if !self.z_depths.is_empty() {
            *self.z_depths.iter().min_by(|a, b| 
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap_or(&std::f64::NAN)
        } else {
            std::f64::NAN
        }
    }
    
    /// Get sorted face colors
    pub fn get_sorted_facecolors(&self) -> Vec<[f64; 4]> {
        if self.facecolors.is_empty() {
            return Vec::new();
        }
        
        self.sorted_indices.iter()
            .map(|&idx| {
                if idx < self.facecolors.len() {
                    self.facecolors[idx]
                } else {
                    self.facecolors[0] // Fallback to the first color
                }
            })
            .collect()
    }
    
    /// Get sorted edge colors
    pub fn get_sorted_edgecolors(&self) -> Vec<[f64; 4]> {
        if self.edgecolors.is_empty() {
            return Vec::new();
        }
        
        self.sorted_indices.iter()
            .map(|&idx| {
                if idx < self.edgecolors.len() {
                    self.edgecolors[idx]
                } else {
                    self.edgecolors[0] // Fallback to the first color
                }
            })
            .collect()
    }
    
    /// Get sorted 2D segments
    pub fn get_sorted_segments_2d(&self) -> Vec<Array2<f64>> {
        self.sorted_indices.iter()
            .map(|&idx| self.segments_2d[idx].clone())
            .collect()
    }
    
    /// Generate normal vectors
    pub fn generate_normals(&self) -> Vec<[f64; 3]> {
        self.segments_3d.iter()
            .map(|poly| {
                if poly.shape()[0] < 3 {
                    return [0.0, 0.0, 1.0]; // Default value, pointing to the z-axis
                }
                
                // Use the first three points of the face to calculate the normal
                let i1 = 0;
                let i2 = poly.shape()[0] / 3;
                let i3 = 2 * poly.shape()[0] / 3;
                
                let v1 = [
                    poly[[i1, 0]] - poly[[i2, 0]],
                    poly[[i1, 1]] - poly[[i2, 1]],
                    poly[[i1, 2]] - poly[[i2, 2]],
                ];
                
                let v2 = [
                    poly[[i2, 0]] - poly[[i3, 0]],
                    poly[[i2, 1]] - poly[[i3, 1]],
                    poly[[i2, 2]] - poly[[i3, 2]],
                ];
                
                // Calculate the cross product to get the normal vector
                let normal = [
                    v1[1] * v2[2] - v1[2] * v2[1],
                    v1[2] * v2[0] - v1[0] * v2[2],
                    v1[0] * v2[1] - v1[1] * v2[0],
                ];
                
                // Normalize the normal vector
                let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if norm > 1e-10 {
                    [normal[0] / norm, normal[1] / norm, normal[2] / norm]
                } else {
                    [0.0, 0.0, 1.0] // Default normal vector
                }
            })
            .collect()
    }
    
    /// Apply lighting effects
    pub fn shade_colors(
        &mut self, 
        light_direction: [f64; 3]
    ) {
        // Normalize the light direction
        let light_dir_norm = (
            light_direction[0] * light_direction[0] + 
            light_direction[1] * light_direction[1] + 
            light_direction[2] * light_direction[2]
        ).sqrt();
        
        let light_dir = [
            light_direction[0] / light_dir_norm,
            light_direction[1] / light_dir_norm,
            light_direction[2] / light_dir_norm,
        ];
        
        let normals = self.generate_normals();
        
        for (i, normal) in normals.iter().enumerate() {
            if i >= self.facecolors.len() { continue; }
            
            // Calculate the dot product of the normal and light direction
            let intensity = normal[0] * light_dir[0] + 
                           normal[1] * light_dir[1] + 
                           normal[2] * light_dir[2];
            
            // Map the dot product from [-1, 1] to a shading intensity in [0.3, 1]
            let shade = 0.3 + 0.7 * (intensity + 1.0) / 2.0;
            
            // Apply the shading to the color
            self.facecolors[i][0] *= shade;
            self.facecolors[i][1] *= shade;
            self.facecolors[i][2] *= shade;
            
            // Also apply the shading to the edge color
            if i < self.edgecolors.len() {
                self.edgecolors[i][0] *= shade;
                self.edgecolors[i][1] *= shade;
                self.edgecolors[i][2] *= shade;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_poly3d_collection() {
        // 创建一个简单的立方体
        let verts = vec![
            // 前面
            ndarray::arr2(&[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            // 后面
            ndarray::arr2(&[
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]),
        ];
        
        let facecolors = vec![
            [1.0, 0.0, 0.0, 1.0], // 红色
            [0.0, 1.0, 0.0, 1.0], // 绿色
        ];
        
        let edgecolors = vec![
            [0.0, 0.0, 0.0, 1.0], // 黑色
            [0.0, 0.0, 0.0, 1.0], // 黑色
        ];
        
        let mut collection = Poly3DCollection::new(verts, facecolors, edgecolors);
        
        // 设置视图矩阵
        let proj = ProjectionMatrix::ortho_transformation(-1.0, 1.0);
        
        // 执行投影
        let z_min = collection.do_3d_projection(&proj);
        
        // 测试结果
        assert!(!z_min.is_nan());
        assert_eq!(collection.segments_2d.len(), 2);
        
        // 测试排序
        let sorted_faces = collection.get_sorted_facecolors();
        assert_eq!(sorted_faces.len(), 2);
    }
}