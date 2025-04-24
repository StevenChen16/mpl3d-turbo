//! 3D Projection and Transformation Functions
//! 
//! This module is equivalent to Matplotlib's `proj3d.py`,
//! providing various projection and transformation matrices used in 3D rendering.

use ndarray::{Array1, Array2};
use ndarray::prelude::s;
use rayon::prelude::*;

/// 3D Projection Matrix
#[derive(Clone, Debug)]
pub struct ProjectionMatrix(pub Array2<f64>);

impl ProjectionMatrix {
    /// Create world transformation matrix
    ///
    /// Creates a matrix that scales coordinates within the specified range to [0, 1],
    /// or if a plotbox aspect ratio is specified, scales to [0, pb_aspect[i]].
    pub fn world_transformation(
        xmin: f64, xmax: f64,
        ymin: f64, ymax: f64,
        zmin: f64, zmax: f64, 
        pb_aspect: Option<[f64; 3]>
    ) -> Self {
        let dx = xmax - xmin;
        let dy = ymax - ymin;
        let dz = zmax - zmin;
        
        let (dx, dy, dz) = if let Some([ax, ay, az]) = pb_aspect {
            (dx / ax, dy / ay, dz / az)
        } else {
            (dx, dy, dz)
        };
        
        let mat = ndarray::arr2(&[
            [1.0/dx, 0.0,    0.0,    -xmin/dx],
            [0.0,    1.0/dy, 0.0,    -ymin/dy],
            [0.0,    0.0,    1.0/dz, -zmin/dz],
            [0.0,    0.0,    0.0,    1.0     ]
        ]);
        
        Self(mat)
    }
    
    /// Create perspective projection matrix
    pub fn persp_transformation(zfront: f64, zback: f64, focal_length: f64) -> Self {
        let e = focal_length;
        let a = 1.0;  // 纵横比
        let b = (zfront + zback) / (zfront - zback);
        let c = -2.0 * (zfront * zback) / (zfront - zback);
        
        let mat = ndarray::arr2(&[
            [e,   0.0,  0.0, 0.0],
            [0.0, e/a,  0.0, 0.0],
            [0.0, 0.0,  b,   c  ],
            [0.0, 0.0, -1.0, 0.0]
        ]);
        
        Self(mat)
    }
    
    /// Create orthographic projection matrix
    pub fn ortho_transformation(zfront: f64, zback: f64) -> Self {
        let a = -(zfront + zback);
        let b = -(zfront - zback);
        
        let mat = ndarray::arr2(&[
            [2.0, 0.0,  0.0, 0.0],
            [0.0, 2.0,  0.0, 0.0],
            [0.0, 0.0, -2.0, 0.0],
            [0.0, 0.0,  a,   b  ]
        ]);
        
        Self(mat)
    }
    
    /// Calculate rotation matrix around arbitrary vector
    pub fn rotation_about_vector(v: &[f64; 3], angle: f64) -> Array2<f64> {
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let vx = v[0] / norm;
        let vy = v[1] / norm;
        let vz = v[2] / norm;
        
        let s = angle.sin();
        let c = angle.cos();
        let t = 2.0 * (angle / 2.0).sin().powi(2);  // More numerically stable way: t = 1-c
        
        ndarray::arr2(&[
            [t*vx*vx + c,    t*vx*vy - vz*s, t*vx*vz + vy*s],
            [t*vy*vx + vz*s, t*vy*vy + c,    t*vy*vz - vx*s],
            [t*vz*vx - vy*s, t*vz*vy + vx*s, t*vz*vz + c   ]
        ])
    }
}

/// 3D Vector Projection and Transformation
pub struct Projection;

impl Projection {
    /// Perform point projection transformation
    ///
    /// Transforms a set of 3D points to screen coordinates through the projection matrix
    pub fn proj_transform_vec(
        vec: &[Array1<f64>; 3], 
        m: &ProjectionMatrix
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let n = vec[0].len();
        let mut result_x = Array1::zeros(n);
        let mut result_y = Array1::zeros(n);
        let mut result_z = Array1::zeros(n);
        
        // Parallel processing for large datasets
        if n > 1000 {
            // Use simple parallel processing
            let chunk_size = n / rayon::current_num_threads().max(1);
            
            // 创建中间向量以修复生命周期问题
            let mut result_x_vec = result_x.to_vec();
            let mut result_y_vec = result_y.to_vec();
            let mut result_z_vec = result_z.to_vec();
            
            // Create mutable ranges for parallel processing
            let mut xs_chunks: Vec<_> = result_x_vec.chunks_mut(chunk_size).collect();
            let mut ys_chunks: Vec<_> = result_y_vec.chunks_mut(chunk_size).collect();
            let mut zs_chunks: Vec<_> = result_z_vec.chunks_mut(chunk_size).collect();
            
            // Parallel iteration to process each chunk
            xs_chunks.par_iter_mut().zip(ys_chunks.par_iter_mut().zip(zs_chunks.par_iter_mut()))
                .enumerate()
                .for_each(|(chunk_idx, (xs_chunk, (ys_chunk, zs_chunk)))| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(n);
                    
                    for i in start..end {
                        if i >= n { break; }
                        let local_i = i - start;
                        
                        let point = [vec[0][i], vec[1][i], vec[2][i], 1.0];
                        let transformed = Self::transform_point(&point, &m.0);
                        
                        let w = transformed[3];
                        if w != 0.0 && local_i < xs_chunk.len() {
                            xs_chunk[local_i] = transformed[0] / w;
                            ys_chunk[local_i] = transformed[1] / w;
                            zs_chunk[local_i] = transformed[2] / w;
                        }
                    }
                });
            
            // Copy processed data back to the original arrays
            let mut offset = 0;
            for chunk in xs_chunks {
                let len = chunk.len();
                if offset + len <= n {
                    result_x.slice_mut(s![offset..offset+len]).assign(&Array1::from_vec(chunk.to_vec()));
                }
                offset += len;
            }
            
            offset = 0;
            for chunk in ys_chunks {
                let len = chunk.len();
                if offset + len <= n {
                    result_y.slice_mut(s![offset..offset+len]).assign(&Array1::from_vec(chunk.to_vec()));
                }
                offset += len;
            }
            
            offset = 0;
            for chunk in zs_chunks {
                let len = chunk.len();
                if offset + len <= n {
                    result_z.slice_mut(s![offset..offset+len]).assign(&Array1::from_vec(chunk.to_vec()));
                }
                offset += len;
            }
        } else {
            // Direct processing for small datasets
            for i in 0..n {
                let point = [vec[0][i], vec[1][i], vec[2][i], 1.0];
                let transformed = Self::transform_point(&point, &m.0);
                
                let w = transformed[3];
                if w != 0.0 {
                    result_x[i] = transformed[0] / w;
                    result_y[i] = transformed[1] / w;
                    result_z[i] = transformed[2] / w;
                }
            }
        }
        
        (result_x, result_y, result_z)
    }
    
    /// Projection transformation with clipping
    pub fn proj_transform_clip_vec(
        vec: &[Array1<f64>; 3], 
        m: &ProjectionMatrix,
        focal_length: f64
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<bool>) {
        let n = vec[0].len();
        let mut result_x = Array1::zeros(n);
        let mut result_y = Array1::zeros(n);
        let mut result_z = Array1::zeros(n);
        let mut visible = Array1::from_elem(n, true);
        
        for i in 0..n {
            let point = [vec[0][i], vec[1][i], vec[2][i], 1.0];
            let transformed = Self::transform_point(&point, &m.0);
            
            let w = transformed[3];
            if w != 0.0 {
                result_x[i] = transformed[0] / w;
                result_y[i] = transformed[1] / w;
                result_z[i] = transformed[2] / w;
                
                // 应用裁剪
                if !focal_length.is_infinite() {
                    visible[i] = (-1.0 <= result_x[i]) && (result_x[i] <= 1.0) &&
                                 (-1.0 <= result_y[i]) && (result_y[i] <= 1.0) &&
                                 (result_z[i] <= 0.0);
                }
            }
        }
        
        (result_x, result_y, result_z, visible)
    }
    
    /// Inverse projection transformation
    pub fn inv_transform_vec(
        vec: &[Array1<f64>; 3],
        m: &ProjectionMatrix
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let n = vec[0].len();
        let mut result_x = Array1::zeros(n);
        let mut result_y = Array1::zeros(n);
        let mut result_z = Array1::zeros(n);
        
        // 计算逆矩阵 - 由于我们没有直接的inv方法，自己实现一个简单的矩阵求逆
        let inv_matrix = self_inverse_matrix(&m.0);
        
        for i in 0..n {
            let point = [vec[0][i], vec[1][i], vec[2][i], 1.0];
            let transformed = Self::transform_point(&point, &inv_matrix);
            
            let w = transformed[3];
            if w != 0.0 {
                result_x[i] = transformed[0] / w;
                result_y[i] = transformed[1] / w;
                result_z[i] = transformed[2] / w;
            }
        }
        
        (result_x, result_y, result_z)
    }
    
    /// Transform a single point
    pub fn transform_point(point: &[f64; 4], matrix: &Array2<f64>) -> [f64; 4] {
        let mut result = [0.0; 4];
        
        for i in 0..4 {
            for j in 0..4 {
                result[i] += matrix[[i, j]] * point[j];
            }
        }
        
        result
    }
    
    /// Get view axes
    pub fn view_axes(
        eye: &[f64; 3],
        center: &[f64; 3],
        up: &[f64; 3],
        roll: f64
    ) -> ([f64; 3], [f64; 3], [f64; 3]) {
        // 计算视图方向
        let mut w = [
            eye[0] - center[0],
            eye[1] - center[1],
            eye[2] - center[2]
        ];
        
        // 归一化w
        let w_norm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        w[0] /= w_norm;
        w[1] /= w_norm;
        w[2] /= w_norm;
        
        // 计算u和v
        let u = Self::cross_product(up, &w);
        let u_norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let u = [u[0] / u_norm, u[1] / u_norm, u[2] / u_norm];
        
        let v = Self::cross_product(&w, &u);
        
        // 应用滚动旋转
        if roll != 0.0 {
            let roll_matrix = ProjectionMatrix::rotation_about_vector(&w, -roll);
            let u_rotated = [
                roll_matrix[[0, 0]] * u[0] + roll_matrix[[0, 1]] * u[1] + roll_matrix[[0, 2]] * u[2],
                roll_matrix[[1, 0]] * u[0] + roll_matrix[[1, 1]] * u[1] + roll_matrix[[1, 2]] * u[2],
                roll_matrix[[2, 0]] * u[0] + roll_matrix[[2, 1]] * u[1] + roll_matrix[[2, 2]] * u[2]
            ];
            
            let v_rotated = [
                roll_matrix[[0, 0]] * v[0] + roll_matrix[[0, 1]] * v[1] + roll_matrix[[0, 2]] * v[2],
                roll_matrix[[1, 0]] * v[0] + roll_matrix[[1, 1]] * v[1] + roll_matrix[[1, 2]] * v[2],
                roll_matrix[[2, 0]] * v[0] + roll_matrix[[2, 1]] * v[1] + roll_matrix[[2, 2]] * v[2]
            ];
            
            return (u_rotated, v_rotated, w);
        }
        
        (u, v, w)
    }
    
    /// Create view transformation matrix
    pub fn view_transformation_uvw(
        u: &[f64; 3],
        v: &[f64; 3],
        w: &[f64; 3],
        eye: &[f64; 3]
    ) -> ProjectionMatrix {
        let mut mr = ndarray::Array2::<f64>::eye(4);
        let mut mt = ndarray::Array2::<f64>::eye(4);
        
        // 设置旋转矩阵
        mr[[0, 0]] = u[0]; mr[[0, 1]] = u[1]; mr[[0, 2]] = u[2];
        mr[[1, 0]] = v[0]; mr[[1, 1]] = v[1]; mr[[1, 2]] = v[2];
        mr[[2, 0]] = w[0]; mr[[2, 1]] = w[1]; mr[[2, 2]] = w[2];
        
        // 设置平移矩阵
        mt[[0, 3]] = -eye[0];
        mt[[1, 3]] = -eye[1];
        mt[[2, 3]] = -eye[2];
        
        // 矩阵乘法
        let m = mr.dot(&mt);
        
        ProjectionMatrix(m)
    }
    
    /// Calculate vector cross product
    fn cross_product(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ]
    }
    
    /// Generate padded array for vertices (add homogeneous coordinates)
    pub fn vec_pad_ones(
        xs: &Array1<f64>,
        ys: &Array1<f64>,
        zs: &Array1<f64>
    ) -> [Array1<f64>; 4] {
        let n = xs.len();
        let ones = Array1::ones(n);
        
        [xs.clone(), ys.clone(), zs.clone(), ones]
    }
}

// 简单的4x4矩阵求逆实现 (替代ndarray_linalg::Inverse)
fn self_inverse_matrix(matrix: &Array2<f64>) -> Array2<f64> {
    // 创建单位矩阵
    let mut result = Array2::<f64>::eye(4);
    let mut m = matrix.clone();
    
    // 高斯-约旦消元
    for i in 0..4 {
        // 查找主元
        let mut max_row = i;
        let mut max_val = m[[i, i]].abs();
        
        for j in (i+1)..4 {
            let val = m[[j, i]].abs();
            if val > max_val {
                max_row = j;
                max_val = val;
            }
        }
        
        // 交换行
        if max_row != i {
            for j in 0..4 {
                let temp = m[[i, j]];
                m[[i, j]] = m[[max_row, j]];
                m[[max_row, j]] = temp;
                
                let temp = result[[i, j]];
                result[[i, j]] = result[[max_row, j]];
                result[[max_row, j]] = temp;
            }
        }
        
        // 缩放行
        let pivot = m[[i, i]];
        for j in 0..4 {
            m[[i, j]] /= pivot;
            result[[i, j]] /= pivot;
        }
        
        // 消除其他行
        for j in 0..4 {
            if j != i {
                let factor = m[[j, i]];
                for k in 0..4 {
                    m[[j, k]] -= factor * m[[i, k]];
                    result[[j, k]] -= factor * result[[i, k]];
                }
            }
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    
    #[test]
    fn test_world_transformation() {
        let m = ProjectionMatrix::world_transformation(
            -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, None
        );
        
        // 变换点 (-1,-1,-1) 到 (0,0,0)
        let point = [-1.0, -1.0, -1.0, 1.0];
        let result = Projection::transform_point(&point, &m.0);
        
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
        
        // 变换点 (1,1,1) 到 (1,1,1)
        let point = [1.0, 1.0, 1.0, 1.0];
        let result = Projection::transform_point(&point, &m.0);
        
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
        assert!((result[3] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rotation_matrix() {
        // 旋转90度
        let axis = [0.0, 0.0, 1.0]; // z轴
        let angle = PI / 2.0;
        let r = ProjectionMatrix::rotation_about_vector(&axis, angle);
        
        // 测试旋转矩阵是否正确变换 (1,0,0) 到 (0,1,0)
        let point = [1.0, 0.0, 0.0, 1.0];
        let rotated = [
            r[[0, 0]] * point[0] + r[[0, 1]] * point[1] + r[[0, 2]] * point[2],
            r[[1, 0]] * point[0] + r[[1, 1]] * point[1] + r[[1, 2]] * point[2],
            r[[2, 0]] * point[0] + r[[2, 1]] * point[1] + r[[2, 2]] * point[2],
        ];
        
        assert!((rotated[0] - 0.0).abs() < 1e-10);
        assert!((rotated[1] - 1.0).abs() < 1e-10);
        assert!((rotated[2] - 0.0).abs() < 1e-10);
    }
}