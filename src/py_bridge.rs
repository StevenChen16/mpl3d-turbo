//! Python Binding Module
//!
//! Provides interfaces for Python interaction, exposing Rust's
//! high-performance 3D plotting capabilities to Python and Matplotlib.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple, PyBool};
use numpy::{PyArray, PyReadonlyArray1, PyReadonlyArray2, PyArray1, PyArray2};
use crate::proj3d::{ProjectionMatrix, Projection};
use crate::poly3d::{Poly3DCollection, ZSortMethod};
use ndarray::{Array1, Array2};

/// Matplotlib 3D plotting functionality implemented in Rust
#[pymodule]
fn mpl3d_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Export version information
    m.add("__version__", crate::VERSION)?;
    
    // Register Python types
    m.add_class::<PyProjection>()?;
    m.add_class::<PyPoly3DCollection>()?;
    
    Ok(())
}

/// Python version of ProjectionMatrix and Projection functionality
#[pyclass]
struct PyProjection;

#[pymethods]
impl PyProjection {
    /// Create world transformation matrix
    #[staticmethod]
    fn world_transformation(
        py: Python,
        xmin: f64, xmax: f64,
        ymin: f64, ymax: f64,
        zmin: f64, zmax: f64,
        pb_aspect: Option<Vec<f64>>,
    ) -> PyResult<PyObject> {
        let pb_aspect_arr = pb_aspect.map(|arr| {
            if arr.len() == 3 {
                Some([arr[0], arr[1], arr[2]])
            } else {
                None
            }
        }).flatten();
        
        let matrix = ProjectionMatrix::world_transformation(
            xmin, xmax, ymin, ymax, zmin, zmax, pb_aspect_arr
        );
        
        // 转换为NumPy数组
        Ok(matrix.0.to_pyarray(py).into())
    }
    
    /// Create perspective projection matrix
    #[staticmethod]
    fn persp_transformation(
        py: Python,
        zfront: f64,
        zback: f64,
        focal_length: f64,
    ) -> PyResult<PyObject> {
        let matrix = ProjectionMatrix::persp_transformation(zfront, zback, focal_length);
        
        // 转换为NumPy数组
        Ok(matrix.0.to_pyarray(py).into())
    }
    
    /// Create orthographic projection matrix
    #[staticmethod]
    fn ortho_transformation(
        py: Python,
        zfront: f64,
        zback: f64,
    ) -> PyResult<PyObject> {
        let matrix = ProjectionMatrix::ortho_transformation(zfront, zback);
        
        // 转换为NumPy数组
        Ok(matrix.0.to_pyarray(py).into())
    }
    
    /// Perform projection transformation
    #[staticmethod]
    fn proj_transform(
        py: Python,
        xs: PyReadonlyArray1<f64>,
        ys: PyReadonlyArray1<f64>,
        zs: PyReadonlyArray1<f64>,
        matrix: PyReadonlyArray2<f64>,
    ) -> PyResult<Py<PyTuple>> {
        // 从NumPy数组转换为ndarray
        let xs_arr = Array1::from_vec(xs.as_slice()?.to_vec());
        let ys_arr = Array1::from_vec(ys.as_slice()?.to_vec());
        let zs_arr = Array1::from_vec(zs.as_slice()?.to_vec());
        
        // 创建矩阵
        let matrix_arr = Array2::from_shape_vec(
            (matrix.shape()[0], matrix.shape()[1]),
            matrix.as_slice()?.to_vec()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("无法转换矩阵: {:?}", e)
        ))?;
        
        let proj_matrix = ProjectionMatrix(matrix_arr);
        
        // 执行投影
        let vec = [xs_arr, ys_arr, zs_arr];
        let (tx, ty, tz) = Projection::proj_transform_vec(&vec, &proj_matrix);
        
        // 返回结果
        let tx_py = tx.to_pyarray(py);
        let ty_py = ty.to_pyarray(py);
        let tz_py = tz.to_pyarray(py);
        
        let tuple = PyTuple::new(
            py, 
            [tx_py.into_py(py), ty_py.into_py(py), tz_py.into_py(py)]
        );
        Ok(tuple.into())
    }
    
    /// Projection transformation with clipping
    #[staticmethod]
    fn proj_transform_clip(
        py: Python,
        xs: PyReadonlyArray1<f64>,
        ys: PyReadonlyArray1<f64>,
        zs: PyReadonlyArray1<f64>,
        matrix: PyReadonlyArray2<f64>,
        focal_length: f64,
    ) -> PyResult<Py<PyTuple>> {
        // 从NumPy数组转换为ndarray
        let xs_arr = Array1::from_vec(xs.as_slice()?.to_vec());
        let ys_arr = Array1::from_vec(ys.as_slice()?.to_vec());
        let zs_arr = Array1::from_vec(zs.as_slice()?.to_vec());
        
        // 创建矩阵
        let matrix_arr = Array2::from_shape_vec(
            (matrix.shape()[0], matrix.shape()[1]),
            matrix.as_slice()?.to_vec()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("无法转换矩阵: {:?}", e)
        ))?;
        
        let proj_matrix = ProjectionMatrix(matrix_arr);
        
        // 执行投影
        let vec = [xs_arr, ys_arr, zs_arr];
        let (tx, ty, tz, visible) = Projection::proj_transform_vec_clip(
            &vec, &proj_matrix, focal_length
        );
        
        // 返回结果
        let tx_py = tx.to_pyarray(py);
        let ty_py = ty.to_pyarray(py);
        let tz_py = tz.to_pyarray(py);
        let visible_py = visible.to_pyarray(py);
        
        let tuple = PyTuple::new(
            py, 
            [tx_py.into_py(py), ty_py.into_py(py), tz_py.into_py(py), visible_py.into_py(py)]
        );
        Ok(tuple.into())
    }
    
    /// View axes calculation
    #[staticmethod]
    fn view_axes(
        py: Python,
        eye_x: f64, eye_y: f64, eye_z: f64,
        center_x: f64, center_y: f64, center_z: f64,
        up_x: f64, up_y: f64, up_z: f64,
        roll: f64,
    ) -> PyResult<Py<PyTuple>> {
        let eye = [eye_x, eye_y, eye_z];
        let center = [center_x, center_y, center_z];
        let up = [up_x, up_y, up_z];
        
        let (u, v, w) = Projection::view_axes(&eye, &center, &up, roll);
        
        // 转换为NumPy数组
        let u_py = PyArray::from_slice(py, &u);
        let v_py = PyArray::from_slice(py, &v);
        let w_py = PyArray::from_slice(py, &w);
        
        let tuple = PyTuple::new(
            py, 
            [u_py.into_py(py), v_py.into_py(py), w_py.into_py(py)]
        );
        Ok(tuple.into())
    }
    
    /// View transformation matrix
    #[staticmethod]
    fn view_transformation_uvw(
        py: Python,
        u_x: f64, u_y: f64, u_z: f64,
        v_x: f64, v_y: f64, v_z: f64,
        w_x: f64, w_y: f64, w_z: f64,
        eye_x: f64, eye_y: f64, eye_z: f64,
    ) -> PyResult<PyObject> {
        let u = [u_x, u_y, u_z];
        let v = [v_x, v_y, v_z];
        let w = [w_x, w_y, w_z];
        let eye = [eye_x, eye_y, eye_z];
        
        let matrix = Projection::view_transformation_uvw(&u, &v, &w, &eye);
        
        // 转换为NumPy数组
        Ok(matrix.0.to_pyarray(py).into())
    }
}

/// Python版本的Poly3DCollection
#[pyclass]
struct PyPoly3DCollection {
    inner: Poly3DCollection,
}

#[pymethods]
impl PyPoly3DCollection {
    /// 创建新的3D多边形集合
    #[new]
    fn new(py: Python, verts: &PyList, facecolors: Option<&PyList>, edgecolors: Option<&PyList>) -> PyResult<Self> {
        // 转换顶点列表
        let mut vertices = Vec::new();
        for vert in verts.iter() {
            let vert_arr = vert.extract::<PyReadonlyArray2<f64>>()?;
            let shape = vert_arr.shape();
            let mut ndarray_vert = Array2::<f64>::zeros((shape[0], shape[1]));
            
            // 复制数据
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    ndarray_vert[[i, j]] = vert_arr.get([i, j])
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                            format!("Index out of bounds: [{}, {}]", i, j)
                        ))?;
                }
            }
            
            vertices.push(ndarray_vert);
        }
        
        // 转换面颜色
        let face_colors = if let Some(colors) = facecolors {
            let mut result = Vec::new();
            for color in colors.iter() {
                let color_tuple = color.extract::<Vec<f64>>()?;
                if color_tuple.len() < 3 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "颜色必须至少有3个元素 (RGB)"
                    ));
                }
                
                let alpha = if color_tuple.len() > 3 { color_tuple[3] } else { 1.0 };
                result.push([color_tuple[0], color_tuple[1], color_tuple[2], alpha]);
            }
            result
        } else {
            vec![[0.5, 0.5, 0.5, 1.0]] // 默认灰色
        };
        
        // 转换边颜色
        let edge_colors = if let Some(colors) = edgecolors {
            let mut result = Vec::new();
            for color in colors.iter() {
                let color_tuple = color.extract::<Vec<f64>>()?;
                if color_tuple.len() < 3 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "颜色必须至少有3个元素 (RGB)"
                    ));
                }
                
                let alpha = if color_tuple.len() > 3 { color_tuple[3] } else { 1.0 };
                result.push([color_tuple[0], color_tuple[1], color_tuple[2], alpha]);
            }
            result
        } else {
            vec![[0.0, 0.0, 0.0, 1.0]] // 默认黑色
        };
        
        // 创建集合
        Ok(Self {
            inner: Poly3DCollection::new(vertices, face_colors, edge_colors),
        })
    }
    
    /// 设置Z排序方法
    fn set_zsort(&mut self, zsort: &str) -> PyResult<()> {
        let method = match zsort {
            "average" => ZSortMethod::Average,
            "min" => ZSortMethod::Min,
            "max" => ZSortMethod::Max,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("无效的zsort值: {}", zsort)
            )),
        };
        
        self.inner.set_zsort(method);
        Ok(())
    }
    
    /// 设置自定义Z排序位置
    fn set_sort_zpos(&mut self, zpos: f64) -> PyResult<()> {
        self.inner.set_sort_zpos(zpos);
        Ok(())
    }
    
    /// 执行3D投影
    fn do_3d_projection(&mut self, _py: Python, matrix: PyReadonlyArray2<f64>) -> PyResult<f64> {
        // 创建投影矩阵
        let matrix_arr = Array2::from_shape_vec(
            (matrix.shape()[0], matrix.shape()[1]),
            matrix.as_slice()?.to_vec()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("无法转换矩阵: {:?}", e)
        ))?;
        
        let proj_matrix = ProjectionMatrix(matrix_arr);
        
        // 执行投影
        Ok(self.inner.do_3d_projection(&proj_matrix))
    }
    
    /// 获取排序后的2D段
    fn get_sorted_segments_2d(&self, py: Python) -> PyResult<Py<PyList>> {
        let segments = self.inner.get_sorted_segments_2d();
        let list = PyList::empty(py);
        
        for segment in segments {
            list.append(segment.to_pyarray(py))?;
        }
        
        Ok(list.into())
    }
    
    /// Get sorted face colors
    fn get_sorted_facecolors(&self, py: Python) -> PyResult<Py<PyList>> {
        let colors = self.inner.get_sorted_facecolors();
        let list = PyList::empty(py);
        
        for color in colors {
            let color_tuple = PyTuple::new(py, &[
                color[0].into_py(py),
                color[1].into_py(py),
                color[2].into_py(py),
                color[3].into_py(py),
            ]);
            list.append(color_tuple)?;
        }
        
        Ok(list.into())
    }
    
    /// Get sorted edge colors
    fn get_sorted_edgecolors(&self, py: Python) -> PyResult<Py<PyList>> {
        let colors = self.inner.get_sorted_edgecolors();
        let list = PyList::empty(py);
        
        for color in colors {
            let color_tuple = PyTuple::new(py, &[
                color[0].into_py(py),
                color[1].into_py(py),
                color[2].into_py(py),
                color[3].into_py(py),
            ]);
            list.append(color_tuple)?;
        }
        
        Ok(list.into())
    }
    
    /// Apply lighting effects
    fn shade_colors(&mut self, light_x: f64, light_y: f64, light_z: f64) -> PyResult<()> {
        self.inner.shade_colors([light_x, light_y, light_z]);
        Ok(())
    }
}
