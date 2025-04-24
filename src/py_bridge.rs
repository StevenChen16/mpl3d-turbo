//! Python Binding Module
//!
//! Provides interfaces for Python interaction, exposing Rust's
//! high-performance 3D plotting capabilities to Python and Matplotlib.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray2};
use numpy::ToPyArray;
use pyo3::PyObject;  // 修复导入路径
use crate::proj3d::{ProjectionMatrix, Projection};
use crate::poly3d::{Poly3DCollection, ZSortMethod};
use ndarray::{Array1, Array2};


/// Matplotlib 3D plotting functionality implemented in Rust
#[pymodule]
fn mpl3d_turbo(_py: Python, m: &PyModule) -> PyResult<()> {
    // Export version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
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
            [tx_py.to_object(py), ty_py.to_object(py), tz_py.to_object(py)]
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
        let (tx, ty, tz, visible) = Projection::proj_transform_clip_vec(&vec, &proj_matrix, focal_length);
        
        // 返回结果
        let tx_py = tx.to_pyarray(py);
        let ty_py = ty.to_pyarray(py);
        let tz_py = tz.to_pyarray(py);
        let visible_py = visible.to_pyarray(py);
        
        let tuple = PyTuple::new(
            py, 
            [tx_py.to_object(py), ty_py.to_object(py), tz_py.to_object(py), visible_py.to_object(py)]
        );
        Ok(tuple.into())
    }
    
    /// Inverse projection transformation
    #[staticmethod]
    fn inv_transform(
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
        
        // 执行逆投影
        let vec = [xs_arr, ys_arr, zs_arr];
        let (u, v, w) = Projection::inv_transform_vec(&vec, &proj_matrix);
        
        // 转换结果为Python对象
        let u_py = u.to_pyarray(py);
        let v_py = v.to_pyarray(py);
        let w_py = w.to_pyarray(py);
        
        let tuple = PyTuple::new(
            py, 
            [u_py.to_object(py), v_py.to_object(py), w_py.to_object(py)]
        );
        Ok(tuple.into())
    }
}

/// Python wrapper for Poly3DCollection
#[pyclass]
pub struct PyPoly3DCollection {
    poly3d: Poly3DCollection,
}

#[pymethods]
impl PyPoly3DCollection {
    /// Create a new PyPoly3DCollection from vertex arrays
    #[new]
    fn new(
        py: Python,
        vertices: &PyAny,
        facecolors: Option<&PyAny>,
        edgecolors: Option<&PyAny>,
    ) -> PyResult<Self> {
        // 转换Python的多边形顶点数组为Rust对象
        let list = vertices.downcast::<PyList>()?;
        let mut poly_vertices = Vec::with_capacity(list.len());
        
        for item in list.iter() {
            let vert_arr = item.downcast::<PyArray2<f64>>()?;
            let shape = vert_arr.shape();
            let mut ndarray_vert = Array2::<f64>::zeros((shape[0], shape[1]));
            
            // 逐元素复制数据
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    // 先获取Option<&f64>，然后用?处理Option，再解引用
                    let value = unsafe { vert_arr.get([i, j]) }
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                            format!("Index out of bounds: [{}, {}]", i, j)
                        ))?;
                    ndarray_vert[[i, j]] = *value;
                }
            }
            
            poly_vertices.push(ndarray_vert);
        }
        
        // 转换face colors
        let fcolors = if let Some(fc) = facecolors {
            let fc_array = fc.downcast::<PyArray2<f64>>()?;
            let shape = fc_array.shape();
            let mut colors = Vec::with_capacity(shape[0]);
            
            for i in 0..shape[0] {
                let mut color = [0.0; 4];
                for j in 0..4 {
                    if j < shape[1] {
                        // 先获取Option<&f64>，然后用?处理Option，再解引用
                        let value = unsafe { fc_array.get([i, j]) }
                            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                                format!("Facecolor index out of bounds: [{}, {}]", i, j)
                            ))?;
                        color[j] = *value;
                    } else {
                        color[j] = 1.0; // Alpha默认为1.0
                    }
                }
                colors.push(color);
            }
            colors
        } else {
            vec![[0.0, 0.0, 0.0, 1.0]]
        };
        
        // 转换edge colors
        let ecolors = if let Some(ec) = edgecolors {
            let ec_array = ec.downcast::<PyArray2<f64>>()?;
            let shape = ec_array.shape();
            let mut colors = Vec::with_capacity(shape[0]);
            
            for i in 0..shape[0] {
                let mut color = [0.0; 4];
                for j in 0..4 {
                    if j < shape[1] {
                        // 先获取Option<&f64>，然后用?处理Option，再解引用
                        let value = unsafe { ec_array.get([i, j]) }
                            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                                format!("Edgecolor index out of bounds: [{}, {}]", i, j)
                            ))?;
                        color[j] = *value;
                    } else {
                        color[j] = 1.0; // Alpha默认为1.0
                    }
                }
                colors.push(color);
            }
            colors
        } else {
            vec![[0.0, 0.0, 0.0, 1.0]]
        };
        
        // 创建Poly3DCollection
        let poly3d = Poly3DCollection::new(poly_vertices, fcolors, ecolors);
        
        Ok(PyPoly3DCollection { poly3d })
    }
    
    /// Perform 3D projection for the collection
    fn do_3d_projection(&mut self, matrix: PyReadonlyArray2<f64>) -> PyResult<f64> {
        // 转换矩阵
        let matrix_arr = Array2::from_shape_vec(
            (matrix.shape()[0], matrix.shape()[1]),
            matrix.as_slice()?.to_vec()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Cannot convert matrix: {:?}", e)
        ))?;
        
        let proj_matrix = ProjectionMatrix(matrix_arr);
        
        // 执行投影
        let z_depth = self.poly3d.do_3d_projection(&proj_matrix);
        
        Ok(z_depth)
    }
    
    /// Get the sorted segments in 2D projection space
    fn get_sorted_segments_2d(&self, py: Python) -> PyResult<Py<PyList>> {
        let segments = self.poly3d.get_sorted_segments_2d();
        
        let list = PyList::empty(py);
        for segment in segments {
            list.append(segment.to_pyarray(py))?;
        }
        
        Ok(list.into())
    }
    
    /// Get sorted face colors
    fn get_sorted_facecolors(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let colors = Array2::from_shape_vec(
            (self.poly3d.facecolors.len(), 4),
            self.poly3d.facecolors.iter().flat_map(|c| c.to_vec()).collect()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Cannot convert colors: {:?}", e)
        ))?;
        
        Ok(colors.to_pyarray(py).into())
    }
    
    /// Get sorted edge colors
    fn get_sorted_edgecolors(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let colors = Array2::from_shape_vec(
            (self.poly3d.edgecolors.len(), 4),
            self.poly3d.edgecolors.iter().flat_map(|c| c.to_vec()).collect()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Cannot convert colors: {:?}", e)
        ))?;
        
        Ok(colors.to_pyarray(py).into())
    }
    
    /// Set Z-sort method
    fn set_zsort(&mut self, zsort: &str) -> PyResult<()> {
        let method = match zsort {
            "average" => ZSortMethod::Average,
            "min" => ZSortMethod::Min,
            "max" => ZSortMethod::Max,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid zsort method: {}", zsort)
            )),
        };
        
        self.poly3d.set_zsort(method);
        Ok(())
    }
    
    /// Set the position used for Z-sorting
    fn set_sort_zpos(&mut self, zpos: f64) -> PyResult<()> {
        self.poly3d.set_sort_zpos(zpos);
        Ok(())
    }
    
    /// Shade colors based on light direction
    fn shade_colors(&mut self, x: f64, y: f64, z: f64) -> PyResult<()> {
        self.poly3d.shade_colors([x, y, z]);
        Ok(())
    }
}