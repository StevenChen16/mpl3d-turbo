"""
mpl3d-turbo: High-Performance Matplotlib 3D Rendering Library

This library provides a high-performance Rust implementation of Matplotlib's mpl_toolkits.mplot3d module,
focused on improving 3D visualization performance for PDE solution results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import the Python fallback implementation
from .py_fallback import py_fast_plot_surface, PyFastPoly3DCollection

# Try to import the Rust library, but use the Python fallback if it fails
USING_RUST = False
try:
    from . import mpl3d_turbo  # Import Rust library
    USING_RUST = True
    print("Successfully loaded Rust implementation of mpl3d_turbo")
except ImportError:
    print("Using Python fallback implementation for mpl3d_turbo")
    # Will use the Python fallback implementation

__version__ = "0.1.0"
if USING_RUST:
    __version__ = getattr(mpl3d_turbo, "__version__", "0.1.0")


class FastPoly3DCollection(PolyCollection):
    """Fast 3D polygon collection implemented in Rust
    
    This class provides an interface compatible with Matplotlib's Poly3DCollection,
    but uses Rust to accelerate 3D projection and depth sorting.
    Falls back to Python implementation if the Rust library is unavailable.
    
    Parameters
    ----------
    verts : sequence of array-like
        Sequence of vertices, each element is an Nx3 array defining a polygon
    *args, **kwargs
        Other arguments passed to PolyCollection
    """
    
    def __init__(self, verts, *args, **kwargs):
        super().__init__([], *args, **kwargs)
        
        self._verts3d = verts
        self._codes3d = kwargs.get('codes', None)
        self._zsort = 'average'
        self._sort_zpos = None
        
        if USING_RUST:
            # 使用Rust实现
            self._rs_collection = mpl3d_turbo.PyPoly3DCollection(
                verts,
                self.get_facecolor(),
                self.get_edgecolor()
            )
        else:
            # 使用Python回退实现
            self._py_collection = PyFastPoly3DCollection(verts, *args, **kwargs)
        
    def set_3d_properties(self):
        """Set 3D properties"""
        if not USING_RUST:
            self._py_collection.set_3d_properties()
        
    def set_sort_zpos(self, val):
        """Set the position used for Z-sorting"""
        self._sort_zpos = val
        if USING_RUST and hasattr(self, '_rs_collection'):
            self._rs_collection.set_sort_zpos(val)
        elif not USING_RUST:
            self._py_collection.set_sort_zpos(val)
        
    def set_zsort(self, zsort):
        """Set Z-sorting method
        
        Parameters
        ----------
        zsort : {'average', 'min', 'max'}
            Function to use for Z-sorting
        """
        self._zsort = zsort
        if USING_RUST and hasattr(self, '_rs_collection'):
            self._rs_collection.set_zsort(zsort)
        elif not USING_RUST:
            self._py_collection.set_zsort(zsort)
        
    def do_3d_projection(self):
        """Perform 3D projection
        
        Uses Rust implementation for high-performance projection and depth sorting,
        falls back to Python implementation if Rust is unavailable
        """
        if USING_RUST:
            # Use Rust implementation
            # Get current view matrix
            M = self.axes.get_proj()
            
            # Call Rust implementation for projection
            z_depth = self._rs_collection.do_3d_projection(M)
            
            # Get projected 2D vertices and sorted colors
            segments_2d = self._rs_collection.get_sorted_segments_2d()
            
            # Update Matplotlib collection
            super().set_verts(segments_2d)
            
            # Update colors (optional, if color mapping exists)
            if self._facecolors.size > 0:
                face_colors = self._rs_collection.get_sorted_facecolors()
                self._facecolors = np.array(face_colors)
                
            if self._edgecolors.size > 0:
                edge_colors = self._rs_collection.get_sorted_edgecolors()
                self._edgecolors = np.array(edge_colors)
            
            return z_depth
        else:
            # 使用Python回退实现
            return self._py_collection.do_3d_projection()


def fast_plot_surface(ax, X, Y, Z, *args, **kwargs):
    """Fast 3D surface plotting implemented in Rust
    Uses Python fallback implementation if Rust library is unavailable
    
    Parameters
    ----------
    ax : Axes3D
        3D axes to plot the surface on
    X, Y : 2D array-like
        Surface X and Y coordinates
    Z : 2D array-like
        Surface Z coordinates
    *args, **kwargs
        Other arguments passed to Poly3DCollection
        
    Returns
    -------
    collection : FastPoly3DCollection or PyFastPoly3DCollection
        The created collection object
    """
    if not USING_RUST:
        # 使用Python回退实现
        return py_fast_plot_surface(ax, X, Y, Z, *args, **kwargs)
    
    # Rust implementation follows
    rows, cols = Z.shape
    
    rstride = kwargs.pop('rstride', 1)
    cstride = kwargs.pop('cstride', 1)
    rcount = kwargs.pop('rcount', 50)
    ccount = kwargs.pop('ccount', 50)
    
    # Calculate stride based on count
    if rstride == 1 and rcount < rows:
        rstride = max(int(rows / rcount), 1)
    if cstride == 1 and ccount < cols:
        cstride = max(int(cols / ccount), 1)
    
    # Extract polygons
    polys = []
    for rs in range(0, rows-1, rstride):
        for cs in range(0, cols-1, cstride):
            ps = []
            for r, c in [(rs, cs), (rs+rstride, cs),
                          (rs+rstride, cs+cstride), (rs, cs+cstride)]:
                if r < rows and c < cols:
                    ps.append([X[r, c], Y[r, c], Z[r, c]])
            polys.append(np.array(ps))
    
    # Handle color mapping
    cmap = kwargs.pop('cmap', None)
    norm = kwargs.pop('norm', None)
    
    if cmap is not None:
        # Calculate color for each polygon
        avg_z = np.array([ps[:, 2].mean() for ps in polys])
        
        # Create color mapping
        if norm is None:
            norm = plt.Normalize(Z.min(), Z.max())
        
        colors = cmap(norm(avg_z))
        kwargs['facecolors'] = colors
    
    # Apply lighting effects
    shade = kwargs.pop('shade', True)
    lightsource = kwargs.pop('lightsource', None)
    
    # Create high-performance collection
    collection = FastPoly3DCollection(polys, *args, **kwargs)
    
    # If lighting effects are needed and using Rust implementation
    if USING_RUST and shade and lightsource is not None:
        light_dir = getattr(lightsource, 'direction', [-1, -1, 0.5])
        collection._rs_collection.shade_colors(*light_dir)
    
    ax.add_collection(collection)
    
    # Auto-adjust view
    ax.auto_scale_xyz(X, Y, Z, True)
    
    return collection


# Export main components
__all__ = ['FastPoly3DCollection', 'fast_plot_surface']