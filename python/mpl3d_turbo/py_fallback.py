"""
mpl3d-rs: Python fallback implementation

This provides a pure Python implementation that mimics the Rust library's interface.
Used when the Rust library cannot be compiled.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource

class PyFastPoly3DCollection(Poly3DCollection):
    """Python implementation of FastPoly3DCollection
    
    This class provides a compatible interface to the Rust implementation
    but uses pure Python/Matplotlib code.
    """
    
    def __init__(self, verts, *args, **kwargs):
        super().__init__(verts, *args, **kwargs)
        self._verts3d = verts
        self._zsort = 'average'
        self._sort_zpos = None
    
    def set_sort_zpos(self, val):
        """Set the position to use for z-sorting"""
        self._sort_zpos = val
        return super().set_sort_zpos(val)
    
    def set_zsort(self, zsort):
        """Set the z-sorting method"""
        self._zsort = zsort
        return super().set_zsort(zsort)
    
    def do_3d_projection(self):
        """Perform 3D projection"""
        return super().do_3d_projection()


def py_fast_plot_surface(ax, X, Y, Z, *args, **kwargs):
    """Python implementation of fast_plot_surface
    
    This provides the same interface as the Rust version but uses
    standard Matplotlib code under the hood.
    """
    rows, cols = Z.shape
    
    rstride = kwargs.pop('rstride', 1)
    cstride = kwargs.pop('cstride', 1)
    rcount = kwargs.pop('rcount', 50)
    ccount = kwargs.pop('ccount', 50)
    
    # Calculate stride based on count if needed
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
    cmap_name = kwargs.pop('cmap', 'viridis')
    norm = kwargs.pop('norm', None)
    
    # Handle the case when cmap is a string (colormap name)
    if isinstance(cmap_name, str):
        cmap = plt.cm.get_cmap(cmap_name)
    else:
        cmap = cmap_name
    
    if cmap is not None:
        # Calculate the color for each polygon
        avg_z = np.array([ps[:, 2].mean() for ps in polys])
        
        # Create color mapping
        if norm is None:
            norm = plt.Normalize(Z.min(), Z.max())
        
        colors = cmap(norm(avg_z))
        kwargs['facecolors'] = colors
    
    # Apply shading if requested
    shade = kwargs.pop('shade', True)
    lightsource = kwargs.pop('lightsource', None)
    
    if shade and lightsource is None:
        lightsource = LightSource(270, 45)
    
    # Create collection
    collection = PyFastPoly3DCollection(polys, *args, **kwargs)
    ax.add_collection(collection)
    
    # Auto-scale the view
    ax.auto_scale_xyz(X, Y, Z, True)
    
    return collection