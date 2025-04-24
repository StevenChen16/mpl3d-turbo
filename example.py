"""
展示如何在PDE项目中使用Rust加速的Matplotlib 3D绘图
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add rust/python directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

# Import original PDE solver
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pde import solve_pde, solve_pde_and_visualize

# Try to import Rust version of visualization tools
try:
    from mpl3d_turbo import fast_plot_surface
    RUST_AVAILABLE = True
except ImportError:
    print("Warning: Rust library not compiled or cannot be imported. Using pure Python version.")
    print("Hint: Please run 'maturin develop --release' in the project directory")
    RUST_AVAILABLE = False

def visualize_pde_with_rust(x_display, t, C_display, time_steps=5):
    """Use Rust-accelerated 3D visualization of PDE solutions"""
    # 2D visualization part remains unchanged
    plt.figure(figsize=(10,6))
    colors = plt.cm.viridis(np.linspace(0, 1, time_steps))
    t_indices = np.linspace(0, len(t)-1, time_steps, dtype=int)
    
    for idx, n in enumerate(t_indices):
        plt.plot(x_display, C_display[:,n], 
                 color=colors[idx],
                 linewidth=2,
                 label=f't = {t[n]:.2f}')
        
    plt.title(f'Solution Profiles (x ∈ [{x_display[0]}, {x_display[-1]}])')
    plt.xlabel('x'), plt.ylabel('c(x,t)')
    plt.legend(), plt.grid(True)
    plt.xlim(x_display[0], x_display[-1])
    plt.show()
    
    # Create 3D plot
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate grid
    X, T = np.meshgrid(x_display, t)
    C_3d = C_display.T
    
    # Use Rust-accelerated surface drawing
    if RUST_AVAILABLE:
        print("Using Rust-accelerated 3D drawing...")
        surf = fast_plot_surface(
            ax, X, T, C_3d, 
            cmap='viridis',
            rstride=max(1, len(t)//50),  # Automatically adjust rendering density
            cstride=max(1, len(x_display)//50),
            linewidth=0.2
        )
    else:
        # Fallback to Matplotlib implementation
        print("Using standard Matplotlib 3D drawing...")
        surf = ax.plot_surface(
            X, T, C_3d, 
            cmap='viridis',
            rstride=max(1, len(t)//50),  # Automatically adjust rendering density
            cstride=max(1, len(x_display)//50),
            linewidth=0.2
        )
    
    ax.set_xlabel('x'), ax.set_ylabel('t'), ax.set_zlabel('c(x,t)')
    ax.view_init(30, -60)
    fig.colorbar(surf, shrink=0.5)
    plt.title('3D Spatiotemporal Evolution' + 
              (' (Rust-accelerated)' if RUST_AVAILABLE else ''))
    plt.show()

def compare_performance():
    """Compare standard Matplotlib and Rust-accelerated versions of performance"""
    # Create same data for comparison
    print("Generating test data...")
    x = np.linspace(-5, 5, 200)
    t = np.linspace(0, 3, 200)
    X, T = np.meshgrid(x, t)
    Z = np.sin(X + 0.1*T) * np.exp(-0.1*T)
    
    print("\nPerformance comparison:")
    
    # Test standard version
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    print("Testing standard Matplotlib version...")
    start_time = time.time()
    surf = ax.plot_surface(X, T, Z, cmap='viridis',
                          rstride=5, cstride=5)
    plt.close(fig)
    mpl_time = time.time() - start_time
    print(f"Standard Matplotlib time: {mpl_time:.4f} seconds")
    
    # Test Rust-accelerated version
    if RUST_AVAILABLE:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        print("Testing Rust-accelerated version...")
        start_time = time.time()
        surf = fast_plot_surface(ax, X, T, Z, cmap='viridis',
                                rstride=5, cstride=5)
        plt.close(fig)
        rust_time = time.time() - start_time
        print(f"Rust-accelerated version time: {rust_time:.4f} seconds")
        
        print(f"\nSpeedup: {mpl_time/rust_time:.2f}x")
    else:
        print("Rust library not available, cannot compare performance.")

def main():
    """Main function: Run PDE solving and performance comparison"""
    # Performance comparison
    compare_performance()
    
    print("\n\nSolving PDE equation and visualizing...")
    # Solve PDE
    x, t, C = solve_pde(
        x_left=-3, 
        x_right=8,
        bc_left=0.0,   # Left boundary fixed at 0
        bc_right=0.0,   # Right boundary fixed at 0
        # Use lower resolution to speed up demo
        dx=1e-2,
        dt=1e-2,
    )
    
    # Visualize with Rust-accelerated version
    visualize_pde_with_rust(x, t, C, time_steps=7)

if __name__ == "__main__":
    main()