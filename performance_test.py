"""
Performance test: Compare standard Matplotlib and mpl3d-turbo in rendering complex models
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import psutil
import gc

# Add rust/python directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

# Try to import Rust version of visualization tools
try:
    from mpl3d_turbo import fast_plot_surface
    RUST_AVAILABLE = True
    print("Successfully imported Rust-accelerated version")
except ImportError:
    print("Warning: Rust library not compiled or cannot be imported. Using pure Python version.")
    print("Hint: Please run 'cargo build --release' and 'python -m pip install -e python/' in rust directory")
    RUST_AVAILABLE = False


def get_memory_usage():
    """Get current process memory usage (MB)"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB


def generate_complex_model(size=500):
    """Generate complex model data, return X, Y, Z grid data"""
    print(f"Generating {size}x{size} complex model data...")
    
    # Create more complex data
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # Overlay multiple complex functions, simulate complex surface
    R = np.sqrt(X**2 + Y**2)
    Z = 3 * (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
    Z += 0.5 * np.cos(R*5) * np.exp(-R/2)
    Z += 0.1 * np.sin(X*10) * np.cos(Y*10)
    Z += np.exp(-(X-1)**2 - (Y-1)**2) - np.exp(-(X+1)**2 - (Y+1)**2)
    
    return X, Y, Z


def test_matplotlib_standard(X, Y, Z, stride=10):
    """Use standard Matplotlib to draw 3D surface, measure time and memory usage"""
    print("\nTesting standard Matplotlib 3D surface drawing...")
    
    # Collect garbage and measure initial memory
    gc.collect()
    start_memory = get_memory_usage()
    
    # Start timing
    start_time = time.time()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use standard plot_surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                          rstride=stride, cstride=stride,
                          linewidth=0.1)
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # End timing
    end_time = time.time()
    
    # Measure final memory
    end_memory = get_memory_usage()
    
    # Close figure, do not display
    plt.close(fig)
    
    # Calculate execution time and memory usage
    execution_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"Standard Matplotlib execution time: {execution_time:.4f} seconds")
    print(f"Standard Matplotlib memory usage: {memory_used:.2f} MB")
    
    return execution_time, memory_used


def test_rust_accelerated(X, Y, Z, stride=10):
    """Use Rust-accelerated version to draw 3D surface, measure time and memory usage"""
    if not RUST_AVAILABLE:
        print("Rust-accelerated version not available, skipping test")
        return None, None
    
    print("\nTesting Rust-accelerated version 3D surface drawing...")
    
    # Collect garbage and measure initial memory
    gc.collect()
    start_memory = get_memory_usage()
    
    # Start timing
    start_time = time.time()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use Rust-accelerated version
    surf = fast_plot_surface(ax, X, Y, Z, cmap='viridis',
                            rstride=stride, cstride=stride,
                            linewidth=0.1)
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # End timing
    end_time = time.time()
    
    # Measure final memory
    end_memory = get_memory_usage()
    
    # Close figure, do not display
    plt.close(fig)
    
    # Calculate execution time and memory usage
    execution_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"Rust-accelerated version execution time: {execution_time:.4f} seconds")
    print(f"Rust-accelerated version memory usage: {memory_used:.2f} MB")
    
    return execution_time, memory_used


def run_comparison_tests(sizes=[100, 200, 500, 1000], stride=10):
    """Run comparison tests with different data sizes"""
    results = []
    
    for size in sizes:
        print(f"\n\n======= Testing data of size {size}x{size} =======")
        
        # Generate data
        X, Y, Z = generate_complex_model(size)
        
        # Test standard version
        std_time, std_memory = test_matplotlib_standard(X, Y, Z, stride)
        
        # Test Rust-accelerated version
        rust_time, rust_memory = test_rust_accelerated(X, Y, Z, stride)
        
        # Calculate speedup and memory ratio
        if RUST_AVAILABLE and std_time and rust_time:
            speedup = std_time / rust_time
            memory_ratio = std_memory / rust_memory if rust_memory > 0 else float('inf')
            print(f"\nSpeedup in {size}x{size} size: {speedup:.2f}x")
            print(f"Memory usage ratio in {size}x{size} size: {memory_ratio:.2f}x")
        
        # Save results
        results.append({
            'size': size,
            'std_time': std_time,
            'std_memory': std_memory,
            'rust_time': rust_time,
            'rust_memory': rust_memory
        })
    
    return results


def show_one_example(size=300, stride=10):
    """Generate and display a real example (suitable for visualization)"""
    print("\nGenerating and displaying a complex 3D surface example...")
    
    # Generate data
    X, Y, Z = generate_complex_model(size)
    
    # Create subplots
    if RUST_AVAILABLE:
        fig = plt.figure(figsize=(16, 8))
        
        # Standard Matplotlib
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis',
                                rstride=stride, cstride=stride,
                                linewidth=0.1)
        ax1.set_title("Standard Matplotlib")
        fig.colorbar(surf1, ax=ax1, shrink=0.5)
        
        # Rust-accelerated version
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = fast_plot_surface(ax2, X, Y, Z, cmap='viridis',
                                 rstride=stride, cstride=stride,
                                 linewidth=0.1)
        ax2.set_title("Rust-accelerated version")
        fig.colorbar(surf2, ax=ax2, shrink=0.5)
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                               rstride=stride, cstride=stride,
                               linewidth=0.1)
        ax.set_title("Standard Matplotlib (Rust version unavailable)")
        fig.colorbar(surf, shrink=0.5)
    
    plt.suptitle(f"Complex 3D surface rendering ({size}x{size} grid)")
    plt.tight_layout()
    plt.show()


def main():
    """Main function"""
    # Show a visualization example
    show_one_example(size=300, stride=5)
    
    # Run performance comparison test
    # For more complex tests, use larger stride to speed up rendering
    sizes = [100, 200, 500, 1000]
    stride = 10
    
    # Ask user if they want to run the full test suite
    response = input("\nRun full performance test suite? (y/n): ")
    if response.lower() == 'y':
        print("\nStarting performance comparison test...")
        results = run_comparison_tests(sizes, stride)
        
        # Display results summary
        print("\n\n========== Performance Comparison Test Results ==========")
        print(f"{'Size':^10}|{'Standard Time (s)':^15}|{'Rust Time (s)':^15}|{'Speedup':^10}|{'Standard Memory (MB)':^15}|{'Rust Memory (MB)':^15}|{'Memory Ratio':^10}")
        print("-" * 85)
        
        for r in results:
            speedup = r['std_time'] / r['rust_time'] if RUST_AVAILABLE and r['rust_time'] else float('nan')
            memory_ratio = r['std_memory'] / r['rust_memory'] if RUST_AVAILABLE and r['rust_memory'] and r['rust_memory'] > 0 else float('nan')
            print(f"{r['size']:^10}|{r['std_time']:^15.4f}|{r['rust_time'] if r['rust_time'] else 'N/A':^15}|{speedup:^10.2f}|{r['std_memory']:^15.2f}|{r['rust_memory'] if r['rust_memory'] else 'N/A':^15}|{memory_ratio:^10.2f}")
    else:
        print("\nSkipping performance test suite.")


if __name__ == "__main__":
    main()
