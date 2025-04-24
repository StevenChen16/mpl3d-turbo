from setuptools import setup, find_packages
from setuptools_rust import Binding, RustExtension

setup(
    name="mpl3d-rs",
    version="0.1.0",
    packages=find_packages(),
    rust_extensions=[RustExtension("mpl3d_rs.mpl3d_rs", binding=Binding.PyO3)],
    # If your rust code is not under the python directory, specify the path
    package_dir={"": "python"},
    # Dependencies needed for rust compilation
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
    ],
    # Ensure Rust library is compiled
    zip_safe=False,
    # PyPI metadata
    author="Elton PDE Team",
    author_email="example@example.com",
    description="High-performance Matplotlib 3D rendering library implemented in Rust",
    keywords="matplotlib, 3d, rust, pde, visualization",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.7",
)
