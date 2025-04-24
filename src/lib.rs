//! Matplotlib 3D rendering engine implemented in Rust
//! 
//! This library provides a high-performance Rust implementation of Matplotlib's 3D functionality,
//! focused on improving 3D visualization performance for PDE solution results.

pub mod proj3d;
pub mod poly3d;
pub mod py_bridge;

// Re-export key types
pub use proj3d::{ProjectionMatrix, Projection};
pub use poly3d::{Poly3DCollection, ZSortMethod};

/// Library version number
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
