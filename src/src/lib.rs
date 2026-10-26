//! Poulpy implementations of OHVPack and Vertical Packing, plus the harness that
//! benchmarks and tests them.
//!
//! - [`ohvpack`], [`vertical_packing`] -- the two evaluations.
//! - [`selector`], [`lut`], [`prepare_data`] -- the inputs they consume: encrypted
//!   selectors, the plaintext table in either residency, and the hoisted DFTs of the
//!   GLWE x plaintext product.
//! - [`context`], [`params`], [`bench_timing`] -- layouts, keys and the benchmark driver
//!   every benchmark and test shares.
//! - [`time_profile`], [`mem_profile`], [`noise_profile`] -- the three feature-gated
//!   instrumentation modes.

// Poulpy threads the module, the scratch arena, the encryption layouts and the keys
// through explicitly rather than bundling them in a context object, so an evaluation
// signature runs to eight or nine parameters and a selector bundle to a nested generic
// tuple. Both are the shape of the API this crate is written against, not accidents.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod bench_timing;
pub mod cmux;
pub mod context;
pub mod ohvpack;
pub mod params;
pub mod lut;
pub mod selector;
pub mod prepare_data;
pub mod mem_profile;
pub mod noise_profile;
pub mod time_profile;
pub mod trivial;
pub mod vertical_packing;
