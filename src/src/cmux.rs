//! The CMux gate, spelled out over `glwe_external_product`.
//!
//! Poulpy ships one: `poulpy_bin_fhe::bdd_arithmetic::Cmux::cmux_assign`. It cannot be
//! used here, because it reaches for `glwe_external_product_dft` -- the *internal* entry
//! point of the external product -- and skips the step the *default* one performs first:
//!
//! ```ignore
//! // poulpy-core 0.8.1, default/external_product/glwe.rs
//! if a_base2k != ggsw_base2k {
//!     let (mut a_conv, _) = scratch.take_glwe_scratch(&GLWELayout { base2k: ggsw.base2k(), .. });
//!     module.glwe_normalize_default(&mut a_conv, a, ..);   // <-- re-limb to the key's base2k
//!     module.glwe_external_product_dft(&mut res_dft, &a_conv, ggsw, ..);
//! } else {
//!     module.glwe_external_product_dft(&mut res_dft, a, ggsw, ..);
//! }
//! ```
//!
//! A GLWE is a stack of `base2k`-wide limbs and the VMP reads those limbs *as* the gadget
//! digits of the key. So the operand has to stand at the key's `base2k` before it goes in,
//! and when it does not, `a_conv` is where it is re-limbed. `cmux_assign` only ever takes
//! the second branch.
//!
//! At every parameter set `paramux` emits, the two disagree: `MuxLayouts` builds the GLWE
//! at `base2k = 19` and the GGSW at `base2k = bg_bits = log2(Bg)`, which is 7 to 10. The
//! VMP then reads 19-bit limbs as 10-bit digits, the gadget recomposition is off by
//! `2^9` per digit, and one gate is enough to bury the message: `examples/bare_cmux.rs`
//! returns `N/p` correct coefficients -- indistinguishable from random -- and
//! `examples/cmux_relimb_probe.rs` runs the two spellings side by side:
//!
//! ```text
//! N= 1024 base2k=19 k_rlwe=30 | GGSW base2k=2^10 dnum=2 | fresh 1024/1024 log2(var)=-49.23
//!       cmux_assign              76/1024 log2(var)=-11.57   <- uniform on the p-ary grid
//!       external_product       1024/1024 log2(var)=-20.66   <- E_ext, as the paper bounds it
//! ```
//!
//! Hence this module. `res = (res - a) * s + a` over the three public operations, the
//! middle one being the external product that re-limbs. It allocates nothing: the
//! subtraction and the addition are in place and the product is `_assign`, so the only
//! scratch is the external product's own -- strictly less than `cmux_tmp_bytes` asks for.

use poulpy_core::{
    GLWEAdd, GLWEExternalProduct, GLWESub,
    layouts::{GGSWInfos, GGSWPreparedToBackendRef, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef},
};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Scratch bytes one [`cmux_assign`] needs.
pub fn cmux_assign_tmp_bytes<BE, R, G>(module: &Module<BE>, res_infos: &R, ggsw_infos: &G) -> usize
where
    BE: Backend,
    Module<BE>: GLWEExternalProduct<BE>,
    R: GLWEInfos,
    G: GGSWInfos,
{
    module.glwe_external_product_tmp_bytes(res_infos, res_infos, ggsw_infos)
}

/// `res <- (res - a) * s + a`: `res` when the bit `s` encrypts is 1, `a` when it is 0.
///
/// Same contract as poulpy's `Cmux::cmux_assign`, and the same operand order, so the two
/// are interchangeable at a call site. See the module header for why this one exists.
pub fn cmux_assign<BE, R, A, S>(
    module: &Module<BE>,
    res: &mut R,
    a: &A,
    s: &S,
    scratch: &mut ScratchArena<BE>,
) where
    BE: Backend,
    Module<BE>: GLWESub<BE> + GLWEAdd<BE> + GLWEExternalProduct<BE>,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    A: GLWEToBackendRef<BE>,
    S: GGSWPreparedToBackendRef<BE> + GGSWInfos,
{
    module.glwe_sub_assign(res, a);
    module.glwe_external_product_assign(res, s, scratch);
    module.glwe_add_assign(res, a);
}
