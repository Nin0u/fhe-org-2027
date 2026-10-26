//! Preparation (DFT) of the operands of a GLWE x plaintext multiplication.
//!
//! Poulpy has these steps, but `glwe_mul_plain` does them internally: it prepares *both*
//! the GLWE and the plaintext on every call. The tree evaluators multiply one GLWE
//! selector by many LUT polynomials, so that re-prepares the same selector once per
//! polynomial.
//!
//! Splitting the two lets the caller hoist the selector: [`prepare_glwe`] runs once
//! outside the loop, [`prepare_plain_into`] runs per LUT polynomial into a reused buffer,
//! and [`mul_prepared`] consumes both. The LUT is deliberately *not* prepared up front --
//! it is the term that scales with the table size, so keeping it streamed is what holds
//! the memory footprint flat; the selector is the O(1) operand and carries the saving.

use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxIdftApplyTmpA,
    },
    layouts::{
        Backend, CnvPVecL, CnvPVecLToBackendMut, CnvPVecLToBackendRef, CnvPVecR,
        CnvPVecRToBackendMut, CnvPVecRToBackendRef, Module, ScratchArena,
        VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxDftToBackendMut, VecZnxToBackendRef,
    },
};
use poulpy_core::{
    layouts::{GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, LWEInfos},
    msb_mask_bottom_limb,
};

// crate-private in poulpy, so reimplement it (4 lines):
fn cnv_offset_to_limb_offset(cnv_offset: usize, base2k: usize) -> (usize, i64) {
    if cnv_offset < base2k {
        (0, -((base2k - (cnv_offset % base2k)) as i64))
    } else {
        ((cnv_offset / base2k).saturating_sub(1), (cnv_offset % base2k) as i64)
    }
}

/// Prepare the GLWE once -> reusable left operand.
///
/// This is the one to hoist: the selector is multiplied by every LUT polynomial, so
/// preparing it here instead of inside `glwe_mul_plain` drops one DFT per polynomial.
pub fn prepare_glwe<BE: Backend, A>(m: &Module<BE>, a: &A, s: &mut ScratchArena<'_, BE>)
    -> CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>
where A: GLWEToBackendRef<BE> + GLWEInfos,
      Module<BE>: CnvPVecAlloc<BE> + Convolution<BE> + ModuleN,
{
    let cols = a.rank().as_usize() + 1;
    let mask = msb_mask_bottom_limb(a.base2k().as_usize(), a.k().as_usize()); // effective k
    let mut prep = m.cnv_pvec_left_alloc(cols, a.size());
    let a_be = a.to_backend_ref();
    let mut prep_be = prep.to_backend_mut();
    m.cnv_prepare_left_lazy(&mut prep_be, a_be.data(), mask, s);
    drop(prep_be);
    prep
}

/// Allocate the right-operand buffer that [`prepare_plain_into`] fills.
///
/// Hoist this out of the streaming loop and reuse the buffer, so preparing the LUT on the
/// fly costs no per-polynomial allocation.
pub fn alloc_plain_prep<BE: Backend, B>(m: &Module<BE>, b: &B) -> CnvPVecR<BE::OwnedBuf, BE::DftWord, BE>
where B: GLWEInfos,
      Module<BE>: CnvPVecAlloc<BE> + ModuleN,
{
    m.cnv_pvec_right_alloc(1, b.size())
}

/// Prepare a plaintext polynomial into an existing buffer -> right operand.
///
/// Called once per LUT polynomial, on the fly: the LUT is what scales with the table
/// size, so it stays streamed rather than prepared up front.
pub fn prepare_plain_into<BE: Backend, B>(
    m: &Module<BE>,
    prep: &mut CnvPVecR<BE::OwnedBuf, BE::DftWord, BE>,
    b: &B,
    s: &mut ScratchArena<'_, BE>,
) where
    B: GLWEToBackendRef<BE> + GLWEInfos,
    Module<BE>: Convolution<BE> + ModuleN,
{
    // IMPORTANT: plaintext is masked at max_k (its full stored width), NOT effective k.
    // Masking at the effective k would zero the low bits of the last limb and lose
    // precision in the convolution.
    let mask = msb_mask_bottom_limb(b.base2k().as_usize(), b.max_k().as_usize());
    let b_be = b.to_backend_ref();
    let mut prep_be = prep.to_backend_mut();
    m.cnv_prepare_right_lazy(&mut prep_be, b_be.data(), mask, s);
}

/// Multiply prepared-by-prepared into `res`.
///
/// The body mirrors poulpy's `glwe_mul_plain_default` from the point where both operands
/// are already prepared, so `cnv_offset`, `a_size`/`b_size` and `ab_base2k` mean what they
/// do there: `a_size`/`b_size` are the limb counts of the two operands (the prepared
/// buffers do not carry them) and `ab_base2k` is their shared base2k.
pub fn mul_prepared<BE: Backend, R>(
    m: &Module<BE>, cnv_offset: usize, res: &mut R,
    a_prep: &CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>, a_size: usize,
    b_prep: &CnvPVecR<BE::OwnedBuf, BE::DftWord, BE>, b_size: usize,
    ab_base2k: usize, scratch: &mut ScratchArena<'_, BE>,
) where R: GLWEToBackendMut<BE> + GLWEInfos,
        Module<BE>: Convolution<BE> + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE>
                  + VecZnxBigNormalizeTmpBytes + VecZnxCopyBackend<BE> + ModuleN,
{
    let cols = res.rank().as_usize() + 1;
    let res_base2k = res.base2k().as_usize();
    let (off_hi, off_lo) = cnv_offset_to_limb_offset(cnv_offset, ab_base2k);
    let res_dft_size = a_size + b_size - off_hi;

    let (mut res_tmp, mut s) = scratch.borrow().take_vec_znx_scratch(m.n(), 1, res.size());
    for i in 0..cols {
        let (mut res_dft, mut s3) = s.borrow().take_vec_znx_dft_scratch(m, 1, res_dft_size);
        {
            let mut d = res_dft.to_backend_mut();
            m.cnv_apply_dft_lazy(off_hi, &mut d, 0,
                &a_prep.to_backend_ref(), i, &b_prep.to_backend_ref(), 0, &mut s3);
        }
        let (mut res_big, mut s4) = s3.take_vec_znx_big_scratch(m, 1, res_dft_size);
        {
            let mut bg = res_big.to_backend_mut();
            let mut d = res_dft.to_backend_mut();
            m.vec_znx_idft_apply_tmpa(&mut bg, 0, &mut d, 0);
        }
        let bg = res_big.to_backend_ref();
        m.vec_znx_big_normalize(&mut res_tmp, res_base2k, off_lo, 0, &bg, ab_base2k, 0, &mut s4.borrow());
        let mut r = res.to_backend_mut();
        m.vec_znx_copy_backend(r.data_mut(), i, &res_tmp.to_backend_ref(), 0);
    }
}