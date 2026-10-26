//! Trivial ciphertexts `(0, P)`, and the external product that consumes one.
//!
//! A trivial ciphertext carries its plaintext in the clear: the mask is zero, so `(0, P)`
//! decrypts to `P` under every key and carries no noise. §2.3.7 hands VPack its table this
//! way, and that is what makes VPack's first tree level cheaper than the ones above it:
//!
//!  - the **A-half** of the gadget decomposition is zero, so its `l` forward transforms
//!    are transforms of the zero polynomial;
//!  - the **B-half** is determined by the table alone -- a first-level gate combines two
//!    table polynomials and nothing the evaluation computed -- so its `l` forward
//!    transforms are precomputable, and §2.2's convention charges no transform to an input
//!    supplied in transformed form.
//!
//! Both halves together are the `2l` forward transforms per gate that §2.3.7 subtracts
//! from VPack's count, taking it from `(2l + 2)nu` to `(l + 2)nu` transforms.
//!
//! [`glwe_external_product_trivial`] is the external product spelled over a
//! [`TrivialOperand`] -- the operand's body, re-limbed to the GGSW's `base2k` and
//! transformed -- rather than over a GLWE. It performs no forward transform of its own, so
//! where the operand is prepared decides which of the two credits a caller takes:
//! [`TrivialOperand::fill`] inside the timed region takes the A-half only, a slice of
//! operands filled before it takes both (see [`crate::vertical_packing::FirstLevel`]).
//!
//! Everything here assumes `dsize = 1`, which is what [`crate::context::MuxLayouts`]
//! builds and what every parameter set `paramux` emits uses; the strided digit loop of
//! poulpy's own external product has no counterpart below and is asserted against.

use poulpy_core::{
    GLWENormalize, ScratchArenaTakeCore,
    api::GLWEBytesOf,
    default::external_product::glwe::glwe_external_product_output_size,
    layouts::{
        BackendGLWE, Dsize, GGSWInfos, GGSWPrepared, GLWEInfos, GLWELayout, GLWEToBackendMut,
        GLWEToBackendRef, LWEInfos, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxDftAlloc, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, Module, ScratchArena, VecZnxDftOwned, VecZnxDftToBackendMut,
        VecZnxBigToBackendRef, VecZnxDftToBackendRef, VmpPMatToBackendRef,
    },
};

/// The trivial GLWE encryption `(0, P)` of an already-encoded plaintext.
///
/// `pt` is copied into the body and every mask column is left at zero, so the result
/// decrypts to `pt` under any key, with no noise and no PRNG draw. The plaintext may be
/// narrower than `infos`: the copy zero-fills the limbs past its end.
pub fn trivial_glwe<BE, A, P>(module: &Module<BE>, infos: &A, pt: &P) -> BackendGLWE<BE>
where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = Vec<u8>, ZnxWord = i64> + VecZnxCopyBackend<BE>,
    A: GLWEInfos,
    P: GLWEToBackendRef<BE>,
{
    // `glwe_alloc_from_infos` zero-initialises, which is already the mask half of a
    // trivial ciphertext.
    let mut ct: BackendGLWE<BE> = module.glwe_alloc_from_infos(infos);
    {
        let pt_ref = pt.to_backend_ref();
        let mut ct_mut = GLWEToBackendMut::<BE>::to_backend_mut(&mut ct);
        module.vec_znx_copy_backend(ct_mut.data_mut(), 0, pt_ref.data(), 0);
    }
    ct
}

/// A trivial ciphertext's body, re-limbed to a GGSW's `base2k` and transformed: the left
/// operand of [`glwe_external_product_trivial`].
///
/// The mask columns are zeroed once at [`TrivialOperand::alloc`] and never written again --
/// they are zero for every trivial ciphertext, which is exactly the A-half credit of
/// §2.3.7 -- so [`TrivialOperand::fill`] transforms the body alone.
pub struct TrivialOperand<BE: Backend> {
    dft: VecZnxDftOwned<BE>,
    /// Layout the operand is re-limbed to before the transform: the GLWE's `n`, `k` and
    /// rank, at the GGSW's `base2k`. This is the conversion poulpy's own external product
    /// performs internally whenever the two disagree (see [`crate::cmux`]); here it happens
    /// once, when the operand is prepared.
    conv: GLWELayout,
}

impl<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>> TrivialOperand<BE> {
    /// Buffer for a trivial `a_infos`-shaped operand of `ggsw_infos`.
    pub fn alloc<A, G>(module: &Module<BE>, a_infos: &A, ggsw_infos: &G) -> Self
    where
        A: GLWEInfos,
        G: GGSWInfos,
        Module<BE>: VecZnxDftAlloc<BE> + VecZnxDftZero<BE> + ModuleN,
    {
        assert_eq!(
            ggsw_infos.dsize(),
            Dsize(1),
            "the trivial-operand external product has no strided digit loop, so it needs dsize = 1"
        );
        let conv: GLWELayout = GLWELayout {
            n: a_infos.n(),
            base2k: ggsw_infos.base2k(),
            k: a_infos.k(),
            rank: a_infos.rank(),
        };
        let cols: usize = (conv.rank + 1).into();
        let mut dft: VecZnxDftOwned<BE> = module.vec_znx_dft_alloc(cols, conv.size());
        {
            let mut dft_mut = dft.to_backend_mut();
            for col in 0..cols {
                module.vec_znx_dft_zero(&mut dft_mut, col);
            }
        }
        Self { dft, conv }
    }

    /// Re-limb and transform the body of the trivial ciphertext `a`.
    ///
    /// This is the `l` forward transforms §2.3.7 credits as precomputable. Calling it
    /// inside a timed region pays them; calling it before one, over the table, does not.
    /// The mask of `a` is *assumed* zero and never read -- passing a genuine ciphertext
    /// here silently drops its mask.
    pub fn fill<A>(&mut self, module: &Module<BE>, a: &A, scratch: &mut ScratchArena<BE>)
    where
        A: GLWEToBackendRef<BE> + GLWEInfos,
        Module<BE>: GLWENormalize<BE> + VecZnxDftApply<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchArenaTakeBasic<'a, BE>,
    {
        debug_assert_eq!(a.k(), self.conv.k, "trivial operand prepared at a different torus precision");
        debug_assert_eq!(a.rank(), self.conv.rank, "trivial operand prepared at a different rank");
        scratch.scope(|s| {
            let (mut a_conv, mut s1) = s.take_glwe_scratch(&self.conv);
            module.glwe_normalize(&mut a_conv, a, &mut s1.borrow());
            let a_ref = GLWEToBackendRef::to_backend_ref(&a_conv);
            let mut dft_mut = self.dft.to_backend_mut();
            module.vec_znx_dft_apply(1, 0, &mut dft_mut, 0, a_ref.data(), 0);
        });
    }
}

/// Scratch bytes one [`TrivialOperand::fill`] needs: the re-limbing buffer, plus what the
/// normalisation that fills it takes.
pub fn trivial_operand_fill_tmp_bytes<BE, A, G>(module: &Module<BE>, a_infos: &A, ggsw_infos: &G) -> usize
where
    BE: Backend,
    Module<BE>: GLWEBytesOf<BE> + GLWENormalize<BE>,
    A: GLWEInfos,
    G: GGSWInfos,
{
    let conv: GLWELayout = GLWELayout {
        n: a_infos.n(),
        base2k: ggsw_infos.base2k(),
        k: a_infos.k(),
        rank: a_infos.rank(),
    };
    module.glwe_bytes_of_from_infos(&conv) + module.glwe_normalize_tmp_bytes()
}

/// Scratch bytes one [`glwe_external_product_trivial`] needs.
///
/// Sized exactly as poulpy's `glwe_external_product_tmp_bytes` minus the re-limbing
/// buffer, which [`TrivialOperand::fill`] has already spent.
pub fn glwe_external_product_trivial_tmp_bytes<BE, R, G>(
    module: &Module<BE>,
    res_infos: &R,
    ggsw_infos: &G,
) -> usize
where
    BE: Backend,
    Module<BE>: VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDftTmpBytes
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalizeTmpBytes,
    R: GLWEInfos,
    G: GGSWInfos,
{
    let align: usize = BE::SCRATCH_ALIGN;
    let cols: usize = (res_infos.rank() + 1).into();
    let in_size: usize = res_infos.k().div_ceil(ggsw_infos.base2k()) as usize;
    let output_size: usize = glwe_external_product_output_size::<BE, _, _, _>(res_infos, res_infos, ggsw_infos);
    let lvl_0: usize = module.bytes_of_vec_znx_dft(cols, output_size);
    let lvl_1: usize = module.bytes_of_vec_znx_big(cols, output_size).next_multiple_of(align)
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let lvl_2: usize =
        module.vmp_apply_dft_to_dft_tmp_bytes(output_size, in_size, in_size, cols, cols, ggsw_infos.size());
    lvl_0.next_multiple_of(align) + lvl_1.max(lvl_2)
}

/// `res <- op * ggsw`, `op` the prepared body of a trivial ciphertext.
///
/// The body of poulpy's `glwe_external_product` from the point where the operand is
/// transformed: one VMP, `rank + 1` inverse transforms, `rank + 1` normalisations. No
/// forward transform happens here -- that is the whole point, and it is why the operand
/// arrives as a [`TrivialOperand`] rather than as a GLWE.
///
/// The limb products are *not* credited: the VMP multiplies the operand's zero mask
/// columns through like any other, which is what §2.3.7 charges (`4l nu M`, first level
/// included).
pub fn glwe_external_product_trivial<BE, R>(
    module: &Module<BE>,
    res: &mut R,
    op: &TrivialOperand<BE>,
    ggsw: &GGSWPrepared<BE::OwnedBuf, BE>,
    scratch: &mut ScratchArena<BE>,
) where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>,
    Module<BE>: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    R: GLWEToBackendMut<BE> + GLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeBasic<'a, BE>,
{
    debug_assert_eq!(ggsw.rank(), res.rank());
    debug_assert_eq!(ggsw.base2k(), op.conv.base2k);
    let cols: usize = (res.rank() + 1).into();
    let res_base2k: usize = res.base2k().into();
    let ggsw_base2k: usize = ggsw.base2k().into();
    let output_size: usize = glwe_external_product_output_size::<BE, _, _, _>(res, res, ggsw);

    let (mut res_dft, scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(module, cols, output_size);
    {
        let mut s = scratch_1;
        module.vmp_apply_dft_to_dft(
            &mut res_dft,
            &op.dft.to_backend_ref(),
            &ggsw.data().to_backend_ref(),
            0,
            &mut s.borrow(),
        );

        let (mut res_big, mut s2) = s.borrow().take_vec_znx_big_scratch(module, cols, res_dft.size());
        {
            let res_dft_ref = res_dft.to_backend_ref();
            for col in 0..cols {
                module.vec_znx_idft_apply(&mut res_big, col, &res_dft_ref, col, &mut s2.borrow());
            }
        }
        let res_big_ref = res_big.to_backend_ref();
        let mut res_ref = res.to_backend_mut();
        for col in 0..cols {
            module.vec_znx_big_normalize(
                res_ref.data_mut(),
                res_base2k,
                0,
                col,
                &res_big_ref,
                ggsw_base2k,
                col,
                &mut s2.borrow(),
            );
        }
    }
}
