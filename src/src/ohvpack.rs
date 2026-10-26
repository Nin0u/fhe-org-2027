//! OHVPack: OHCollapse followed by a binary CMux tree (paper, §5).
//!
//! OHCollapse ([`ohcollapse`]) consumes the `G = L/(alpha*N)` one-hot TRLWE selectors and
//! reduces the `nu = L/N` plaintext polynomials to `alpha` TRLWE ciphertexts whose constant
//! coefficients are the table entries still addressable by `b`, using no Packing and no
//! external product. Selecting one of `alpha` ciphertexts is exactly what a CMux tree does,
//! so the second stage is Vertical Packing's tree over those accumulators.
//!
//! Against Vertical Packing, which reads every bit of `i / N` through a CMux tree over all
//! `nu` polynomials and finishes with a rotation:
//!
//! | | VPack | OHVPack |
//! |---|---|---|
//! | first stage | none | `nu` plaintext x TRLWE products |
//! | CMux tree | `log2(nu)` levels, `nu - 1` gates | `log2(alpha)` levels, `alpha - 1` gates |
//! | final rotation | one external product by `X^{-i0}` | none |
//! | selectors | `log2(nu)` TRGSW bits + one TRGSW monomial | `G` TRLWE + `log2(alpha)` TRGSW bits |
//!
//! OHVPack contains **no PackAuto at all**, so it needs no automorphism keys.
//!
//! There is also no final rotation. `X^{-i0}` was folded into the one-hot vector by the
//! first stage, so the target coefficient is already at the constant term of every
//! accumulator and the `log2(N)` low-order selectors of §2.3.7 -- or the single external
//! product that replaces them -- have nothing left to do.

use poulpy_core::{
    GLWEAdd, GLWEExternalProduct, GLWESub, LWESampleExtract,
    ScratchArenaTakeCore,
    layouts::{
        BackendGLWE, BackendLWE, GGSWInfos, GGSWPrepared, GGSWPreparedToBackendRef, GLWEInfos,
        GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{
        CnvPVecAlloc, Convolution, ModuleN, ScratchArenaTakeBasic, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxCopyBackend, VecZnxIdftApplyTmpA,
    },
    layouts::{Backend, CnvPVecL, HostDataMut, Module, ScratchArena},
};

use crate::cmux::cmux_assign;
use crate::lut::{Lut, LutFeed};
use crate::params::MuxParams;
use crate::prepare_data::mul_prepared;
use crate::{mem_span, noise_span, time_span};

/// First stage of OHVPack, the OHCollapse substitution (paper, §4.3).
///
/// Each degree-N polynomial packs N consecutive LUT entries as coefficients; with a
/// LUT of size N^d this yields `nu = N^(d-1)` polynomials, split into `G = ceil(nu/alpha)`
/// contiguous groups of `alpha` polys. Poly `m` sits in group `g = m / alpha` at position
/// `b = m % alpha` and is multiplied by that group's GLWE selector -- so the stage
/// consumes `G = L/(alpha*N)` selectors, one per group, and *not* `alpha` of them. Only
/// the selected group is non-zero (its selector is X^{-tc}), so summing position-wise
/// across groups leaves, for each position `b`, one rotated polynomial `outputs[b]` whose
/// constant term is a candidate LUT value. Returns those `min(alpha, nu)` accumulators.
///
/// `glwe_alloc` zero-initialises, so each accumulator starts as a (noiseless)
/// encryption of 0.
pub fn ohcollapse<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &MuxParams,
    scratch: &mut ScratchArena<BE>,
    lut: Lut<BE>,
    selectors_prep: &[CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>],
) -> Vec<BackendGLWE<BE>>
where
    Module<BE>: GLWEAdd<BE> + CnvPVecAlloc<BE> + Convolution<BE> + ModuleN + VecZnxIdftApplyTmpA<BE>
        + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes + VecZnxCopyBackend<BE>,
    for <'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchArenaTakeBasic<'a, BE>,
{
    mem_span!("stage 1: plaintext x one-hot selectors");
    let n: usize = params.n_rlwe;
    let alpha: usize = params.alpha.max(1);
    let n_polys: usize = lut.len().div_ceil(n);
    // One accumulator per position inside a group, capped at the `nu` positions a
    // polynomial actually reaches: at `alpha > nu` the group is padded and the tail
    // accumulators would stay at zero. See `MuxParams::ohcollapse_n_candidates`.
    let n_candidates: usize = alpha.min(n_polys);

    // The stage tiles n_polys polynomials into groups of `alpha`, one selector per group.
    // A short selector vector leaves the tail unreachable and every LUT entry in it
    // decodes to garbage -- silently -- which is what capping the vector at `alpha`
    // entries used to do for every alpha below sqrt(n_polys).
    let n_groups: usize = n_polys.div_ceil(alpha);
    assert!(
        selectors_prep.len() >= n_groups,
        "ohcollapse: {n_polys} polynomials in groups of {alpha} need {n_groups} \
         one-hot selectors, got {}",
        selectors_prep.len()
    );

    let mut outputs: Vec<BackendGLWE<BE>> = (0..n_candidates)
        .map(|_| module.glwe_alloc(params.base2k.into(), params.k_rlwe.into(), params.rank.into()))
        .collect();

    // The selectors are already in DFT; the LUT polynomials come from the feed -- prepared
    // per iteration into one reused buffer (streaming), or read from a table prepared up
    // front (full). See `crate::lut`.
    let mut feed: LutFeed<BE> = LutFeed::new(module, params, lut, n_polys, n);
    let pt_size: usize = feed.poly_size();
    let sel_size: usize = params.k_rlwe.div_ceil(params.base2k);

    for m in 0..n_polys {
        let g: usize = m / alpha; // group -> GLWE selector index (< n_groups, per the assert)
        let b: usize = m % alpha; // position within group -> output accumulator index

        let mut mul_res: BackendGLWE<BE> = module.glwe_alloc(
            params.base2k.into(),
            params.k_rlwe.into(),
            params.rank.into(),
        );
        {
            // Poly m holds lut[m*n .. m*n + n] (zero-padded past the table's end).
            let lut_prep = feed.poly(module, m, scratch);
            time_span!("mul_prepared");
            mul_prepared(
                module,
                params.log_p,
                &mut mul_res,
                &selectors_prep[g],
                sel_size,
                lut_prep,
                pt_size,
                params.base2k,
                scratch,
            );
        }
        {
            time_span!("glwe_add_assign");
            module.glwe_add_assign(&mut outputs[b], &mul_res);
        }
    }
    outputs
}

/// Number of CMux levels the second stage runs, i.e. `log2(alpha)`.
///
/// The tree is binary, so `alpha` must be a power of two -- which the collapsing factor of
/// §4.6 already imposes. No constraint relates `alpha` to `N`: the depth simply grows as
/// `log2(alpha)`.
pub fn ohvpack_tree_depth(alpha: usize) -> usize {
    assert!(alpha.is_power_of_two(), "OHVPack's CMux tree is binary, so alpha must be a power of two, got {alpha}");
    alpha.trailing_zeros() as usize
}

/// OHVPack evaluation: `T[i]` in the constant term of the returned LWE (paper, §5.1).
///
/// `selectors_prep` is the first stage's one-hot TRLWE vector, prepared, built by
/// [`crate::selector::ohvpack_make_selectors`]. `tree_selectors` supplies the second
/// stage's bits, one TRGSW per level: entry `j` encrypts bit `j` of `b = (i / N) mod alpha`.
///
/// The tree is folded LSB-first over adjacent pairs, so level `j` combines the two blocks
/// that differ in bit `j` of `b` and `cmux_assign(hi, lo, bit)` -- `hi = (hi - lo) * bit +
/// lo` -- takes `hi` when the bit is 1 and `lo` when it is 0. After `log2(alpha)` levels
/// one accumulator remains; a single SampleExtract closes the evaluation.
///
/// The tree runs over the `min(alpha, nu)` accumulators the first stage leaves, which is
/// `alpha` at every `alpha <= nu` -- every shipped parameter set included.
///
/// At `alpha = 1` the tree is empty and this is the pure one-hot evaluation of §4.8: the
/// first stage returns the single accumulator `acc_0 = [X^{-i0} P_g]` whose constant
/// coefficient is already `T[i]`.
pub fn ohvpack_eval<BE>(
    module: &Module<BE>,
    params: &MuxParams,
    scratch: &mut ScratchArena<BE>,
    lut: Lut<BE>,
    selectors_prep: &[CnvPVecL<BE::OwnedBuf, BE::DftWord, BE>],
    tree_selectors: &[GGSWPrepared<BE::OwnedBuf, BE>],
) -> BackendLWE<BE>
where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + 'static,
    Module<BE>: GLWESub<BE> + GLWEExternalProduct<BE> + LWESampleExtract<BE> + GLWEAdd<BE> + CnvPVecAlloc<BE> + Convolution<BE>
        + ModuleN + VecZnxIdftApplyTmpA<BE> + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes
        + VecZnxCopyBackend<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchArenaTakeBasic<'a, BE>,
    GGSWPrepared<BE::OwnedBuf, BE>: GGSWInfos + GGSWPreparedToBackendRef<BE> + 'static,
    BackendGLWE<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'a> BE::BufMut<'a>: HostDataMut + AsMut<[u8]> + AsRef<[u8]> + Sync,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    mem_span!("ohvpack_eval (all layers)");
    let n: usize = params.n_rlwe;
    let levels: usize = ohvpack_tree_depth(params.ohcollapse_n_candidates());
    assert_eq!(
        tree_selectors.len(),
        levels,
        "the CMux tree runs one level per bit of b, so it needs log2(alpha) = {levels} selectors"
    );

    // ---- First stage (OHCollapse) ----: alpha accumulators, each holding a candidate table
    // entry in its constant term.
    let mut cts: Vec<BackendGLWE<BE>> = ohcollapse(module, params, scratch, lut, selectors_prep);
    noise_span!(glwe_max: 1, &cts);

    // ---- Second stage: a binary CMux tree over the alpha accumulators ----
    // alpha leaves, so alpha - 1 internal nodes, all nodes of a level sharing one selector.
    mem_span!("second stage: CMux tree over alpha candidates");
    for level in 0..levels {
        if cts.len() == 1 {
            break;
        }
        // The first stage is layer 1, so this level is layer `level + 2`.
        mem_span!(crate::mem_profile::layer_label(level + 2));
        let bit: &GGSWPrepared<BE::OwnedBuf, BE> = &tree_selectors[level];

        let mut next: Vec<BackendGLWE<BE>> = Vec::with_capacity(cts.len() / 2);
        let mut pairs = cts.into_iter();
        while let Some(lo) = pairs.next() {
            let mut hi: BackendGLWE<BE> = pairs
                .next()
                .expect("alpha is a power of two, so every level has an even number of nodes");
            {
                time_span!("cmux_assign");
                cmux_assign(module, &mut hi, &lo, bit, scratch);
            }
            next.push(hi);
        }
        cts = next; // the level's inputs are freed here
        noise_span!(glwe_max: level + 2, &cts);
    }

    debug_assert_eq!(cts.len(), 1, "a power-of-two fold collapses to a single accumulator");

    // No final rotation: the first stage already folded `X^{-i0}` into the one-hot vector,
    // so the answer is the constant term.
    let result: BackendGLWE<BE> = cts.swap_remove(0);
    let mut lwe: BackendLWE<BE> = module.lwe_alloc(n.into(), params.base2k.into(), params.k_rlwe.into());
    {
        time_span!("lwe_sample_extract");
        module.lwe_sample_extract(&mut lwe, &result);
    }
    lwe
}
