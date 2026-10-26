// Vertical Packing.
//
// The LUT is packed `N` entries per GLWE ciphertext, so entry `k` lives at coefficient
// `k % N` of candidate `k / N`. Evaluation therefore splits the selector in two at `log2(N)`:
//
//  1. the high `log2(LUT_size) - log2(N)` bits pick the candidate, via a CMux tree over
//     that many GGSW selector bits (`2^(log2(LUT_size) - log2(N)) - 1` external
//     products);
//  2. the low `log2(N)` bits rotate the wanted coefficient down to the constant term.
//
// The original Vertical Packing spends a blind rotation on step 2, i.e. one CMux per bit,
// `log2(N)` external products. This is the optimized version: the remainder
// `r = selector % N` is encoded *directly* as an encrypted ring rotation `X^{-r}` in a
// single GGSW, so step 2 costs exactly one external product regardless of `N`.
//
// The candidates are **trivial ciphertexts** `(0, P)`, as §2.3.7 specifies -- the table is
// an input, not something the evaluation encrypts. That is what makes the first tree level
// cheaper than the ones above it, and the credit is taken here rather than assumed: see
// [`crate::trivial`] for the two halves of it, and [`FirstLevel`] for which of them a given
// run takes. Feeding genuine ciphertexts instead (a `glwe_encrypt_sk` table) costs the whole
// credit -- `(2l + 2)nu` transforms against the model's `(l + 2)nu`, up to 1.5x at `l = 2`.

use std::collections::HashMap;

use poulpy_bin_fhe::bdd_arithmetic::{FheUintPrepared, GetGGSWBit, UnsignedInteger};
use poulpy_core::{
    GLWEAdd, GLWECopy, GLWEExternalProduct, GLWENormalize, GLWESub, ScratchArenaTakeCore,
    api::GLWEBytesOf,
    layouts::{
        BackendGLWE, GGSWInfos, GGSWPrepared, GGSWPreparedToBackendRef, GLWEInfos, GLWEToBackendMut,
        GLWEToBackendRef, ModuleCoreAlloc,
    },
};
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxBigBytesOf, VecZnxBigNormalize,
        VecZnxBigNormalizeTmpBytes, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf,
        VecZnxDftZero, VecZnxIdftApply, VecZnxIdftApplyTmpBytes, VmpApplyDftToDft,
        VmpApplyDftToDftTmpBytes,
    },
    layouts::{Backend, HostDataMut, Module, ScratchArena},
};

use crate::cmux::{cmux_assign, cmux_assign_tmp_bytes};
use crate::trivial::{
    TrivialOperand, glwe_external_product_trivial, glwe_external_product_trivial_tmp_bytes,
    trivial_operand_fill_tmp_bytes,
};
use crate::{mem_span, noise_span, time_span};

/// Where the first tree level's operands come from, and hence which part of §2.3.7's
/// credit a run takes.
///
/// The level's gates combine two table polynomials and nothing else, so their gadget
/// decomposition is table-only data: `2l` forward transforms per gate that §2.2's
/// convention charges to no one. Both variants below are honest about a different half of
/// that, and neither pretends to work it has not done:
///
///  - [`FirstLevel::Prepared`] -- the operands are transformed *before* the evaluation, so
///    the level pays no forward transform at all. This is the table "supplied in
///    transformed form" the convention describes, and it is what
///    [`vpack_prepare_first_level`] builds; the cost is holding `nu/2` transformed
///    operands.
///  - [`FirstLevel::Streaming`] -- each operand is transformed inside the evaluation, into
///    one reused buffer. The A-half is still free (it is zero, and is never transformed),
///    so the level pays `l` forward transforms per gate rather than `2l`. This is the same
///    trade OHVPack makes between [`crate::lut::LutMode`]s: the table's
///    transforms are the term that scales with the table, and streaming keeps it off the
///    heap by paying for it online.
pub enum FirstLevel<'a, BE: Backend> {
    /// Operand `i` is the prepared body of `candidate[2i+1] - candidate[2i]`.
    Prepared(&'a [TrivialOperand<BE>]),
    /// Prepare each operand inside the evaluation.
    Streaming,
}

impl<BE: Backend> Clone for FirstLevel<'_, BE> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<BE: Backend> Copy for FirstLevel<'_, BE> {}

impl<BE: Backend> FirstLevel<'_, BE> {
    /// The prepared operand of gate `pair`, or `None` when the caller prepares it itself.
    fn operand(&self, pair: usize) -> Option<&TrivialOperand<BE>> {
        match self {
            FirstLevel::Prepared(ops) => Some(&ops[pair]),
            FirstLevel::Streaming => None,
        }
    }
}

/// Scratch bytes required by [`vpack_eval`], sized against every step.
///
/// The deeper tree levels and step 2 are external products -- the gates are
/// [`crate::cmux::cmux_assign`], step 2 is one directly -- and the first level is the
/// trivial-operand product of [`crate::trivial`], which is the same shape minus the
/// re-limbing buffer plus, in [`FirstLevel::Streaming`], the one that prepares the operand.
pub fn vpack_eval_tmp_bytes<BE, R, K>(module: &Module<BE>, res_infos: &R, ggsw_infos: &K) -> usize
where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + 'static,
    Module<BE>: GLWEExternalProduct<BE> + GLWENormalize<BE> + GLWEBytesOf<BE>
        + VecZnxDftBytesOf + VecZnxBigBytesOf + VmpApplyDftToDftTmpBytes
        + VecZnxIdftApplyTmpBytes + VecZnxBigNormalizeTmpBytes,
    R: GLWEInfos,
    K: GGSWInfos,
{
    let first_level: usize = glwe_external_product_trivial_tmp_bytes(module, res_infos, ggsw_infos)
        .max(trivial_operand_fill_tmp_bytes(module, res_infos, ggsw_infos));
    cmux_assign_tmp_bytes(module, res_infos, ggsw_infos)
        .max(module.glwe_external_product_tmp_bytes(res_infos, res_infos, ggsw_infos))
        .max(first_level)
}

/// The first tree level's operands, transformed up front: entry `i` is the body of
/// `cts[2i+1] - cts[2i]`, re-limbed to the GGSW's `base2k`.
///
/// This is the table "supplied in transformed form" of §2.2, so it belongs *outside* the
/// timed region -- it is the whole point of [`FirstLevel::Prepared`]. It holds `nu/2`
/// transformed operands, which is why the streaming evaluation does not use it.
///
/// `cts` must be the trivial candidates in index order, and it is only read: the
/// difference of two trivial ciphertexts is trivial, so the mask half never enters.
pub fn vpack_prepare_first_level<BE, K>(
    module: &Module<BE>,
    cts: &[BackendGLWE<BE>],
    ggsw_infos: &K,
    scratch: &mut ScratchArena<BE>,
) -> Vec<TrivialOperand<BE>>
where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + 'static,
    Module<BE>: GLWESub<BE> + GLWECopy<BE> + GLWENormalize<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>, ZnxWord = i64>
        + ModuleN + VecZnxDftAlloc<BE> + VecZnxDftZero<BE> + VecZnxDftApply<BE>,
    K: GGSWInfos,
    BackendGLWE<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchArenaTakeBasic<'a, BE>,
{
    assert!(cts.len().is_multiple_of(2), "the first level pairs candidates up, so it needs an even count");
    let mut diff: BackendGLWE<BE> = module.glwe_alloc_from_infos(&cts[0]);
    cts.chunks_exact(2)
        .map(|pair| {
            module.glwe_copy(&mut diff, &pair[1]);
            module.glwe_sub_assign(&mut diff, &pair[0]);
            let mut op: TrivialOperand<BE> = TrivialOperand::alloc(module, &cts[0], ggsw_infos);
            op.fill(module, &diff, scratch);
            op
        })
        .collect()
}

/// One gate of the first tree level: `res <- (res - a) * s + a`, over trivial operands.
///
/// `prepared` is the transformed body of `res - a` when the caller supplied one; otherwise
/// the difference is taken here and transformed into `buf`. Either way no forward transform
/// of the mask half happens -- it is zero, and [`TrivialOperand`] never touches it.
fn cmux_trivial<BE>(
    module: &Module<BE>,
    res: &mut BackendGLWE<BE>,
    a: &BackendGLWE<BE>,
    s: &GGSWPrepared<BE::OwnedBuf, BE>,
    prepared: Option<&TrivialOperand<BE>>,
    buf: &mut TrivialOperand<BE>,
    scratch: &mut ScratchArena<BE>,
) where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + 'static,
    Module<BE>: GLWESub<BE> + GLWEAdd<BE> + GLWENormalize<BE> + ModuleN + VecZnxDftApply<BE>
        + VecZnxDftBytesOf + VecZnxBigBytesOf + VmpApplyDftToDft<BE> + VmpApplyDftToDftTmpBytes
        + VecZnxIdftApply<BE> + VecZnxIdftApplyTmpBytes + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    BackendGLWE<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchArenaTakeBasic<'a, BE>,
{
    let op: &TrivialOperand<BE> = match prepared {
        Some(op) => op,
        None => {
            // `res - a` is again trivial, and its body is all the product reads.
            module.glwe_sub_assign(res, a);
            buf.fill(module, &*res, scratch);
            &*buf
        }
    };
    // No `time_span!` on either half: both sit inside the tree's own span, and the
    // time-profile report derives `other` as `total - sum(spans)`, so a nested span would
    // be counted twice.
    glwe_external_product_trivial(module, res, op, s, scratch);
    module.glwe_add_assign(res, a);
}

/// Two-step Vertical Packing evaluation: `res <- LUT[k]` in the constant term.
///
/// `cts` maps candidate index `c` to the GLWE packing LUT entries `c*N .. (c+1)*N`; the
/// tree consumes it in place. `tree_selector` supplies the CMux-tree bits (only bits
/// `log_n .. log_table_size` of `k` are read), and `rot_selector` is the GGSW encrypting
/// `X^{-(k % N)}` built by [`vpack_make_selectors`].
///
/// The tree is poulpy's `glwe_blind_selection` rewritten over [`cmux_assign`], for the
/// reason given in [`crate::cmux`]: the gate poulpy's own selection is built from skips the
/// re-limbing the external product needs whenever the GLWE and the GGSW disagree on
/// `base2k`, which is every parameter set `paramux` emits. Folding LSB-first over adjacent
/// pairs rather than MSB-first over blocks reaches the same leaf and lets
/// [`vpack_eval_streaming`] share the schedule.
pub fn vpack_eval<T, BE>(
    module: &Module<BE>,
    res: &mut BackendGLWE<BE>,
    cts: HashMap<usize, &mut BackendGLWE<BE>>,
    first_level: FirstLevel<BE>,
    tree_selector: &FheUintPrepared<BE::OwnedBuf, T, BE>,
    rot_selector: &GGSWPrepared<BE::OwnedBuf, BE>,
    log_n: usize,
    log_table_size: usize,
    scratch: &mut ScratchArena<BE>,
) where
    T: UnsignedInteger,
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + 'static,
    Module<BE>: GLWESub<BE> + GLWEAdd<BE> + GLWECopy<BE> + GLWEExternalProduct<BE>
        + GLWENormalize<BE> + ModuleN + VecZnxDftAlloc<BE> + VecZnxDftZero<BE>
        + VecZnxDftApply<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes + VecZnxIdftApply<BE> + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    FheUintPrepared<BE::OwnedBuf, T, BE>: GetGGSWBit<BE> + 'static,
    GGSWPrepared<BE::OwnedBuf, BE>: GGSWInfos + GGSWPreparedToBackendRef<BE> + 'static,
    BackendGLWE<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchArenaTakeBasic<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    mem_span!("vpack_eval");
    debug_assert!(log_table_size >= log_n, "LUT must span at least one candidate polynomial");
    let levels: usize = log_table_size - log_n;

    // Step 1: CMux tree over the high `levels` bits -> the surviving candidate is the
    // packed GLWE holding the wanted entry at coefficient k % N.
    //
    // The candidates arrive as `&mut`, so the fold runs on them in place and allocates
    // nothing: each level pairs them up, `cmux_assign` writes the winner into the `hi`
    // half, and the `lo` half is dropped from the working set.
    {
        mem_span!("blind_selection (CMux tree)");
        time_span!("blind_selection (CMux tree)");
        let mut cts: HashMap<usize, &mut BackendGLWE<BE>> = cts;
        let mut live: Vec<&mut BackendGLWE<BE>> = (0..1usize << levels)
            .map(|c| cts.remove(&c).expect("vpack_eval needs every candidate 0..2^(log_table_size - log_n)"))
            .collect();
        // The level-0 gates consume the table as trivial ciphertexts. `buf` backs the
        // [`FirstLevel::Streaming`] case and is one operand wide, not `nu/2`; it is
        // allocated on first use, so [`FirstLevel::Prepared`] never holds it. It is shaped
        // against `rot_selector` because every GGSW here shares one layout and that one
        // exists at every table size, where `get_bit` has no level 0 to read at `nu = 1`.
        let mut buf: Option<TrivialOperand<BE>> = None;
        for level in 0..levels {
            let bit: &GGSWPrepared<BE::OwnedBuf, BE> = tree_selector.get_bit(log_n + level);
            let mut next: Vec<&mut BackendGLWE<BE>> = Vec::with_capacity(live.len() / 2);
            let mut pairs = live.into_iter();
            let mut pair: usize = 0;
            while let Some(lo) = pairs.next() {
                let hi: &mut BackendGLWE<BE> = pairs.next().expect("a power-of-two level has an even number of nodes");
                // `cmux(hi, lo, bit)` keeps `hi` when the bit is 1 and `lo` when it is 0 --
                // the pair differing in bit `log_n + level` of `k`.
                if level == 0 {
                    let prepared: Option<&TrivialOperand<BE>> = first_level.operand(pair);
                    let op: &mut TrivialOperand<BE> =
                        buf.get_or_insert_with(|| TrivialOperand::alloc(module, &*hi, rot_selector));
                    cmux_trivial(module, hi, &*lo, bit, prepared, op, scratch);
                } else {
                    cmux_assign(module, hi, &*lo, bit, scratch);
                }
                pair += 1;
                next.push(hi);
            }
            live = next;
        }
        debug_assert_eq!(live.len(), 1, "a power-of-two fold collapses to a single candidate");
        module.glwe_copy(res, &*live[0]);
    }
    // The tree is reported as a single layer rather than one row per level: layer 1 is the
    // selected candidate, layer 2 the rotated result.
    noise_span!(glwe: 1, &*res);

    // Step 2: one external product against X^{-(k % N)} brings that coefficient to the
    // constant term. This replaces the log_n-CMux blind rotation of the original scheme.
    {
        mem_span!("glwe_external_product (rotation)");
        time_span!("glwe_external_product (rotation)");
        module.glwe_external_product_assign(res, rot_selector, scratch);
    }
    noise_span!(glwe: 2, &*res);
}

/// Streaming Vertical Packing evaluation: same result as [`vpack_eval`], but the candidate
/// table is never fully resident.
///
/// [`vpack_eval`] takes every candidate up front (a `HashMap` of all `N^(depth-1)` GLWEs),
/// which at e.g. `N = 2048, depth = 3` is `2^22` ciphertexts -- hundreds of GiB. Its fold is
/// the same one this runs, but it is handed the whole table and cannot avoid holding it.
///
/// The selection is a reduction, and reductions can be folded over an *ordered stream*:
/// `gen_candidate(c, scratch)` produces candidate `c` on demand, and each is folded into a
/// small stack of partial results. Only `O(log_2(n_candidates))` partial GLWEs are ever live
/// (≈23 at `2^22` candidates), so peak memory drops from `O(n_candidates)` to
/// `O(log n_candidates)`.
///
/// `n_candidates` must be `2^(log_table_size - log_n)` (the scheme always packs a power of
/// two of candidates). Scratch is sized exactly as for [`vpack_eval`] via
/// [`vpack_eval_tmp_bytes`]; `gen_candidate` receives the same `scratch` between folds.
#[allow(clippy::too_many_arguments)]
pub fn vpack_eval_streaming<T, BE>(
    module: &Module<BE>,
    res: &mut BackendGLWE<BE>,
    n_candidates: usize,
    mut gen_candidate: impl FnMut(usize, &mut ScratchArena<BE>) -> BackendGLWE<BE>,
    first_level: FirstLevel<BE>,
    tree_selector: &FheUintPrepared<BE::OwnedBuf, T, BE>,
    rot_selector: &GGSWPrepared<BE::OwnedBuf, BE>,
    log_n: usize,
    log_table_size: usize,
    scratch: &mut ScratchArena<BE>,
) where
    T: UnsignedInteger,
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + 'static,
    Module<BE>: GLWESub<BE> + GLWEAdd<BE> + GLWEExternalProduct<BE>
        + GLWENormalize<BE> + ModuleN + VecZnxDftAlloc<BE> + VecZnxDftZero<BE>
        + VecZnxDftApply<BE> + VecZnxDftBytesOf + VecZnxBigBytesOf + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes + VecZnxIdftApply<BE> + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE> + VecZnxBigNormalizeTmpBytes,
    FheUintPrepared<BE::OwnedBuf, T, BE>: GetGGSWBit<BE> + 'static,
    GGSWPrepared<BE::OwnedBuf, BE>: GGSWInfos + GGSWPreparedToBackendRef<BE> + 'static,
    BackendGLWE<BE>: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE> + ScratchArenaTakeBasic<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut + AsMut<[u8]> + AsRef<[u8]> + Sync,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    mem_span!("vpack_eval_streaming");
    debug_assert!(log_table_size >= log_n, "LUT must span at least one candidate polynomial");
    let bit_mask: usize = log_table_size - log_n;
    let bit_rsh: usize = log_n;
    assert_eq!(
        n_candidates,
        1usize << bit_mask,
        "streaming vpack expects a power-of-two candidate count (2^{bit_mask}), got {n_candidates}"
    );

    // Step 1: streaming CMux tree. `stack` holds `(level, partial)` pairs, innermost last;
    // a fully reduced block of `2^level` candidates sits at each entry. Folding a new
    // candidate merges equal-level tops until the level is unique, exactly a bottom-up
    // (LSB-first) binary reduction -- so at most `bit_mask + 1` partials are ever resident.
    let mut stack: Vec<(usize, BackendGLWE<BE>)> = Vec::with_capacity(bit_mask + 1);
    {
        mem_span!("blind_selection (streaming CMux tree)");
        time_span!("blind_selection (streaming CMux tree)");
        // One operand wide: the fold has at most one level-0 gate open at a time, so
        // [`FirstLevel::Streaming`] costs a single buffer here however large the table is.
        // Allocated on first use, so [`FirstLevel::Prepared`] never holds it.
        let mut buf: Option<TrivialOperand<BE>> = None;
        for c in 0..n_candidates {
            let mut cur: BackendGLWE<BE> = gen_candidate(c, scratch);
            let mut level: usize = 0;
            while stack.last().is_some_and(|(l, _)| *l == level) {
                let (_, lo) = stack.pop().unwrap();
                // `lo` is the lower-index block (pushed earlier), `cur` the higher one.
                // The gate computes `cur = (cur - lo)*bit + lo`, i.e. bit=1 takes the
                // higher block `cur`, bit=0 the lower `lo` -- the same selection
                // [`vpack_eval`] makes at its level `level`. The merged block stays in
                // `cur`.
                let bit: &GGSWPrepared<BE::OwnedBuf, BE> = tree_selector.get_bit(bit_rsh + level);
                if level == 0 {
                    // Both blocks are still table polynomials, so this is the credited
                    // level; `c/2` is its gate index, the same one [`vpack_eval`] pairs.
                    let prepared: Option<&TrivialOperand<BE>> = first_level.operand(c / 2);
                    let op: &mut TrivialOperand<BE> =
                        buf.get_or_insert_with(|| TrivialOperand::alloc(module, &cur, rot_selector));
                    cmux_trivial(module, &mut cur, &lo, bit, prepared, op, scratch);
                } else {
                    cmux_assign(module, &mut cur, &lo, bit, scratch);
                }
                level += 1;
            }
            stack.push((level, cur));
        }
    }
    debug_assert_eq!(stack.len(), 1, "a power-of-two fold collapses to a single block");
    let (_, selected): (usize, BackendGLWE<BE>) = stack.pop().expect("non-empty candidate set");
    noise_span!(glwe: 1, &selected);

    // Step 2: rotate the wanted coefficient (k % N) down to the constant term.
    {
        mem_span!("glwe_external_product (rotation)");
        time_span!("glwe_external_product (rotation)");
        module.glwe_external_product(res, &selected, rot_selector, scratch);
    }
    noise_span!(glwe: 2, &*res);
}
