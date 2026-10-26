//! OHVPack end-to-end: evaluate a LUT at a set of indices and check
//! each decryption.
//!
//! The sweeps cover the interior of the collapsing-factor range, a tree deeper than
//! `log2(N)`, both endpoints (`alpha = 1` and `alpha = nu`), and the full-LUT residency.

use ohvpack::context::{MuxLayouts, SecretKeys, secret_keys};
use ohvpack::lut::{Lut, LutMode, PreparedLut, build_lut};
use ohvpack::ohvpack::{ohvpack_eval, ohvpack_tree_depth};
use ohvpack::params::{MuxParams, init_toy};
use ohvpack::prepare_data::prepare_glwe;
use ohvpack::selector::ohvpack_make_selectors;
use ohvpack::vertical_packing::vpack_eval_tmp_bytes;

use poulpy_core::layouts::{BackendLWEPlaintext, GGSWPreparedFactory, ModuleCoreAlloc};
use poulpy_core::{GGSWEncryptSk, GLWEEncryptSk, LWEDecrypt};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CnvPVecL, Module, ScratchOwned},
};

use poulpy_cpu_ref::FFT64Ref;
type B = FFT64Ref;

/// Non-linear table: a linear `i % p` would alias several selection bugs onto the right
/// value, so a scrambled table is what actually pins down that the one-hot vector picked
/// the group, the CMux tree picked the polynomial, and the folded rotation picked the
/// coefficient.

fn run_ohvpack(params: &mut MuxParams, lut_indices: &[u32]) {
    run_ohvpack_mode(params, lut_indices, LutMode::Streaming)
}

fn run_ohvpack_mode(params: &mut MuxParams, lut_indices: &[u32], lut_mode: LutMode) {
    let layouts: MuxLayouts = MuxLayouts::new(params);
    let module: Module<B> = Module::<B>::new(params.n_rlwe as u64);

    // OHVPack contains no PackAuto, so its arena is sized for encryption and the CMux tree
    // only -- no `glwe_pack` term, and no automorphism keys anywhere in the evaluation.
    let mut scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&layouts.glwe)
            .max(module.ggsw_encrypt_sk_tmp_bytes(&layouts.ggsw))
            .max(module.ggsw_prepare_tmp_bytes(&layouts.ggsw))
            .max(module.lwe_decrypt_tmp_bytes(&layouts.lwe))
            // `u64` to match `benchmarks/bench_ohvpack.rs`. OHVPack drives `cmux_assign`
            // directly, so `T` never reaches the sizing arithmetic -- but the two callers
            // asking the same question should spell it the same way.
            .max(vpack_eval_tmp_bytes::<B, _, _>(&module, &layouts.glwe, &layouts.ggsw)),
    );

    let sk: SecretKeys<B> = secret_keys(&module, params, &layouts);
    let lut: Vec<i32> = build_lut(params.n_rlwe.pow(params.depth as u32), params.p);

    let prepared_lut: Option<PreparedLut<B>> = match lut_mode {
        LutMode::Streaming => None,
        LutMode::Full => Some(PreparedLut::for_tree(&module, params, &lut, &mut scratch.borrow())),
    };
    let lut_arg: Lut<B> = match &prepared_lut {
        Some(prep) => Lut::Full(prep),
        None => Lut::Streaming(&lut),
    };

    for &k in lut_indices {
        let (glwe_selectors, tree_selectors) = ohvpack_make_selectors::<B>(
            &module,
            params,
            &layouts.glwe,
            &layouts.ggsw,
            &sk.glwe_dft,
            k as i64,
            &mut scratch.borrow(),
        );

        // The one-hot selectors multiply every LUT polynomial, so their DFT is hoisted out
        // of the evaluation.
        let glwe_selectors_prep: Vec<CnvPVecL<Vec<u8>, <B as Backend>::DftWord, B>> = glwe_selectors
            .iter()
            .map(|sel| prepare_glwe(&module, sel, &mut scratch.borrow()))
            .collect();

        let result_lwe = ohvpack_eval::<B>(
            &module,
            params,
            &mut scratch.borrow(),
            lut_arg,
            &glwe_selectors_prep,
            &tree_selectors,
        );

        let mut result_pt: BackendLWEPlaintext<B> =
            module.lwe_plaintext_alloc(params.base2k.into(), params.k_rlwe.into());
        module.lwe_decrypt(&result_lwe, &mut result_pt, &sk.lwe, &mut scratch.borrow());
        let p: i64 = params.p as i64;
        let got: i64 = result_pt.decode_i64(params.log_p.into()).rem_euclid(p);
        let expected: i64 = (lut[k as usize] as i64).rem_euclid(p);
        assert_eq!(got, expected, "ohvpack_eval(k={k}) decoded {got} != expected {expected}");
    }
}

/// The shape §5.4 calls the interior: `alpha` strictly between the two degenerate
/// endpoints, so the one-hot vector is `G = nu/alpha` wide and the CMux tree runs
/// `log2(alpha)` levels over `alpha` accumulators.
#[test]
fn test_ohvpack() {
    let mut params: MuxParams = init_toy();
    params.alpha = 32;
    assert_eq!(params.depth, 3);
    assert_eq!(params.lut_n_polys(), 65536); // nu = N^2 at N = 256
    assert_eq!(params.ohcollapse_n_groups(), 2048); // G = nu/alpha
    assert_eq!(ohvpack_tree_depth(params.alpha), 5);

    let n: u32 = params.n_rlwe as u32;
    let alpha: u32 = params.alpha as u32;
    let lut_indices: Vec<u32> = vec![
        0,                                   // group 0, b = 0, i0 = 0
        1,                                   // group 0, b = 0, i0 = 1
        n - 1,                               // group 0, b = 0, i0 = N-1
        (5 * alpha + 7) * n + 130,           // group 5, b = 7, i0 = 130
        (2047 * alpha + 31) * n + (n - 1),   // last group, last b, last coefficient
        41337,                               // arbitrary
    ];
    run_ohvpack(&mut params, &lut_indices);
}

/// A deeper tree: `alpha = 1024` is 4N at N = 256, so the second stage runs 10 CMux levels.
/// The depth of OHVPack's tree is unconstrained by `N` (§5.3), which is exactly what this
/// exercises.
#[test]
fn test_ohvpack_deep_tree() {
    let mut params: MuxParams = init_toy();
    params.alpha = 1024;
    assert_eq!(ohvpack_tree_depth(params.alpha), 10);
    assert_eq!(params.ohcollapse_n_groups(), 64);

    let n: u32 = params.n_rlwe as u32;
    let alpha: u32 = params.alpha as u32;
    let lut_indices: Vec<u32> = vec![
        0,
        (3 * alpha + 260) * n + 42,          // b = 260 exercises bits 2 and 8 of the tree
        (63 * alpha + 1023) * n + (n - 1),   // last group, last candidate
    ];
    run_ohvpack(&mut params, &lut_indices);
}

/// `alpha = 1`: §5.4's lower endpoint. The tree is empty, `G = nu`, and the construction
/// collapses to the pure one-hot evaluation of §4.8 -- no CMux gate, no TRGSW selector
/// consumed, the first stage's single accumulator already carrying `T[i]` at its constant
/// term.
#[test]
fn test_ohvpack_alpha_one_is_pure_one_hot() {
    let mut params: MuxParams = init_toy();
    params.depth = 2;
    params.alpha = 1;
    assert_eq!(ohvpack_tree_depth(params.alpha), 0);
    assert_eq!(params.ohcollapse_n_groups(), params.lut_n_polys()); // G = nu

    let n: u32 = params.n_rlwe as u32;
    let lut_indices: Vec<u32> = vec![0, 1, n - 1, n, 5 * n + 130, 255 * n + 255, 41337];
    run_ohvpack(&mut params, &lut_indices);
}

/// `alpha = nu`: §5.4's upper endpoint. One group, so the one-hot vector reduces to the
/// single TRLWE ciphertext `[X^{-i0}]` and the construction becomes VPack with its rotation
/// performed first rather than last.

#[test]
fn test_ohvpack_alpha_equals_nu_is_vpack_rotated_first() {
    let mut params: MuxParams = init_toy();
    params.depth = 2;
    params.alpha = 256; // nu = N^(d-1) = 256
    assert_eq!(params.lut_n_polys(), params.alpha);
    assert_eq!(params.ohcollapse_n_groups(), 1); // a single group: one TRLWE selector
    assert_eq!(ohvpack_tree_depth(params.alpha), 8);

    let n: u32 = params.n_rlwe as u32;
    let lut_indices: Vec<u32> = vec![0, 1, n - 1, n, 5 * n + 130, 255 * n + 255, 41337];
    run_ohvpack(&mut params, &lut_indices);
}

/// Full-LUT counterpart: the same evaluation with every LUT polynomial prepared up front
/// instead of on the fly. Both residencies feed the first stage the same operands, so they
/// must decode identically.
#[test]
fn test_ohvpack_full_lut() {
    let mut params: MuxParams = init_toy();
    params.depth = 2;
    params.alpha = 32;
    run_ohvpack_mode(&mut params, &[0u32, 3 * 32 * 256 + 7 * 256 + 130], LutMode::Full);
}
