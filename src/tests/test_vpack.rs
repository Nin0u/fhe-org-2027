//! Vertical Packing end-to-end, in both residencies.
//!
//! [`LutMode::Full`] drives `vpack_eval` over a candidate table encrypted up front;
//! [`LutMode::Streaming`] drives `vpack_eval_streaming`, which encrypts each candidate on
//! demand and folds it LSB-first. The two must select the identical element, so both run
//! the same index sweeps.

use std::collections::HashMap;

use ohvpack::context::{MuxLayouts, SecretKeys, secret_keys};
use ohvpack::lut::{LutMode, build_lut};
use ohvpack::params::{MuxParams, init_toy};
use ohvpack::selector::{vpack_make_selectors, vpack_make_selectors_tmp_bytes};
use ohvpack::trivial::{TrivialOperand, trivial_glwe};
use ohvpack::vertical_packing::{
    FirstLevel, vpack_eval, vpack_eval_streaming, vpack_eval_tmp_bytes, vpack_prepare_first_level,
};

use poulpy_core::layouts::{
    BackendGLWE, BackendGLWEPlaintext, BackendLWE, BackendLWEPlaintext, GGSWPreparedFactory,
    GLWEPlaintextLayout, ModuleCoreAlloc,
};
use poulpy_core::{GLWEEncryptSk, LWEDecrypt, LWESampleExtract};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchArena, ScratchOwned},
};

use poulpy_cpu_ref::FFT64Ref;
type B = FFT64Ref;

/// Non-linear table: a linear `i % p` would alias several selection bugs onto the right
/// value (picking the wrong candidate or the wrong coefficient can still land on a
/// congruent entry), so a scrambled table is what actually pins down that step 1 chose the
/// candidate and step 2 rotated by exactly `k % N`.

/// Evaluates Vertical Packing on a `2^log_table_size` LUT at every index in `lut_indices`
/// and asserts each decodes back to `lut[k]`.
///
/// `prepared_first_level` picks between the two [`FirstLevel`] variants: transforming the
/// first level's operands up front, or inside the evaluation. They must select the same
/// element -- the credit §2.3.7 takes is about *when* a transform happens, not what it
/// computes -- so both are swept here.
fn run_vpack(mode: LutMode, log_table_size: usize, lut_indices: &[u32]) {
    run_vpack_first_level(mode, log_table_size, lut_indices, false)
}

fn run_vpack_first_level(mode: LutMode, log_table_size: usize, lut_indices: &[u32], prepared_first_level: bool) {
    let mut params: MuxParams = init_toy();
    let layouts: MuxLayouts = MuxLayouts::new(&params);
    let module: Module<B> = Module::<B>::new(params.n_rlwe as u64);

    // Candidates carry only the `log_p`-bit message, so their plaintext is one limb wide
    // rather than the full `k_rlwe` of `layouts.glwe_pt`.
    let pt_infos: GLWEPlaintextLayout = GLWEPlaintextLayout {
        n: params.n_rlwe.into(),
        base2k: params.base2k.into(),
        k: params.log_p.into(),
    };

    let n: usize = params.n_rlwe;
    let log_n: usize = n.trailing_zeros() as usize;
    let table_size: usize = 1 << log_table_size;
    let n_candidates: usize = table_size >> log_n;
    let log_p: usize = params.log_p;
    let p: i64 = params.p as i64;

    let mut scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(
        module
            .glwe_encrypt_sk_tmp_bytes(&layouts.glwe)
            .max(vpack_make_selectors_tmp_bytes(&module, n, &layouts.ggsw))
            .max(module.ggsw_prepare_tmp_bytes(&layouts.ggsw))
            .max(module.lwe_decrypt_tmp_bytes(&layouts.lwe))
            .max(vpack_eval_tmp_bytes::<B, _, _>(&module, &layouts.glwe, &layouts.ggsw)),
    );

    let sk: SecretKeys<B> = secret_keys(&module, &mut params, &layouts);
    let lut: Vec<i32> = build_lut(table_size, params.p);

    for &k in lut_indices {
        assert!((k as usize) < table_size, "selector {k} outside the 2^{log_table_size} table");

        let (tree_selector, rot_selector) = vpack_make_selectors::<u32, B>(
            &module,
            &mut params,
            &layouts.ggsw,
            &sk.glwe_dft,
            k,
            &mut scratch.borrow(),
        );

        // Candidate `c` holds entries `c*N .. (c+1)*N` in its N coefficients, so entry `k`
        // sits at coefficient `k % N` of candidate `k / N`. The candidates are **trivial**
        // ciphertexts `(0, P)`: §2.3.7 hands VPack its table as an input, and it is that
        // triviality the first tree level's credit is paid out of (see `ohvpack::trivial`).
        // Buffers are reused across calls.
        let mut pt: BackendGLWEPlaintext<B> = module.glwe_plaintext_alloc_from_infos(&pt_infos);
        let mut coeffs: Vec<i64> = vec![0i64; n];
        let mut gen_ct = |c: usize, _scratch: &mut ScratchArena<B>| -> BackendGLWE<B> {
            for (j, slot) in coeffs.iter_mut().enumerate() {
                *slot = (lut[c * n + j] as i64).rem_euclid(p);
            }
            pt.encode_vec_i64(&coeffs, log_p.into());
            trivial_glwe(&module, &layouts.glwe, &pt)
        };

        let mut res: BackendGLWE<B> = module.glwe_alloc_from_infos(&layouts.glwe);
        match mode {
            // Blind selection consumes the candidates in place, so each index needs its own
            // freshly built table.
            LutMode::Full => {
                let mut cts: Vec<BackendGLWE<B>> =
                    (0..n_candidates).map(|c| gen_ct(c, &mut scratch.borrow())).collect();
                // Built before the evaluation, as §2.2's convention has the table be.
                let ops: Vec<TrivialOperand<B>> = if prepared_first_level && n_candidates > 1 {
                    vpack_prepare_first_level(&module, &cts, &layouts.ggsw, &mut scratch.borrow())
                } else {
                    Vec::new()
                };
                let first_level: FirstLevel<B> =
                    if ops.is_empty() { FirstLevel::Streaming } else { FirstLevel::Prepared(&ops) };
                let cts_map: HashMap<usize, &mut BackendGLWE<B>> = cts.iter_mut().enumerate().collect();
                vpack_eval::<u32, B>(
                    &module,
                    &mut res,
                    cts_map,
                    first_level,
                    &tree_selector,
                    &rot_selector,
                    log_n,
                    log_table_size,
                    &mut scratch.borrow(),
                );
            }
            LutMode::Streaming => vpack_eval_streaming::<u32, B>(
                &module,
                &mut res,
                n_candidates,
                &mut gen_ct,
                FirstLevel::Streaming,
                &tree_selector,
                &rot_selector,
                log_n,
                log_table_size,
                &mut scratch.borrow(),
            ),
        }

        // The wanted entry is now the constant term.
        let mut res_lwe: BackendLWE<B> = module.lwe_alloc_from_infos(&layouts.lwe);
        module.lwe_sample_extract(&mut res_lwe, &res);
        let mut res_pt: BackendLWEPlaintext<B> =
            module.lwe_plaintext_alloc(params.base2k.into(), params.k_rlwe.into());
        module.lwe_decrypt(&res_lwe, &mut res_pt, &sk.lwe, &mut scratch.borrow());

        let got: i64 = res_pt.decode_i64(params.log_p.into()).rem_euclid(p);
        let expected: i64 = (lut[k as usize] as i64).rem_euclid(p);
        assert_eq!(got, expected, "vpack_eval(k={k}, {mode:?}) decoded {got} != expected {expected}");
    }
}

/// The `r = k % N` shapes the `X^{-r}` encoding branches on -- r = 0 (the GGSW is
/// `X^0 = 1`), r = 1, r = N-1 -- each at a different candidate index `k / N`, so a failure
/// in the tree and a failure in the rotation stay distinguishable. The stride sweep then
/// spreads over both halves of the index space.
fn sweep_indices(n: u32) -> Vec<u32> {
    let mut lut_indices: Vec<u32> = vec![
        0,             // candidate 0,    r = 0
        1,             // candidate 0,    r = 1
        n - 1,         // candidate 0,    r = N-1
        n,             // candidate 1,    r = 0
        5 * n + 130,   // candidate 5,    r = 130
        255 * n + 255, // last candidate, r = N-1
    ];
    lut_indices.extend((0..64u32).map(|i| i.wrapping_mul(7919) % (1 << 16)));
    lut_indices
}

/// Full two-step evaluation over a 2^16 table: N = 256 (log_n = 8) gives 256 candidates,
/// so the CMux tree runs over `log_table_size - log_n` = 8 selector bits and step 2 is the
/// single external product.
#[test]
fn test_vpack_eval() {
    let n: u32 = init_toy().n_rlwe as u32;
    run_vpack(LutMode::Full, 16, &sweep_indices(n));
}

/// A LUT of exactly one polynomial degenerates the CMux tree to zero bits, leaving the
/// ring rotation as the whole evaluation. Sweeping every `r` here is what pins the
/// `X^{-r}` encoding down across its full range, independently of the tree.
#[test]
fn test_vpack_eval_single_candidate() {
    let n: u32 = init_toy().n_rlwe as u32;
    run_vpack(LutMode::Full, n.trailing_zeros() as usize, &(0..n).collect::<Vec<u32>>());
}

/// Streaming counterpart of [`test_vpack_eval`]. Passing here pins that the LSB-first
/// streaming fold selects the identical element as the in-place block-pair tree across the
/// whole index space.
#[test]
fn test_vpack_eval_streaming() {
    let n: u32 = init_toy().n_rlwe as u32;
    run_vpack(LutMode::Streaming, 16, &sweep_indices(n));
}

/// Streaming counterpart of [`test_vpack_eval_single_candidate`]: a single-candidate LUT
/// degenerates the fold to zero levels, so this isolates that the streaming path also
/// leaves the ring rotation as the whole evaluation.
#[test]
fn test_vpack_eval_streaming_single_candidate() {
    let n: u32 = init_toy().n_rlwe as u32;
    run_vpack(LutMode::Streaming, n.trailing_zeros() as usize, &(0..n).collect::<Vec<u32>>());
}

/// [`FirstLevel::Prepared`]: the first tree level's operands transformed before the
/// evaluation rather than inside it, which is the residency §2.2's convention prices and
/// the one that takes the whole `2l` transforms per gate §2.3.7 credits.
///
/// The two residencies must select the identical element -- the credit is about when a
/// transform happens, not about what it computes -- so this is [`test_vpack_eval`]'s sweep
/// run again over the prepared operands.
#[test]
fn test_vpack_eval_prepared_first_level() {
    let n: u32 = init_toy().n_rlwe as u32;
    run_vpack_first_level(LutMode::Full, 16, &sweep_indices(n), true);
}
