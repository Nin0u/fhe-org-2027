//! Vertical Packing benchmark.
//!
//! Four mutually exclusive modes, selected by cargo feature:
//!   cargo bench --bench vpack                          # wall clock, 10 fixed samples
//!   cargo bench --bench vpack --features time-profile  # per-operation timing breakdown
//!   cargo bench --bench vpack --features mem-profile   # per-section peak heap
//!   cargo bench --bench vpack --features noise-profile # per-layer output noise variance
//! The default mode uses `ohvpack::bench_timing` rather than Criterion, so the number of
//! evaluations is fixed instead of derived from a time budget (see that module). The
//! `mem-profile` mode additionally installs an instrumenting global allocator.
//!
//! Orthogonal to those, `LUT_MODE` picks how the candidate table is held:
//!   LUT_MODE=streaming (default)  # `vpack_eval_streaming`: candidates built on demand
//!   LUT_MODE=full                 # `vpack_eval`: the whole table built up front
//! It is a runtime variable rather than a cargo feature so one build measures both. Note
//! that the two do not time the same work: the streaming eval generates its candidates as
//! it folds them, so building them sits *inside* its timed region, whereas the full eval
//! only reads a table that already exists. Compare it against the streaming table build,
//! not against the full eval alone.
//!
//! The candidates are trivial ciphertexts `(0, P)`, which is how §2.3.7 supplies the table
//! and what its transform count assumes. `LUT_MODE` therefore also picks which half of the
//! first tree level's credit the run takes: `full` prepares that level's operands outside
//! the timed region (`FirstLevel::Prepared`, the model's own `(l + 2)nu`), `streaming`
//! transforms each inside it and pays `l` forward transforms per gate rather than the
//! uncredited `2l`. See `ohvpack::trivial` and `ohvpack::vertical_packing::FirstLevel`.

use std::collections::HashMap;
use std::time::Instant;

use ohvpack::bench_timing::run_modes_sampled;
use ohvpack::context::{MuxLayouts, SecretKeys, secret_keys};
use ohvpack::lut::{LutMode, lut_entry};
use ohvpack::mem_profile::Span;
use ohvpack::params::{MuxParams, params_from_json};
use ohvpack::selector::{vpack_make_selectors, vpack_make_selectors_tmp_bytes};
use ohvpack::trivial::{TrivialOperand, trivial_glwe};
use ohvpack::vertical_packing::{
    FirstLevel, vpack_eval, vpack_eval_streaming, vpack_eval_tmp_bytes, vpack_prepare_first_level,
};

use poulpy_core::layouts::{BackendGLWE, BackendGLWEPlaintext, ModuleCoreAlloc};
use poulpy_core::GLWEEncryptSk;
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Module, ScratchArena, ScratchOwned},
};

#[cfg(feature = "mem-profile")]
#[global_allocator]
static GLOBAL: ohvpack::mem_profile::TrackingAlloc = ohvpack::mem_profile::TrackingAlloc;

// Backend selected at build time via Cargo features (see Cargo.toml):
//   default           -> reference backend
//   --features avx    -> AVX backend
//   --features avx512 -> AVX-512 backend
#[cfg(feature = "avx512")]
use poulpy_cpu_avx512::FFT64Avx512 as B;
#[cfg(all(feature = "avx", not(feature = "avx512")))]
use poulpy_cpu_avx::FFT64Avx as B;
#[cfg(not(any(feature = "avx", feature = "avx512")))]
use poulpy_cpu_ref::FFT64Ref as B;

/// The index the LUT is evaluated at. Every cost here is index-independent, so a fixed one
/// is representative.
///
/// `u64`, not `u32`: the CMux tree reads selector bits up to `log_table_size - 1`, which is
/// 32 for a 2^33 table (N=2048, d=3). A `u32` FheUint only holds bit-ciphertexts 0..31, so
/// `get_bit(32)` panics -- the selector width, not memory, is what caps the table size.
const SELECTOR: u64 = 2048;

fn main() {
    // Parameters file: taken from the `PARAMS_FILE` environment variable so it can be
    // supplied on the benchmark command line, e.g.
    //   PARAMS_FILE=../paramux/example_finder/vertical_packing/depth_2/s1.json cargo bench --bench vpack
    let params_path: String = std::env::var("PARAMS_FILE")
        .expect("PARAMS_FILE environment variable must be set to the parameters JSON path");
    let param_set: Vec<MuxParams> = params_from_json(&params_path);
    // Streaming or fully materialised candidate table, from `LUT_MODE` (see the header).
    let lut_mode: LutMode = LutMode::from_env();

    for mut params in param_set {
        // Memory mode: reset the peak counter at the top of the iteration, then open a
        // `setup` span covering the module, scratch and keys, which runs until just before
        // the evaluation.
        //
        // Unlike the tree benches the `lut` section is nested inside `setup` and only
        // exists in full mode: it holds the trivial candidate table and the first tree
        // level's prepared operands. In streaming mode there is no such section at all --
        // the table never exists as a block and its cost shows up inside the eval spans
        // instead.
        // Nor are there layer rows -- vertical packing is a single CMux tree plus the
        // closing rotation, which the eval already reports as two sections. Every eval
        // span's baseline is this resident setup.
        ohvpack::mem_profile::reset();
        let setup_span = Span::new("setup");

        let layouts: MuxLayouts = MuxLayouts::new(&params);
        let module: Module<B> = Module::<B>::new(params.n_rlwe as u64);

        let n: usize = params.n_rlwe;
        let log_n: usize = n.trailing_zeros() as usize;
        let log_table_size: usize = log_n * params.depth;
        let n_candidates: usize = 1 << (log_table_size - log_n);
        let p: i64 = params.p as i64;
        let log_p: usize = params.log_p;

        let sk: SecretKeys<B> = secret_keys(&module, &mut params, &layouts);

        // Scratch space -- sized against every op invoked in the timed region. Vertical
        // packing neither packs nor keyswitches, so `context::scratch_bytes` would size for
        // keys this bench never builds.
        let mut scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(
            module
                .glwe_encrypt_sk_tmp_bytes(&layouts.glwe)
                .max(vpack_make_selectors_tmp_bytes(&module, params.n_rlwe, &layouts.ggsw))
                .max(vpack_eval_tmp_bytes::<B, _, _>(&module, &layouts.glwe, &layouts.ggsw)),
        );

        // Encrypt the two selectors: the CMux-tree bits of `SELECTOR`, and the single GGSW
        // ring rotation X^{-(SELECTOR % N)} that step 2 consumes.
        let (tree_selector, rot_selector) = vpack_make_selectors::<u64, B>(
            &module,
            &mut params,
            &layouts.ggsw,
            &sk.glwe_dft,
            SELECTOR,
            &mut scratch.borrow(),
        );

        // Candidate `c` packs LUT entries `c*N .. (c+1)*N` at coefficients `0 .. N`, as the
        // trivial ciphertext `(0, P_c)` -- the table is an input,
        // not something the evaluation encrypts (§2.3.7). One generator, used by both
        // residencies: full mode calls it `n_candidates` times up front, streaming mode
        // inside the fold. The buffers are reused across calls.
        let mut pt: BackendGLWEPlaintext<B> = module.glwe_plaintext_alloc_from_infos(&layouts.glwe_pt);
        let mut coeffs: Vec<i64> = vec![0i64; n];
        let mut gen_ct = |c: usize, _scratch: &mut ScratchArena<B>| -> BackendGLWE<B> {
            let base_idx: usize = c * n;
            for (j, slot) in coeffs.iter_mut().enumerate() {
                // `lut_entry`, not `(base_idx + j) % p`: `p` divides `N` at every parameter
                // set, so the linear table makes all `nu` candidates *bit-identical*. The
                // first level's `hi - lo` is then zero, and the VMP skips a zero operand
                // outright -- the credited level would measure as free for a reason that
                // has nothing to do with §2.3.7. See `lut::build_lut`.
                *slot = lut_entry(base_idx + j, p as usize) as i64;
            }
            // At `log_p`, the scale the result is decoded at: encoding at `base2k` would
            // put the message `2^(base2k - log_p)` too low, which costs the timing modes
            // nothing but makes `noise-profile` count the whole message as noise (the
            // probe's `decode_i64(log_p)` reads 0). See `tests/test_vpack.rs`.
            pt.encode_vec_i64(&coeffs, log_p.into());
            trivial_glwe(&module, &layouts.glwe, &pt)
        };

        // Full mode evaluates the in-place `vpack_eval`, whose blind selection combines
        // far-apart block pairs and so needs the whole candidate table resident: build it
        // up front. Streaming mode instead generates candidates on demand inside
        // `vpack_eval_streaming` and never materialises the table -- for a 2^33 LUT the full
        // table is 2^22 GLWEs (~hundreds of GiB), which is exactly what overflowed 64 GiB
        // here.
        let mut cts: Vec<BackendGLWE<B>> = match lut_mode {
            LutMode::Streaming => Vec::new(),
            LutMode::Full => {
                let _lut_span = Span::new("lut (trivial table)");
                (0..n_candidates).map(|c| gen_ct(c, &mut scratch.borrow())).collect()
            }
        };

        // The first tree level's operands, transformed here rather than inside the
        // evaluation: §2.2 charges no transform to the table, and this is what supplying it
        // in transformed form means. Streaming mode cannot hoist them without holding
        // `n_candidates/2` of them, which is the thing it exists to avoid, so it prepares
        // each inside the fold and pays for it.
        let first_level_ops: Vec<TrivialOperand<B>> = match lut_mode {
            LutMode::Streaming => Vec::new(),
            LutMode::Full if n_candidates > 1 => {
                let _ops_span = Span::new("lut (first level, transformed)");
                vpack_prepare_first_level(&module, &cts, &layouts.ggsw, &mut scratch.borrow())
            }
            LutMode::Full => Vec::new(),
        };
        let first_level: FirstLevel<B> = if first_level_ops.is_empty() {
            FirstLevel::Streaming
        } else {
            FirstLevel::Prepared(&first_level_ops)
        };

        // Destination ciphertext (holds the selected, then rotated, GLWE).
        let mut res: BackendGLWE<B> = module.glwe_alloc_from_infos(&layouts.glwe);

        let label: String = format!("vpack candidates={n_candidates} lut={}", lut_mode.as_str());
        run_modes_sampled(
            &label,
            log_p,
            setup_span,
            // Noise mode only: install the decryption probes. There is no layer-0 row here
            // -- the CMux tree runs inside poulpy's blind selection and reports as a single
            // layer (see `ohvpack::vertical_packing`).
            || {
                #[cfg(feature = "noise-profile")]
                ohvpack::noise_profile::install_probe::<B>(n, log_p, &layouts, &sk.glwe);
            },
            // Full mode rebuilds its candidate map before every sample -- blind selection
            // mutates the ciphertexts in place and consumes the map -- and that rebuild
            // stays out of the timed region, hence the self-timing sample. The ciphertexts
            // themselves are reused: the cost is value-independent, so re-running over
            // already-selected buffers measures the same work as cloning the table would.
            || match lut_mode {
                LutMode::Full => {
                    let cts_map: HashMap<usize, &mut BackendGLWE<B>> = cts.iter_mut().enumerate().collect();
                    let start: Instant = Instant::now();
                    vpack_eval::<u64, B>(
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
                    (start.elapsed(), ())
                }
                LutMode::Streaming => {
                    let start: Instant = Instant::now();
                    vpack_eval_streaming::<u64, B>(
                        &module,
                        &mut res,
                        n_candidates,
                        &mut gen_ct,
                        first_level,
                        &tree_selector,
                        &rot_selector,
                        log_n,
                        log_table_size,
                        &mut scratch.borrow(),
                    );
                    (start.elapsed(), ())
                }
            },
        );
    }
}
