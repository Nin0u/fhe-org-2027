//! OHVPack benchmark.
//!
//! Four mutually exclusive modes, selected by cargo feature:
//!   cargo bench --bench ohvpack                          # wall clock, 10 fixed samples
//!   cargo bench --bench ohvpack --features time-profile  # per-operation timing breakdown
//!   cargo bench --bench ohvpack --features mem-profile   # per-section peak heap
//!   cargo bench --bench ohvpack --features noise-profile # per-layer output noise variance
//!
//! Orthogonal to those, `LUT_MODE` picks how the LUT is held (see `ohvpack::lut`):
//!   LUT_MODE=streaming (default)  # each polynomial is encoded + DFT-ed inside the evaluation
//!   LUT_MODE=full                 # the whole table is prepared up front, outside the
//!                                 # timed region, and stays resident
//!
//! Takes a `PARAMS_FILE` from `paramux/example_finder/ohvpack/`, which carries the
//! collapsing factor `alpha` alongside the TFHE parameters. Compare against the vpack
//! benchmark at the matching file in `paramux/example_finder/vertical_packing/`.

use ohvpack::bench_timing::run_modes;
use ohvpack::context::{MuxLayouts, SecretKeys, secret_keys};
use ohvpack::lut::{Lut, LutMode, PreparedLut, build_lut};
use ohvpack::mem_profile::Span;
use ohvpack::ohvpack::{ohvpack_eval, ohvpack_tree_depth};
use ohvpack::params::{MuxParams, params_from_json};
use ohvpack::prepare_data::prepare_glwe;
use ohvpack::selector::ohvpack_make_selectors;
use ohvpack::vertical_packing::vpack_eval_tmp_bytes;

use poulpy_core::layouts::GGSWPreparedFactory;
use poulpy_core::{GGSWEncryptSk, GLWEEncryptSk};
use poulpy_hal::{
    api::{ScratchOwnedAlloc, ScratchOwnedBorrow},
    layouts::{Backend, CnvPVecL, Module, ScratchOwned},
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
/// `i64` because the index spans `log2(L)` bits -- 33 for a 2^33 table -- and because that
/// is what every `make_selectors` in `ohvpack::selector` takes.
const SELECTOR: i64 = 2048;

fn main() {
    // Parameters file: taken from the `PARAMS_FILE` environment variable so it can be
    // supplied on the benchmark command line, e.g.
    //   PARAMS_FILE=../paramux/example_finder/ohvpack/depth_2/s1.json cargo bench --bench ohvpack
    let params_path: String = std::env::var("PARAMS_FILE")
        .expect("PARAMS_FILE environment variable must be set to the parameters JSON path");
    let param_set: Vec<MuxParams> = params_from_json(&params_path);
    let lut_mode: LutMode = LutMode::from_env();

    for mut params in param_set {
        ohvpack::mem_profile::reset();

        // Read-only input, streamed polynomial-by-polynomial during the evaluation.
        let lut: Vec<i32> = {
            let _lut_span = Span::new("lut");
            build_lut(params.n_rlwe.pow(params.depth as u32), params.p)
        };

        let setup_span = Span::new("setup");

        // `alpha` comes from the parameters file. §5.3 adds one constraint: the tree is
        // binary, so alpha must be a power of two. §4.6's choice already satisfies it.
        assert!(
            params.alpha > 0,
            "the parameters file must carry `alpha` for OHVPack (see paramux)"
        );
        assert!(
            params.alpha.is_power_of_two(),
            "OHVPack's CMux tree is binary, so alpha must be a power of two, got {}",
            params.alpha
        );

        let layouts: MuxLayouts = MuxLayouts::new(&params);
        let module: Module<B> = Module::<B>::new(params.n_rlwe as u64);

        // No `glwe_pack` term and no automorphism keys: OHVPack's second stage is CMux
        // gates, so the packing machinery is absent from the evaluation entirely (§5.1).
        let mut scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(
            module
                .glwe_encrypt_sk_tmp_bytes(&layouts.glwe)
                .max(module.ggsw_encrypt_sk_tmp_bytes(&layouts.ggsw))
                .max(module.ggsw_prepare_tmp_bytes(&layouts.ggsw))
                .max(vpack_eval_tmp_bytes::<B, _, _>(&module, &layouts.glwe, &layouts.ggsw)),
        );

        let sk: SecretKeys<B> = secret_keys(&module, &mut params, &layouts);

        let (glwe_selectors, tree_selectors) = ohvpack_make_selectors::<B>(
            &module,
            &mut params,
            &layouts.glwe,
            &layouts.ggsw,
            &sk.glwe_dft,
            SELECTOR,
            &mut scratch.borrow(),
        );

        // Precomputation: the one-hot selectors are multiplied by every LUT polynomial, so
        // their DFT is hoisted out of the timed region.
        let glwe_selectors_prep: Vec<CnvPVecL<Vec<u8>, <B as Backend>::DftWord, B>> = glwe_selectors
            .iter()
            .map(|sel| prepare_glwe(&module, sel, &mut scratch.borrow()))
            .collect();

        let prepared_lut: Option<PreparedLut<B>> = match lut_mode {
            LutMode::Streaming => None,
            LutMode::Full => {
                let _prep_span = Span::new("lut prep (full)");
                Some(PreparedLut::for_tree(&module, &params, &lut, &mut scratch.borrow()))
            }
        };
        let lut_arg: Lut<B> = match &prepared_lut {
            Some(prep) => Lut::Full(prep),
            None => Lut::Streaming(&lut),
        };

        let label: String = format!(
            "ohvpack N={} alpha={} levels={} lut={}",
            params.n_rlwe,
            params.alpha,
            ohvpack_tree_depth(params.alpha),
            lut_mode.as_str()
        );
        run_modes(
            &label,
            params.log_p,
            setup_span,
            // Noise mode only: install the probes, then record layer 0 -- the fresh noise of
            // the one-hot vector, so the layer-by-layer growth has a baseline to be read
            // against.
            #[allow(unused_variables)]
            || {
                #[cfg(feature = "noise-profile")]
                {
                    ohvpack::noise_profile::install_probe::<B>(params.n_rlwe, params.log_p, &layouts, &sk.glwe);
                    ohvpack::noise_profile::record_glwe_avg(0, &glwe_selectors);
                }
            },
            || {
                ohvpack_eval::<B>(
                    &module,
                    &params,
                    &mut scratch.borrow(),
                    lut_arg,
                    &glwe_selectors_prep,
                    &tree_selectors,
                )
            },
        );
    }
}
