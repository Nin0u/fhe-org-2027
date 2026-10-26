//! Per-layer output noise breakdown, gated behind the `noise-profile` feature.
//!
//! `time-profile` answers "where does the wall-clock go?" and `mem-profile` "which
//! function drives the peak heap?". This module answers the third question the paper's
//! analyses raise: **how does the output noise variance actually grow, layer by layer?**
//! The bounds of Section 5.2 predict one additive `E_ext` per level of the CMux tree, on
//! top of whatever the first stage costs; a run under this feature prints the measured variance after each layer so the prediction can be checked
//! against the implementation.
//!
//! ```text
//! cargo bench --bench ohvpack --features noise-profile
//! ```
//!
//! The last layer is the evaluation's output. It is recorded on the final GLWE, before the
//! closing sample-extract, and not on the extracted LWE the evaluation returns. The two
//! carry the same noise -- the extract copies coefficient 0 -- but the GLWE gives `N`
//! residuals to average where the LWE gives one, and a single residual is worth about an
//! order of magnitude. Reading a one-observation row as if it were a measurement is how a
//! run-to-run swing of a bit gets mistaken for a regression.
//!
//! Layer `0` is the *input*: the fresh encryption noise of the TRLWE selectors the
//! construction is handed. Without it the first recorded layer has nothing to be compared
//! against and the growth the bounds predict for the first level cannot be read off the
//! table -- only the levels after it. The benchmarks record it right after installing the
//! probes (see `benchmarks/bench_ohvpack.rs`); it is printed as `0 (sel)` and, unlike the
//! evaluation layers, reports the pooled RMS over the sampled selectors rather than the
//! worst one, a one-hot vector being almost entirely zero-encryptions whose spread carries
//! no information.
//!
//! # How the noise is measured
//!
//! No expected plaintext is needed. Every ciphertext the constructions produce carries
//! messages on the `p`-ary grid -- the layers only rotate, permute and negate LUT entries,
//! never move them off it -- so decrypting and subtracting the *nearest* grid point leaves
//! exactly the noise. The variance reported for a layer is the variance of those residuals,
//! in torus units.
//!
//! The corollary is that the measurement is only meaningful while the noise stays inside
//! the decryption radius `1/(2p)`. Past that point a coefficient rounds to the wrong grid
//! point and its residual is *under*-reported, so a layer whose parameters have already
//! failed reads as quieter than it is, not noisier. Compare the printed
//! `log2(std)` against `-log2(2p)`: a value approaching it means the measurement, and the
//! parameter set, are both at their limit.
//!
//! # Installing a probe
//!
//! The evaluation functions are generic over the backend and hold no key material, so they
//! cannot decrypt anything themselves. Instead a benchmark installs two closures that can
//! ([`install`]), and the [`noise_span!`] macro calls them. Because the ciphertext types
//! are `GLWE<Vec<u8>>` and `LWE<Vec<u8>>` -- concrete, not generic, since every evaluation
//! function is bounded on `Backend<OwnedBuf = Vec<u8>>` -- the closures need no type
//! parameter, and the module needs no unsafe lifetime tricks. They must be `'static`, so a
//! benchmark gives them their own `Module`, prepared secret and scratch rather than
//! borrowing the ones the evaluation uses; see `benchmarks/bench_ohvpack.rs`.
//!
//! # Sampling
//!
//! A wide layer holds far too many ciphertexts to decrypt them all -- the first CMux level
//! of VPack at `N = 1024, d = 3` produces `2^19` GLWEs, and one decryption plus arbitrary-precision
//! decode costs far more than the layer that produced it. Each recorded layer therefore
//! measures at most [`samples`] ciphertexts, evenly spaced across the layer. Raise it with
//! `NOISE_SAMPLES` when a layer's spread matters more than the run time.
//!
//! A GLWE carries `N` residuals of its own, so its probe returns their RMS and a layer
//! reports the worst ciphertext it sampled -- the quantity the bounds track. An LWE
//! carries one, so its probe returns that single residual and the layer's figure is the
//! RMS across the ciphertexts sampled. Both estimate the same variance, but the LWE one
//! rests on [`samples`] observations rather than `N`, so it is the coarser of the two --
//! and on a layer that holds a single LWE it is one observation, which is an order of
//! magnitude at best. RMS rather than a mean-subtracted standard deviation throughout:
//! the noise is zero-mean by construction, so there is nothing to centre.

#[cfg(feature = "noise-profile")]
use std::{cell::RefCell, rc::Rc};

#[cfg(feature = "noise-profile")]
use poulpy_core::{
    GLWEDecrypt, LWEDecrypt,
    layouts::{
        GLWESecretPrepared, LWEInfos, ModuleCoreAlloc,
        BackendGLWE, BackendLWE, BackendGLWEPlaintext, BackendLWEPlaintext, BackendLWESecret,
    },
};
#[cfg(feature = "noise-profile")]
use poulpy_hal::{
    api::ScratchOwnedBorrow, layouts::{Backend, Module, ScratchOwned},
};


/// Records the output noise of one layer against a layer number.
///
/// Three forms, matching what a layer leaves behind:
///
/// ```ignore
/// noise_span!(glwe: 1, &result);      // a single GLWE
/// noise_span!(glwe_max: 1, &outputs); // a slice of GLWEs, worst sampled one reported
/// noise_span!(glwe_avg: 0, &selectors); // a slice of GLWEs, pooled RMS reported
/// noise_span!(lwe_max: 2, &lwes);     // a slice of LWEs, RMS over the samples reported
/// ```
///
/// Expands to nothing without the `noise-profile` feature.
#[macro_export]
macro_rules! noise_span {
    (glwe: $layer:expr, $ct:expr) => {
        $crate::noise_profile::record_glwe($layer, std::slice::from_ref($ct));
    };
    (glwe_max: $layer:expr, $cts:expr) => {
        $crate::noise_profile::record_glwe($layer, $cts);
    };
    (glwe_avg: $layer:expr, $cts:expr) => {
        $crate::noise_profile::record_glwe_avg($layer, $cts);
    };
    (lwe: $layer:expr, $ct:expr) => {
        $crate::noise_profile::record_lwe($layer, std::slice::from_ref($ct));
    };
    (lwe_max: $layer:expr, $cts:expr) => {
        $crate::noise_profile::record_lwe($layer, $cts);
    };
}

#[cfg(feature = "noise-profile")]
mod imp {
    use std::cell::RefCell;

    use poulpy_core::layouts::{GLWE, LWE};

    /// Ciphertexts decrypted per recorded layer, unless `NOISE_SAMPLES` overrides it.
    ///
    /// Small on purpose: the cost is a decryption plus an arbitrary-precision decode per
    /// sample, against layers that can hold a million ciphertexts.
    pub const SAMPLES: usize = 16;

    /// The number of ciphertexts to sample per layer: [`SAMPLES`], unless `NOISE_SAMPLES`
    /// overrides it.
    pub fn samples() -> usize {
        match std::env::var("NOISE_SAMPLES") {
            Ok(v) => v.parse::<usize>().ok().filter(|n| *n > 0).unwrap_or(SAMPLES),
            Err(_) => SAMPLES,
        }
    }

    /// Measures a ciphertext's noise, in torus units: the RMS of a GLWE's `N` residuals,
    /// or an LWE's single residual.
    ///
    /// Installed by the benchmark, which owns the key material; see [`install`].
    type GlweProbe = Box<dyn Fn(&GLWE<Vec<u8>, i64>) -> f64>;
    type LweProbe = Box<dyn Fn(&LWE<Vec<u8>, i64>) -> f64>;

    /// One recorded layer: its noise standard deviation, how many ciphertexts were
    /// sampled, and how many the layer actually held.
    struct Row {
        layer: usize,
        std: f64,
        measured: usize,
        total: usize,
    }

    thread_local! {
        static PROBES: RefCell<Option<(GlweProbe, LweProbe)>> = const { RefCell::new(None) };
        static TABLE: RefCell<Vec<Row>> = const { RefCell::new(Vec::new()) };
        /// Set while a probe runs, so the decryptions a probe performs cannot recurse into
        /// the recorder if the probe itself ever passes through instrumented code.
        static IN_PROBE: RefCell<bool> = const { RefCell::new(false) };
    }

    /// Installs the two measurement closures. Replaces any previous pair.
    pub fn install(glwe: GlweProbe, lwe: LweProbe) {
        PROBES.with(|p| *p.borrow_mut() = Some((glwe, lwe)));
    }

    /// Indices of the ciphertexts to measure: `n` evenly spaced positions in `0..len`.
    fn sample_indices(len: usize) -> Vec<usize> {
        let n: usize = samples().min(len);
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![0];
        }
        (0..n).map(|i| i * (len - 1) / (n - 1)).collect()
    }

    /// Merges a layer's measurement into the table, keeping the worst seen.
    fn push(layer: usize, std: f64, measured: usize, total: usize) {
        TABLE.with(|c| {
            let mut t = c.borrow_mut();
            if let Some(r) = t.iter_mut().find(|r| r.layer == layer) {
                if std > r.std {
                    r.std = std;
                }
                r.measured += measured;
                r.total += total;
            } else {
                t.push(Row { layer, std, measured, total });
            }
        });
    }

    /// Runs `f` with the installed probes, guarding against re-entrancy.
    fn with_probes<F: FnOnce(&(GlweProbe, LweProbe))>(f: F) {
        if IN_PROBE.with(|g| *g.borrow()) {
            return;
        }
        IN_PROBE.with(|g| *g.borrow_mut() = true);
        PROBES.with(|p| {
            if let Some(probes) = p.borrow().as_ref() {
                f(probes);
            }
        });
        IN_PROBE.with(|g| *g.borrow_mut() = false);
    }

    /// Records a layer of GLWEs: each probe already averages over the ciphertext's `N`
    /// residuals, so the layer reports the worst of the sampled ciphertexts.
    pub fn record_glwe(layer: usize, cts: &[GLWE<Vec<u8>, i64>]) {
        if cts.is_empty() {
            return;
        }
        with_probes(|(glwe, _)| {
            let idx: Vec<usize> = sample_indices(cts.len());
            let worst: f64 = idx.iter().map(|&i| glwe(&cts[i])).fold(0.0f64, f64::max);
            push(layer, worst, idx.len(), cts.len());
        });
    }

    /// Records a layer of GLWEs by their *pooled* RMS rather than the worst one.
    ///
    /// This is what layer 0 wants. The TRLWE selectors are a one-hot vector, so all but one
    /// of them encrypt zero and every one of them is a fresh encryption: they are draws from
    /// a single distribution, and the figure that belongs next to the bounds is its standard
    /// deviation, not the largest draw. Taking the worst instead would report the tail of
    /// `G` samples and overstate the input the first layer builds on.
    ///
    /// Each probe already returns the RMS of one ciphertext's `N` residuals, so the RMS of
    /// those is the RMS over all sampled coefficients.
    pub fn record_glwe_avg(layer: usize, cts: &[GLWE<Vec<u8>, i64>]) {
        if cts.is_empty() {
            return;
        }
        with_probes(|(glwe, _)| {
            let idx: Vec<usize> = sample_indices(cts.len());
            let sum_sq: f64 = idx.iter().map(|&i| glwe(&cts[i])).map(|r| r * r).sum();
            let rms: f64 = (sum_sq / idx.len() as f64).sqrt();
            push(layer, rms, idx.len(), cts.len());
        });
    }

    /// Records a layer of LWEs: each probe returns one residual, so the layer's figure is
    /// their RMS -- the sampled ciphertexts *are* the observations here.
    pub fn record_lwe(layer: usize, cts: &[LWE<Vec<u8>, i64>]) {
        if cts.is_empty() {
            return;
        }
        with_probes(|(_, lwe)| {
            let idx: Vec<usize> = sample_indices(cts.len());
            let sum_sq: f64 = idx.iter().map(|&i| lwe(&cts[i])).map(|r| r * r).sum();
            let rms: f64 = (sum_sq / idx.len() as f64).sqrt();
            push(layer, rms, idx.len(), cts.len());
        });
    }

    /// Drops every recorded layer. Call before the run you want to measure; the installed
    /// probes are kept.
    pub fn reset() {
        TABLE.with(|c| c.borrow_mut().clear());
    }

    /// Prints the per-layer noise in layer order.
    ///
    /// `log2(var)` is the figure to compare against the paper's bounds, which are stated
    /// as variances; `log2(std)` is the one to compare against the decryption radius
    /// `1/(2p)`, printed alongside as `budget` so the margin is readable directly.
    ///
    /// Layer `0`, when recorded, is the fresh noise of the TRLWE selectors and prints as
    /// `0 (sel)`. The `growth` column gives each layer's variance increase over the row
    /// above it, so the first layer's cost is readable directly rather than having to be
    /// subtracted by hand.
    pub fn report(header: &str, log_p: usize) {
        let mut rows: Vec<(usize, f64, usize, usize)> =
            TABLE.with(|c| c.borrow().iter().map(|r| (r.layer, r.std, r.measured, r.total)).collect());
        rows.sort_by_key(|r| r.0);

        // Full torus: a coefficient decodes correctly while |noise| < 1/(2p).
        let budget: f64 = -((log_p + 1) as f64);

        println!("\n[noise-profile] {header}  decryption radius = 2^{budget} (p = 2^{log_p})");
        println!(
            "  {:<8} {:>14} {:>14} {:>10} {:>18}",
            "layer", "log2(std)", "log2(var)", "growth", "measured/total"
        );
        let mut prev: Option<f64> = None;
        for (layer, std, measured, total) in &rows {
            let log2_std: f64 = if *std > 0.0 { std.log2() } else { f64::NEG_INFINITY };
            let log2_var: f64 = 2.0 * log2_std;
            // Variance added since the row above, in bits. The first row has nothing to
            // grow from; layer 0 is the selectors' fresh noise, so row 1's figure is
            // exactly what the first level of the tree costs.
            let growth: String = match prev {
                Some(p) if log2_var.is_finite() && p.is_finite() => format!("{:+.2}", log2_var - p),
                _ => "-".to_string(),
            };
            let name: String = if *layer == 0 { "0 (sel)".to_string() } else { layer.to_string() };
            println!(
                "  {:<8} {:>14.2} {:>14.2} {:>10} {:>18}",
                name,
                log2_std,
                log2_var,
                growth,
                format!("{measured}/{total}")
            );
            prev = Some(log2_var);
        }
        if let Some((layer, std, _, _)) = rows.last()
            && *std > 0.0
            && std.log2() >= budget
        {
            println!(
                "  WARNING: layer {layer} is at or past the decryption radius -- the \
                 residuals wrap to the wrong grid point and every figure above is a \
                 lower bound."
            );
        }
    }
}

#[cfg(not(feature = "noise-profile"))]
mod imp {
    use poulpy_core::layouts::{GLWE, LWE};

    #[inline(always)]
    pub fn record_glwe(_layer: usize, _cts: &[GLWE<Vec<u8>, i64>]) {}

    #[inline(always)]
    pub fn record_glwe_avg(_layer: usize, _cts: &[GLWE<Vec<u8>, i64>]) {}

    #[inline(always)]
    pub fn record_lwe(_layer: usize, _cts: &[LWE<Vec<u8>, i64>]) {}

    #[inline(always)]
    pub fn reset() {}

    #[inline(always)]
    pub fn report(_header: &str, _log_p: usize) {}
}

#[cfg(feature = "noise-profile")]
pub use imp::{SAMPLES, install, samples};
pub use imp::{record_glwe, record_glwe_avg, record_lwe, report, reset};

/// Builds the two probes from key material the caller hands over, and installs them.
///
/// The probes must outlive the evaluation and be `'static`, so everything they touch is
/// *moved* in: give them a `Module`, a prepared GLWE secret, an LWE secret and a scratch
/// arena of their own rather than the ones the evaluation uses. Building a second copy of
/// each is cheap next to what a profiling run costs, and it keeps the borrow checker out
/// of the picture entirely.
///
/// Call it once per parameter set, before [`reset`] and the evaluation.
#[cfg(feature = "noise-profile")]
pub fn install_owned<BE>(
    module: Module<BE>,
    sk_glwe_dft: GLWESecretPrepared<Vec<u8>, BE>,
    sk_lwe: BackendLWESecret<BE>,
    scratch: ScratchOwned<BE>,
    n: usize,
    log_p: usize,
) where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + 'static,
    Module<BE>: GLWEDecrypt<BE> + LWEDecrypt<BE> + ModuleCoreAlloc<OwnedBuf = Vec<u8>, ZnxWord = i64>,
{

    // One cell for everything the probes mutate, so both closures can reach the scratch
    // arena without either owning it exclusively.
    let cell: Rc<RefCell<(Module<BE>, GLWESecretPrepared<Vec<u8>, BE>, BackendLWESecret<BE>, ScratchOwned<BE>)>> =
        Rc::new(RefCell::new((module, sk_glwe_dft, sk_lwe, scratch)));

    let glwe_cell = Rc::clone(&cell);
    let glwe_probe: Box<dyn Fn(&BackendGLWE<BE>) -> f64> = Box::new(move |ct: &BackendGLWE<BE>| {
        let mut guard = glwe_cell.borrow_mut();
        let (module, sk, _, scratch) = &mut *guard;

        // Decode at the ciphertext's *stored* precision, `size * base2k`, not at its
        // nominal `k`. Several layers hand on ciphertexts whose `k` is only `log_p` -- the
        // noise then lives entirely in the limb bits below `k`, and decoding at `k` would
        // round it away and report a clean zero.
        let base2k: usize = ct.base2k().into();
        let k_full: usize = base2k * ct.size();

        let mut pt: BackendGLWEPlaintext<BE> = module.glwe_plaintext_alloc(base2k.into(), k_full.into());
        module.glwe_decrypt(ct, &mut pt, sk, &mut scratch.borrow());

        let mut full: Vec<i64> = vec![0; n];
        pt.decode_vec_i64(&mut full, k_full.into());
        let mut messages: Vec<i64> = vec![0; n];
        pt.decode_vec_i64(&mut messages, log_p.into());

        let scale: f64 = (-(k_full as f64)).exp2();
        let decoded: Vec<f64> = full.iter().map(|&x| x as f64 * scale).collect();
        residual_rms(&decoded, &messages, log_p)
    });

    let lwe_cell = Rc::clone(&cell);
    let lwe_probe: Box<dyn Fn(&BackendLWE<BE>) -> f64> = Box::new(move |ct: &BackendLWE<BE>| {
        let mut guard = lwe_cell.borrow_mut();
        let (module, _, sk, scratch) = &mut *guard;

        let base2k: usize = ct.base2k().into();
        let k_full: usize = base2k * ct.size();

        let mut pt: BackendLWEPlaintext<BE> = module.lwe_plaintext_alloc(base2k.into(), k_full.into());
        module.lwe_decrypt(ct, &mut pt, sk, &mut scratch.borrow());

        let scale: f64 = (-(k_full as f64)).exp2();
        let decoded: [f64; 1] = [pt.decode_i64(k_full.into()) as f64 * scale];
        let messages: [i64; 1] = [pt.decode_i64(log_p.into())];
        residual_rms(&decoded, &messages, log_p)
    });

    install(glwe_probe, lwe_probe);
}

/// Residual noise of a decoded plaintext, in torus units.
///
/// `decoded` holds the plaintext's coefficients as torus values and `messages` the same
/// coefficients rounded to the `p`-ary grid (i.e. decoded at `log_p` bits of precision).
/// The difference is the noise, and this returns its RMS -- not mean-subtracted, the
/// noise being zero-mean by construction.
///
/// Lives outside the `cfg` blocks so a benchmark can build its probes with it without
/// duplicating the arithmetic, and so it is type-checked in every build.
pub fn residual_rms(decoded: &[f64], messages: &[i64], log_p: usize) -> f64 {
    debug_assert_eq!(decoded.len(), messages.len());
    if decoded.is_empty() {
        return 0.0;
    }
    let scale: f64 = 1.0 / (1u64 << log_p) as f64;
    let sum_sq: f64 = decoded
        .iter()
        .zip(messages)
        .map(|(x, &m)| {
            let r: f64 = x - (m as f64) * scale;
            r * r
        })
        .sum();
    (sum_sq / decoded.len() as f64).sqrt()
}

/// Builds the probes' own module, secret and scratch from the evaluation's key material and
/// installs them.
///
/// A convenience over [`install_owned`], which needs everything moved in: the probes are
/// `'static`, so they cannot borrow the evaluation's module or scratch and get a second copy
/// of each instead. That copy is what every benchmark used to spell out for itself.
#[cfg(feature = "noise-profile")]
pub fn install_probe<BE>(
    n: usize,
    log_p: usize,
    layouts: &crate::context::MuxLayouts,
    sk_glwe: &poulpy_core::layouts::BackendGLWESecret<BE>,
) where
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + poulpy_hal::oep::HalModuleImpl<BE> + 'static,
    Module<BE>: GLWEDecrypt<BE>
        + LWEDecrypt<BE>
        + ModuleCoreAlloc<OwnedBuf = Vec<u8>, ZnxWord = i64>
        + poulpy_core::layouts::GLWESecretPreparedFactory<BE>
        + poulpy_core::layouts::SecretConversion<BE>,
{
    use poulpy_core::layouts::{GLWESecretPreparedFactory, SecretConversion};
    use poulpy_hal::api::ScratchOwnedAlloc;

    let module: Module<BE> = Module::<BE>::new(n as u64);
    let mut sk_glwe_dft: GLWESecretPrepared<Vec<u8>, BE> =
        module.glwe_secret_prepared_alloc_from_infos(&layouts.glwe);
    module.glwe_secret_prepare(&mut sk_glwe_dft, sk_glwe);
    let sk_lwe: BackendLWESecret<BE> = module.lwe_secret_from_glwe_secret(sk_glwe, n.into());
    let scratch: ScratchOwned<BE> = ScratchOwned::<BE>::alloc(
        module
            .glwe_decrypt_tmp_bytes(&layouts.glwe)
            .max(module.lwe_decrypt_tmp_bytes(&layouts.lwe)),
    );
    install_owned(module, sk_glwe_dft, sk_lwe, scratch, n, log_p);
}
