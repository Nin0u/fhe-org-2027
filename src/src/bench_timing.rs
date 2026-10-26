//! Fixed-sample wall-clock harness for the benchmarks, in place of Criterion.
//!
//! Criterion sizes its own workload: a 3 s warm-up, then as many iterations as fit a 5 s
//! target measurement time, repeated over N samples. That is the right design for
//! microbenchmarks and the wrong one here. A single `d = 3` evaluation already runs for
//! minutes, so Criterion spends hours per parameter set chasing a time budget the first
//! iteration blew past -- and the extra iterations buy nothing, because the run-to-run
//! spread of a minutes-long FHE evaluation is a fraction of a percent.
//!
//! This harness runs exactly [`SAMPLES`] evaluations, with no warm-up and no adaptive
//! iteration count, and prints the whole distribution so the noise stays visible rather
//! than being hidden behind a single averaged number.
//!
//! Read the **median** (or the min) rather than the mean: with no warm-up, the first
//! sample carries first-touch page faults on the scratch arena and any lazily-populated
//! table, so it is routinely the slowest of the ten. That cost is real but it is paid once,
//! not per evaluation.

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Evaluations timed per parameter set.
///
/// Fixed rather than derived from a time budget -- see the module docs. Override with the
/// `BENCH_SAMPLES` environment variable for a quick smoke run on an expensive parameter
/// set (`BENCH_SAMPLES=1 cargo bench --bench ohvpack`).
pub const SAMPLES: usize = 10;

/// The sample count to use: [`SAMPLES`], unless `BENCH_SAMPLES` overrides it.
pub fn samples() -> usize {
    match std::env::var("BENCH_SAMPLES") {
        Ok(v) => v.parse::<usize>().ok().filter(|n| *n > 0).unwrap_or(SAMPLES),
        Err(_) => SAMPLES,
    }
}

/// Times `eval` `n` times and prints the distribution under `label`.
///
/// The result of each evaluation is black-boxed so the call cannot be optimised away, and
/// dropped *after* the clock is read so deallocation is not counted.
pub fn bench<T, F: FnMut() -> T>(label: &str, n: usize, mut eval: F) {
    bench_samples(label, n, || {
        let start: Instant = Instant::now();
        let out: T = eval();
        let elapsed: Duration = start.elapsed();
        black_box(out);
        elapsed
    });
}

/// Like [`bench`], but the closure times itself and returns the duration.
///
/// For evaluations that need per-sample setup which must stay out of the timed region --
/// vertical packing rebuilds its candidate map before every call, since blind selection
/// consumes it.
pub fn bench_samples<F: FnMut() -> Duration>(label: &str, n: usize, mut sample: F) {
    let mut times: Vec<Duration> = Vec::with_capacity(n);
    for _ in 0..n {
        times.push(sample());
    }
    report(label, &mut times);
}

/// Prints `min / median / mean / max` plus the raw samples. Sorts `times` in place.
fn report(label: &str, times: &mut [Duration]) {
    if times.is_empty() {
        println!("\n[bench] {label}  no samples");
        return;
    }
    times.sort_unstable();

    let n: usize = times.len();
    let total: Duration = times.iter().sum();
    let mean: Duration = total / n as u32;
    // Even sample counts take the lower of the two middle values rather than averaging
    // them: with n = 10 the difference is noise, and this keeps the figure an observed one.
    let median: Duration = times[(n - 1) / 2];

    println!("\n[bench] {label}  samples = {n}");
    println!(
        "  min {}   median {}   mean {}   max {}",
        fmt(times[0]),
        fmt(median),
        fmt(mean),
        fmt(times[n - 1])
    );
    let all: Vec<String> = times.iter().map(|d| fmt(*d)).collect();
    println!("  sorted: {}", all.join(", "));
}

/// Formats a duration in whichever of s / ms / µs keeps it readable.
fn fmt(d: Duration) -> String {
    let s: f64 = d.as_secs_f64();
    if s >= 1.0 {
        format!("{s:.3} s")
    } else if s >= 1e-3 {
        format!("{:.3} ms", s * 1e3)
    } else {
        format!("{:.3} µs", s * 1e6)
    }
}

/// Runs `eval` in whichever of the four benchmark modes the build selected, and prints
/// that mode's report under `label`.
///
/// The modes are mutually exclusive cargo features, and each wants the evaluation run
/// differently -- ten timed samples, or one instrumented run for the timing, memory or
/// noise breakdown. Every benchmark used to spell all four out around four verbatim copies
/// of its `*_eval` call; they hand the call over as a closure instead.
///
/// See [`run_modes_sampled`] for the arguments, and for the variant to use when a sample
/// needs setup that must stay out of the timed region.
pub fn run_modes<T>(
    label: &str,
    log_p: usize,
    setup_span: crate::mem_profile::Span,
    arm_noise_probe: impl FnMut(),
    mut eval: impl FnMut() -> T,
) {
    run_modes_sampled(label, log_p, setup_span, arm_noise_probe, || {
        let start: Instant = Instant::now();
        let out: T = eval();
        (start.elapsed(), out)
    });
}

/// [`run_modes`] for evaluations whose samples time themselves.
///
/// Vertical packing needs this: blind selection consumes its candidate map, so the map is
/// rebuilt before every sample and that rebuild must not be counted.
///
/// `setup_span` is the `mem-profile` span covering everything built before the evaluation.
/// It is dropped here, right before the measured run, so setup and evaluation are attributed
/// disjointly; without that feature it is a ZST and dropping it does nothing.
///
/// `arm_noise_probe` installs the decryption probes and records whatever layer-0 noise the
/// construction starts from. It runs only in `noise-profile` mode, after
/// [`crate::noise_profile::reset`], and is otherwise never called.
pub fn run_modes_sampled<T>(
    label: &str,
    #[allow(unused_variables)] log_p: usize,
    #[allow(unused_variables)] setup_span: crate::mem_profile::Span,
    #[allow(unused_mut, unused_variables)] mut arm_noise_probe: impl FnMut(),
    #[allow(unused_mut)] mut sample: impl FnMut() -> (Duration, T),
) {
    // Ten timed evaluations, no warm-up and no adaptive iteration count -- see the module docs.
    #[cfg(not(any(feature = "mem-profile", feature = "time-profile", feature = "noise-profile")))]
    bench_samples(label, samples(), || {
        let (elapsed, out) = sample();
        black_box(out);
        elapsed
    });

    // One instrumented run, printing how the wall-clock splits across the operations the
    // evaluation calls -- which the plain timing mode cannot do, timing only the whole.
    #[cfg(feature = "time-profile")]
    {
        crate::time_profile::reset();
        let start: Instant = Instant::now();
        black_box(sample());
        crate::time_profile::report(label, start.elapsed());
    }

    // One run with the allocator instrumented, reporting peak heap per section.
    #[cfg(feature = "mem-profile")]
    {
        drop(setup_span);
        black_box(sample());
        crate::mem_profile::report(label);
    }

    // One run with decryption probes installed, reporting the noise each layer leaves.
    #[cfg(feature = "noise-profile")]
    {
        crate::noise_profile::reset();
        arm_noise_probe();
        black_box(sample());
        crate::noise_profile::report(label, log_p);
    }
}
