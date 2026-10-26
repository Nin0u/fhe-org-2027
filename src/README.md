# Implementation of OHVPack and Vertical Packing using Poulpy

Two LUT-evaluation constructions: `ohvpack` (paper §5) and `vpack` (§2.3.7), the TFHE
baseline it is compared against.

OHVPack is OHCollapse (`src/ohvpack.rs::ohcollapse`) followed by a binary CMux tree over the
`alpha` accumulators it leaves. Neither construction packs, so no automorphism or
keyswitching key is generated or used anywhere: the only evaluation keys are the selectors.

## Requirements

Poulpy needs an unstable rustc feature, so this crate only builds on nightly. The version
is pinned in `rust-toolchain.toml` and rustup picks it up automatically; a plain
`cargo build` on stable fails inside `poulpy-hal`.

```bash
rustup toolchain install nightly-2026-05-14
```

## Tests

```bash
cargo test --test         # all three
cargo test --test ohvpack # just one
```
Test names: `selector`, `ohvpack`, `vpack`.

The integration tests run against `init_toy()`, whose torus precision is deliberately
generous. They do **not** tell you whether a construction survives the parameter sets
`paramux` emits — for that see `examples/ohvpack_check.rs`.

## Benchmarks

Every benchmark takes its parameters from the `PARAMS_FILE` environment variable, which must be a path to a parameters JSON produced by `paramux`. The file may hold several parameter sets; each is benchmarked in turn.

```bash
PARAMS_FILE=../paramux/example_finder/ohvpack/depth_2/s1.json cargo bench --bench ohvpack
PARAMS_FILE=../paramux/example_finder/vertical_packing/depth_2/s1.json cargo bench --bench vpack
```

Bench names: `ohvpack`, `vpack`. `ohvpack` needs a file that carries the collapsing factor
`alpha`, i.e. one from `example_finder/ohvpack/`. `runall.sh` runs every shipped parameter
set through both, in all four modes. Two environment variables tune a run:

| variable | default | effect |
|---|---|---|
| `BENCH_SAMPLES` | `10` | evaluations timed per parameter set; set to `1` for a smoke run on an expensive set |
| `LUT_MODE` | `streaming` | `streaming` encodes and DFTs each LUT polynomial inside the evaluation; `full` prepares the whole table up front, outside the timed region, and keeps it resident. For `vpack` it also picks which half of the first tree level's transform credit the run takes -- see `src/vertical_packing.rs::FirstLevel` |

### Backends

Cargo features select the backend, and the vectorised ones **need `RUSTFLAGS`** -- without it
`poulpy-cpu-avx` fails to build rather than falling back, so it is easy to end up measuring
the reference backend by accident:

```bash
cargo bench --bench ohvpack                                                  # reference
RUSTFLAGS="-C target-cpu=native" cargo bench --bench ohvpack --features avx     # AVX2 + FMA
RUSTFLAGS="-C target-cpu=native" cargo bench --bench ohvpack --features avx512  # AVX-512
```

The gap is large between backends so quote it alongside any timing.

### Four mutually exclusive modes

| mode | command | reports |
|---|---|---|
| timing (default) | `cargo bench --bench ohvpack` | Wall clock, 10 fixed samples |
| memory | `cargo bench --bench ohvpack --features mem-profile` | peak heap of one evaluation, per section |
| timing breakdown | `cargo bench --bench ohvpack --features time-profile` | per-operation share of one evaluation |
| noise | `cargo bench --bench ohvpack --features noise-profile` | output noise variance after each layer |

`mem-profile`, `time-profile` and `noise-profile` each replace the timing harness with a single instrumented run.

`time-profile` exists because the default harness can only time an evaluation as a whole -- it has no way to attribute time to the subfunctions inside one, which is what you need to answer "how much of this is external products?". It prints, e.g.:

```text
[time-profile] ohvpack N=1024 alpha=32 levels=5 lut=streaming  total = 11.30 ms
  operation                          time    share      calls
  mul_prepared                     7.39ms    65.4%       1024
  prepare_plain (LUT DFT)          1.38ms    12.2%       1024
  cmux_assign                      0.89ms     7.8%         31
  glwe_add_assign                  0.43ms     3.8%       1024
  lwe_sample_extract               0.00ms     0.0%          1
  other                            1.21ms    10.7%          -
```

(`ohvpack/depth_2/s1.json`, AVX2. The first stage -- `mul_prepared`, `prepare_plain` and `glwe_add_assign`, one call per LUT polynomial -- dominates; the `alpha - 1 = 31` CMux gates of the tree are under 8%.)

`other` is the wall-clock outside every span (allocation, encoding). Instrumentation is compiled out entirely without the feature, so the default timing numbers are unaffected.

To add a span, wrap the innermost operation:

```rust
{
    time_span!("glwe_external_product");
    module.glwe_external_product(&mut res, &a, ggsw, scratch);
}
```

Span the innermost operations only: the report derives its `other` row as `total - sum(spans)`, so nesting one span inside another double-counts.

### Noise profile

`noise-profile` is the measured counterpart of the noise bounds in the paper: it decrypts a sample of each layer's output, subtracts the nearest `p`-ary grid point and reports the variance of what is left.

```text
[noise-profile] ohvpack N=1024 alpha=32 levels=5 lut=streaming  decryption radius = 2^-5 (p = 2^4)
  layer         log2(std)      log2(var)     growth     measured/total
  0 (sel)          -24.56         -49.13          -              16/32
  1                -12.91         -25.82     +23.30              16/32
  2                -10.28         -20.57      +5.26              16/16
  3                 -9.79         -19.59      +0.98                8/8
  4                 -9.52         -19.04      +0.55                4/4
  5                 -9.37         -18.75      +0.29                2/2
  6                 -9.23         -18.45      +0.29                1/1
```

Layer `0 (sel)` is the input: the fresh noise of the one-hot TRLWE selectors, reported as their pooled RMS rather than the worst draw, so `growth` on row 1 is what the first stage actually costs. Rows 2 onwards are the five CMux levels, each adding one `E_ext = 2^-20.6`: `2^-25.8 + 5 E_ext = 2^-18.4`, which is `(nu/alpha) ||P||^2 V_TRLWE + log2(alpha) E_ext` as §5.2 states it.

`measured/total` is how many of the layer's ciphertexts were sampled; a layer can hold a million of them, so the default cap is 16 (raise it with `NOISE_SAMPLES`). Read `log2(var)` against the paper's bounds and `log2(std)` against the decryption radius printed in the header -- once the two meet, the residuals wrap to the wrong grid point and every figure becomes a lower bound. See `src/noise_profile.rs` for the measurement's assumptions and for how a benchmark installs the probes that do the decryption.

Two things to hold on to when reading a report:

- The bound the parameter sets are built to is a **decryption-failure probability** of `2^-30`, not a variance (`ERFC_THRESHOLD` in `paramux/param_finder.py`). At `p = 16` that is `erfc(1 / (2 sqrt(2 V) p)) < 2^-30`, i.e. an output variance budget of **`2^-15.2`**. Compare the last row's `log2(var)` against that number, not against `2^-30`.
- Each row is the worst of `NOISE_SAMPLES` ciphertexts under a fresh seed per run, so a row moves by a bit or two between runs. Read differences of 4+ bits, not of 1.
