// This file treats where the LUT lives while an evaluation runs:
//      - streamed i.e. out of the plaintext table,
//      - prepared in full before the evaluation starts.
//
//  Every evaluator here multiplies one GLWE selector by many LUT polynomials, so each
//  polynomial has to be encoded and DFT-ed at some point. There are two places to pay for
//  that, and they trade time against memory:
// 
//   - [`LutMode::Streaming`] -- encode and DFT polynomial `m` inside the evaluation, into a
//     single reused buffer. One polynomial is resident at a time, and every evaluation pays
//     `L/N` DFTs.
//   - [`LutMode::Full`] -- encode and DFT the whole table up front into a [`PreparedLut`],
//     once, and let the evaluation read prepared operands straight out of it. The DFTs
//     leave the timed region entirely, at the cost of holding `L/N` prepared polynomials
//     resident -- the term that scales with the table, which is why the evaluators stream
//     by default (see [`crate::prepare_data`]).
// 
//  The two produce bit-identical results: [`LutFeed`] is the single point where the choice
//  is made, and both branches encode the same coefficients at the same scale. The
//  benchmarks pick between them with [`LutMode::from_env`] so one build can measure both.

use poulpy_core::layouts::{BackendGLWEPlaintext, GLWEPlaintext, LWEInfos, ModuleCoreAlloc, TorusPrecision};
use poulpy_hal::{
    api::{CnvPVecAlloc, Convolution, ModuleN},
    layouts::{Backend, CnvPVecR, Module, ScratchArena},
};

use crate::params::MuxParams;
use crate::prepare_data::{alloc_plain_prep, prepare_plain_into};
use crate::time_span;

/// Which of the two LUT residencies an evaluation runs against.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LutMode {
    /// Encode + DFT each polynomial inside the evaluation; one is resident at a time.
    Streaming,
    /// Encode + DFT the whole table before the evaluation; all of it stays resident.
    Full,
}

impl LutMode {
    /// The mode named by the `LUT_MODE` environment variable, defaulting to
    /// [`LutMode::Streaming`].
    ///
    /// The benchmarks read this rather than a cargo feature so a single build can measure
    /// both residencies:
    ///
    /// ```text
    /// LUT_MODE=full PARAMS_FILE=... cargo bench --bench ohvpack
    /// ```
    ///
    /// An unrecognised value is a hard error rather than a silent fall back to the default:
    /// a typo would otherwise be reported as a streaming run.
    pub fn from_env() -> Self {
        match std::env::var("LUT_MODE") {
            Err(_) => LutMode::Streaming,
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "streaming" | "stream" => LutMode::Streaming,
                "full" | "precomputed" => LutMode::Full,
                other => panic!("LUT_MODE must be `streaming` or `full`, got `{other}`"),
            },
        }
    }

    /// Short tag for benchmark headers, so a run's mode is visible in its output.
    pub fn as_str(self) -> &'static str {
        match self {
            LutMode::Streaming => "streaming",
            LutMode::Full => "full",
        }
    }
}

/// Loads polynomial `m` of `lut` into `coeffs`, zero-padding past the end of the table.
///
/// Only the first `entries` coefficients are written: callers zero the tail once, since it
/// stays zero for every polynomial.
fn load_poly(coeffs: &mut [i64], lut: &[i32], m: usize, entries: usize) {
    let base_idx: usize = m * entries;
    for (j, slot) in coeffs.iter_mut().take(entries).enumerate() {
        let idx: usize = base_idx + j;
        *slot = if idx < lut.len() { lut[idx] as i64 } else { 0 };
    }
}

/// The whole LUT, encoded and DFT-ed: one prepared right operand per polynomial.
///
/// This is the [`LutMode::Full`] representation. Its footprint is `n_polys` prepared
/// polynomials -- for a table of `N^d` entries that is `N^(d-1)` of them -- so it is built
/// once, outside the timed region, and reused across evaluations.
pub struct PreparedLut<BE: Backend> {
    polys: Vec<CnvPVecR<BE::OwnedBuf, BE::DftWord, BE>>,
    /// Limb count of the plaintext the polynomials were prepared from (`mul_prepared`
    /// needs it: the prepared buffers do not carry their own size).
    poly_size: usize,
    entries_per_poly: usize,
    /// Length of the plaintext table this was built from, so `Lut::len()` reads the same
    /// in both modes.
    n_entries: usize,
}

impl<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>> PreparedLut<BE> {
    /// Encode and DFT `n_polys` polynomials of `entries` LUT entries each.
    ///
    pub fn new(
        module: &Module<BE>,
        params: &MuxParams,
        lut: &[i32],
        n_polys: usize,
        entries: usize,
        scratch: &mut ScratchArena<BE>,
    ) -> Self
    where
        Module<BE>: CnvPVecAlloc<BE> + Convolution<BE> + ModuleN,
    {
        let msg_k: TorusPrecision = params.log_p.into();
        let mut pt: BackendGLWEPlaintext<BE> = module.glwe_plaintext_alloc(params.base2k.into(), msg_k);
        let poly_size: usize = pt.size();
        // The tail past `entries` stays zero for every polynomial, so it is zeroed once here.
        let mut coeffs: Vec<i64> = vec![0i64; params.n_rlwe];

        let polys: Vec<CnvPVecR<BE::OwnedBuf, BE::DftWord, BE>> = (0..n_polys)
            .map(|m| {
                load_poly(&mut coeffs, lut, m, entries);
                pt.encode_vec_i64(&coeffs, msg_k);
                let mut prep: CnvPVecR<BE::OwnedBuf, BE::DftWord, BE> = alloc_plain_prep(module, &pt);
                prepare_plain_into(module, &mut prep, &pt, scratch);
                prep
            })
            .collect();

        Self { polys, poly_size, entries_per_poly: entries, n_entries: lut.len() }
    }

    /// Prepared table for OHCollapse: `ceil(L/N)` polynomials of `N` entries.
    pub fn for_tree(
        module: &Module<BE>,
        params: &MuxParams,
        lut: &[i32],
        scratch: &mut ScratchArena<BE>,
    ) -> Self
    where
        Module<BE>: CnvPVecAlloc<BE> + Convolution<BE> + ModuleN,
    {
        let n_polys: usize = lut.len().div_ceil(params.n_rlwe);
        Self::new(module, params, lut, n_polys, params.n_rlwe, scratch)
    }

    /// Limb count of the prepared polynomials, i.e. `mul_prepared`'s `b_size`.
    pub fn poly_size(&self) -> usize {
        self.poly_size
    }
}

/// The LUT an evaluator reads, in whichever residency the caller chose.
///
/// Cheap to pass by value: both variants are a reference plus a discriminant.
pub enum Lut<'a, BE: Backend> {
    /// The plaintext table; polynomials are encoded and DFT-ed as the evaluation reaches them.
    Streaming(&'a [i32]),
    /// A table prepared up front (see [`PreparedLut`]).
    Full(&'a PreparedLut<BE>),
}

impl<BE: Backend> Clone for Lut<'_, BE> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<BE: Backend> Copy for Lut<'_, BE> {}

impl<BE: Backend> Lut<'_, BE> {
    /// Number of LUT entries, identical in both modes -- the evaluators derive their
    /// polynomial and layer counts from it.
    pub fn len(&self) -> usize {
        match self {
            Lut::Streaming(lut) => lut.len(),
            Lut::Full(prep) => prep.n_entries,
        }
    }

    /// True for a zero-entry table. An evaluator handed one has nothing to select from, so
    /// this exists to be asserted on rather than to be branched on.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Buffers backing the streaming path: one plaintext, one coefficient vector, one prepared
/// operand, all reused across polynomials. Absent in [`LutMode::Full`], where nothing is
/// encoded during the evaluation.
struct StreamBuffers<BE: Backend> {
    pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord>,
    coeffs: Vec<i64>,
    prep: CnvPVecR<BE::OwnedBuf, BE::DftWord, BE>,
}

/// Hands LUT polynomials to an evaluator as prepared right operands, hiding which
/// residency they come from.
///
/// One instance per evaluation loop: construct it with the loop's polynomial width, then
/// call [`LutFeed::poly`] per index. In streaming mode that encodes and DFTs into the
/// feed's own buffer; in full mode it is an index into the prepared table.
pub struct LutFeed<'a, BE: Backend> {
    lut: Lut<'a, BE>,
    entries_per_poly: usize,
    msg_k: TorusPrecision,
    poly_size: usize,
    stream: Option<StreamBuffers<BE>>,
}

impl<'a, BE: Backend<OwnedBuf = Vec<u8>>> LutFeed<'a, BE> {
    /// `n_polys` is how many polynomials the caller's loop will ask for and `entries` how
    /// many LUT entries one of them carries. Both must agree with how a [`PreparedLut`] was
    /// built, which is checked here rather than surfacing as
    /// an index panic partway through the loop.
    pub fn new(
        module: &Module<BE>,
        params: &MuxParams,
        lut: Lut<'a, BE>,
        n_polys: usize,
        entries: usize,
    ) -> Self
    where
        Module<BE>: CnvPVecAlloc<BE> + ModuleN,
    {
        let msg_k: TorusPrecision = params.log_p.into();
        match lut {
            Lut::Full(prep) => {
                assert_eq!(
                    prep.entries_per_poly, entries,
                    "prepared LUT packs {} entries per polynomial, evaluator expects {entries}",
                    prep.entries_per_poly
                );
                assert!(
                    prep.polys.len() >= n_polys,
                    "prepared LUT holds {} polynomials, evaluator reads {n_polys}",
                    prep.polys.len()
                );
                Self { lut, entries_per_poly: entries, msg_k, poly_size: prep.poly_size, stream: None }
            }
            Lut::Streaming(_) => {
                let pt: GLWEPlaintext<BE::OwnedBuf, BE::ZnxWord> = module.glwe_plaintext_alloc(params.base2k.into(), msg_k);
                let poly_size: usize = pt.size();
                let prep: CnvPVecR<BE::OwnedBuf, BE::DftWord, BE> = alloc_plain_prep(module, &pt);
                // Tail past `entries` is zero for every polynomial, so it is zeroed once here.
                let coeffs: Vec<i64> = vec![0i64; params.n_rlwe];
                Self {
                    lut,
                    entries_per_poly: entries,
                    msg_k,
                    poly_size,
                    stream: Some(StreamBuffers { pt, coeffs, prep }),
                }
            }
        }
    }

    /// Limb count of the polynomials this feed yields, i.e. `mul_prepared`'s `b_size`.
    ///
    /// Read it once before the loop: [`LutFeed::poly`] borrows the feed for as long as the
    /// operand it returns is alive.
    pub fn poly_size(&self) -> usize {
        self.poly_size
    }

    /// The prepared right operand for LUT polynomial `m`.
    ///
    /// Streaming mode encodes and DFTs it into the feed's buffer, so the returned reference
    /// is only valid until the next call; full mode just indexes the prepared table.
    pub fn poly(
        &mut self,
        module: &Module<BE>,
        m: usize,
        scratch: &mut ScratchArena<BE>,
    ) -> &CnvPVecR<BE::OwnedBuf, BE::DftWord, BE>
    where
        Module<BE>: Convolution<BE> + ModuleN,
        BE : Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>
    {
        match (self.lut, &mut self.stream) {
            (Lut::Full(prep), _) => &prep.polys[m],
            (Lut::Streaming(raw), Some(buf)) => {
                load_poly(&mut buf.coeffs, raw, m, self.entries_per_poly);
                buf.pt.encode_vec_i64(&buf.coeffs, self.msg_k);
                {
                    time_span!("prepare_plain (LUT DFT)");
                    prepare_plain_into(module, &mut buf.prep, &buf.pt, scratch);
                }
                &buf.prep
            }
            (Lut::Streaming(_), None) => unreachable!("streaming feed always allocates its buffers"),
        }
    }
}

/// The benchmark LUT: a scrambled table, `lut[i]` a hash of `i` reduced mod `p`.
///
/// It used to be `lut[i] = i % p`, which is readable without a second table and wrong for
/// every purpose but timing. `p` divides `N` at every parameter set `paramux` emits, so
/// `i % p` makes **all `nu` LUT polynomials bit-identical** -- each is `0, 1, .., p-1`
/// repeated `N/p` times. Every candidate a second stage selects among is then the same
/// ciphertext, which is invisible in a timing or memory run and fatal in the other two:
///
///  - a CMux's `hi - lo` is exactly zero, noise included, so `--features noise-profile`
///    reports a *flat* CMux tree for VPack and OHVPack -- the second stage appears to cost
///    no noise at all, where it in fact costs one `E_ext` per level (paper, §5.2);
///  - any selection bug lands on a congruent entry and passes unnoticed. The tests and
///    `examples/ohvpack_check.rs` had each grown their own scrambled table for exactly this
///    reason; the benchmarks were the ones still on the linear one.
///
/// The hash is splitmix64's finalizer, which is cheap enough to stay negligible against
/// building a `2^33`-entry table and which decorrelates adjacent indices, so neighbouring
/// polynomials differ in essentially every coefficient.
///
/// `i32` entries rather than `i64`: at `N = 2048, d = 3` the table is `2^33` entries, and
/// halving the element width is the difference between 32 GiB and 64 GiB of plaintext.
pub fn build_lut(size: usize, p: usize) -> Vec<i32> {
    (0..size).map(|i| lut_entry(i, p)).collect()
}

/// Entry `i` of [`build_lut`]'s table, without materialising the table.
///
/// For the evaluators that never hold the table -- vertical packing generates candidate `c`
/// from entries `c*N .. (c+1)*N` on demand -- this is the entry point: a `2^33`-entry
/// `build_lut` is 32 GiB of plaintext, which is exactly what streaming exists to avoid.
pub fn lut_entry(i: usize, p: usize) -> i32 {
    let mut h: u64 = (i as u64).wrapping_add(0x9E3779B97F4A7C15);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    (h % p as u64) as i32
}
