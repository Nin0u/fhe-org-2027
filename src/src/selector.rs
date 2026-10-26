use poulpy_bin_fhe::bdd_arithmetic::{FheUintPrepared, FheUintPreparedEncryptSk, ToBits, UnsignedInteger};
use poulpy_core::{
    EncryptionLayout, GGSWEncryptSk, GLWEEncryptSk, ScratchArenaTakeCore, layouts::{
        BackendGLWE, BackendGLWEPlaintext, GGSW, GGSWLayout, GGSWPrepared, GGSWPreparedFactory, GLWELayout, GLWESecretPrepared, ModuleCoreAlloc,
    },
};
use poulpy_hal::layouts::{Backend, HostBackend, HostDataMut, Module, ScalarZnx, ScratchArena, ZnxViewMut};
use crate::ohvpack::ohvpack_tree_depth;
use crate::params::MuxParams;

/// Writes the coefficients of `X^{-r}` in `Z[X]/(X^N + 1)` into `coeffs` (assumed
/// zeroed, length `N`).
///
/// `select` is first reduced mod `2N`, which is the order of the rotation, so any `|r| >= 2N` is handled.
/// The reduced r can now be either positive or negative.
/// If r is positive or zero :
/// - (r == 0): `X^{0} = 1 `.
/// - (0 < r <= N): `X^{-r} = X^{N-r} * X^{-N} = -X^{N-r}`.
/// - (N < r < 2N): `X^{-r} = X^{2N-r}` with `0 < 2N-r < N`, so the exponent is already in range and the sign stays positive.
///
/// If r is negative, write `m = -r > 0`:
/// - (0 < m < N): `X^{-r} = X^{m}`.
/// - (N <= m < 2N): `X^{m} = X^{m-N} * X^{N} = -X^{m-N}`, with `0 <= m-N < N`.
///
/// Note this always encodes a *rotation*, never the zero polynomial: callers that need a
/// one-hot vector's zero entries pass `None` to [`make_glwe_selector`] /
/// [`make_ggsw_selector`] rather than a negative `select`.
pub fn encode_neg_rotation(coeffs: &mut [i64], select: i64, n: usize) {
    let r: i64 = select % (2 * n as i64);
    if r < 0 {
        let m: usize = (-r) as usize;
        // m == n folds to -X^0, which the first branch cannot express (index n is out of
        // range), hence the strict `<`.
        if m < n { coeffs[m] = 1; }
        else { coeffs[m - n] = -1; }
    }
    else {
        if r == 0 { coeffs[0] = 1; }
        else if r <= n as i64 { coeffs[n - r as usize] = -1; }
        else { coeffs[2 * n - r as usize] = 1; }
    }

}

/// Builds GLWE selector as an encrypted ring rotation.
///
/// `select` is `Some(r)` for the rotation `X^{-r}` and `None` for an encryption of the
/// zero polynomial -- the non-selected entries of a one-hot vector.
///
/// WARNING : The output is not prepared !
pub fn make_glwe_selector<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &mut MuxParams,
    layout: &EncryptionLayout<GLWELayout>,
    sk_glwe_dft: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    select: Option<i64>,
    scratch: &mut ScratchArena<BE>,
) -> BackendGLWE<BE>
where
    Module<BE>: GLWEEncryptSk<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    // Encode the selector as a ring rotation, or leave it at 0 for a one-hot zero entry.
    let mut selector_pt: BackendGLWEPlaintext<BE> = module.glwe_plaintext_alloc(params.base2k.into(), params.log_p.into());
    let mut vec_glwe_selector: Vec<i64> = vec![0; params.n_rlwe];

    if let Some(r) = select {
        encode_neg_rotation(&mut vec_glwe_selector, r, params.n_rlwe);
    }
    selector_pt.encode_vec_i64(&vec_glwe_selector, params.log_p.into());

    let mut glwe_selector: BackendGLWE<BE> = module.glwe_alloc_from_infos(layout);
    module.glwe_encrypt_sk(
        &mut glwe_selector,
        &selector_pt,
        sk_glwe_dft,
        layout,
        &mut params.source_xe,
        &mut params.source_xa,
        scratch,
    );
    glwe_selector
}

/// Builds GGSW selector as an encrypted ring rotation.
///
/// `select` is `Some(r)` for the rotation `X^{-r}` and `None` for an encryption of the
/// zero polynomial -- the non-selected entries of a one-hot vector.
///
/// The output is prepared.
pub fn make_ggsw_selector<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &mut MuxParams,
    layout: &EncryptionLayout<GGSWLayout>,
    sk_glwe_dft: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    select: Option<i64>,
    scratch: &mut ScratchArena<BE>,
) -> GGSWPrepared<BE::OwnedBuf, BE>
where
    Module<BE>: GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    for <'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{

    // Encode the selector as a ring rotation, or leave it at 0 for a one-hot zero entry.
    let mut selector_pt: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(1);
    selector_pt.raw_mut().fill(0);
    if let Some(r) = select {
        encode_neg_rotation(selector_pt.raw_mut(), r, params.n_rlwe);
    }

    encrypt_ggsw_prepared(module, params, layout, sk_glwe_dft, &selector_pt, scratch)
}

/// Builds a GGSW encrypting the constant bit `bit`, which is what a CMux gate consumes.
///
/// This is *not* [`make_ggsw_selector`]'s encoding: a CMux needs `[b]` with `b in {0, 1}`
/// as a constant polynomial, where `make_ggsw_selector(Some(r))` encodes the monomial
/// `X^{-r}`. The two agree only at `r = 0`, so the bit case gets its own entry point
/// rather than being spelled as the `Some(0)` / `None` pun.
///
/// The output is prepared.
pub fn make_ggsw_bit<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &mut MuxParams,
    layout: &EncryptionLayout<GGSWLayout>,
    sk_glwe_dft: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    bit: bool,
    scratch: &mut ScratchArena<BE>,
) -> GGSWPrepared<BE::OwnedBuf, BE>
where
    Module<BE>: GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    for <'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let mut bit_pt: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(1);
    bit_pt.raw_mut().fill(0);
    bit_pt.raw_mut()[0] = bit as i64;

    encrypt_ggsw_prepared(module, params, layout, sk_glwe_dft, &bit_pt, scratch)
}

/// Encrypts an already-encoded scalar plaintext into a prepared GGSW.
///
/// The tail shared by [`make_ggsw_selector`] and [`make_ggsw_bit`]: the two differ only in
/// how they encode their plaintext, not in how it is encrypted and prepared.
fn encrypt_ggsw_prepared<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &mut MuxParams,
    layout: &EncryptionLayout<GGSWLayout>,
    sk_glwe_dft: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    selector_pt: &ScalarZnx<BE::OwnedBuf, BE::ZnxWord>,
    scratch: &mut ScratchArena<BE>,
) -> GGSWPrepared<BE::OwnedBuf, BE>
where
    Module<BE>: GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    for <'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    // Encryption that outputs a non-preprared ciphertext
    let mut ggsw_selector: GGSW<BE::OwnedBuf, BE::ZnxWord> = module.ggsw_alloc_from_infos(layout);
    module.ggsw_encrypt_sk(
        &mut ggsw_selector,
        selector_pt,
        sk_glwe_dft,
        layout,
        &mut params.source_xe,
        &mut params.source_xa,
        scratch,
    );

    let mut ggsw_selector_dft= module.ggsw_prepared_alloc_from_infos(layout);
    module.ggsw_prepare(&mut ggsw_selector_dft, &ggsw_selector, scratch);
    ggsw_selector_dft
}

/// What [`ohvpack_make_selectors`] builds: the first stage's one-hot GLWE vector plus one
/// TRGSW-encrypted bit per level of the CMux tree.
pub type OhvpackSelectors<BE> = (Vec<BackendGLWE<BE>>, Vec<GGSWPrepared<<BE as Backend>::OwnedBuf, BE>>);

/// Selectors for OHVPack (paper, §5.1): the one-hot TRLWE vector of the shared first stage,
/// and the `log2(alpha)` TRGSW-encrypted bits that drive the second stage's CMux tree.
///
/// The one-hot vector is the object §4.3 describes: `G = nu/alpha` entries, hot at
/// `g = i/(alpha*N)`, the hot entry carrying `X^{-i0}` with `i0 = i mod N`; every other
/// entry encrypts 0, which is what makes the stage a one-hot encoding.
///
/// The second stage takes `log2(alpha)` TRGSW encryptions of the bits of
/// `b = (i / N) mod alpha`, one per level -- the shape §5.1 specifies.
///
/// It deliberately does *not* reuse the [`FheUintPrepared`] over `i` that
/// [`vpack_make_selectors`] builds, even though bits `log2(N) .. log2(N) + log2(alpha)` of
/// `i` are exactly the ones wanted. That type allocates *and encrypts* `T::BITS` GGSWs
/// whatever `alpha` is, and `T` is pinned by the width of `i` rather than by the width of
/// the window: a 2^33 table needs `u64`, so 64 GGSWs get built where `log2(alpha)` -- 5 to
/// 11 across the shipped sets -- are read. That is ~20 MiB of resident key material per
/// selector at `N = 1024`, against 1.6 MiB here. VPack keeps `FheUintPrepared`, whose tree
/// really does read a `log2(L/N)`-wide window of `i`.
pub fn ohvpack_make_selectors<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &mut MuxParams,
    glwe_layout: &EncryptionLayout<GLWELayout>,
    ggsw_layout: &EncryptionLayout<GGSWLayout>,
    sk_glwe_dft: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    selector: i64,
    scratch: &mut ScratchArena<BE>,
) -> OhvpackSelectors<BE>
where
    Module<BE>: GLWEEncryptSk<BE> + GGSWEncryptSk<BE> + GGSWPreparedFactory<BE>,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    let n: i64 = params.n_rlwe as i64;
    let alpha: i64 = params.alpha as i64;
    assert!(
        params.alpha.is_power_of_two(),
        "OHVPack's CMux tree is binary, so alpha must be a power of two, got {}",
        params.alpha
    );

    let index: i64 = selector;
    let n_groups: usize = params.ohcollapse_n_groups();
    let g: usize = (index / (alpha * n)) as usize;
    assert!(
        g < n_groups,
        "first-stage selector index {g} out of range: only {n_groups} groups for a LUT of \
         {} polynomials at alpha = {}",
        params.lut_n_polys(),
        params.alpha
    );
    let i0: i64 = index % n;

    // One-hot: the hot entry carries the rotation, every other encrypts zero.
    let selectors_glwe: Vec<BackendGLWE<BE>> = (0..n_groups)
        .map(|k| {
            let select: Option<i64> = if k == g { Some(i0) } else { None };
            make_glwe_selector(module, params, glwe_layout, sk_glwe_dft, select, scratch)
        })
        .collect();

    // The CMux tree's bits, LSB first: level `j` consumes bit `j` of `b`. The tree runs
    // over the `min(alpha, nu)` accumulators the first stage leaves, which is `alpha`
    // everywhere but past the `alpha = nu` endpoint.
    let n_candidates: usize = params.ohcollapse_n_candidates();
    let mut b: i64 = (index / n) % n_candidates as i64;
    let levels: usize = ohvpack_tree_depth(n_candidates);

    let mut selectors_ggsw: Vec<GGSWPrepared<BE::OwnedBuf, BE>> = Vec::with_capacity(levels);
    for _ in 0..levels {
        selectors_ggsw.push(make_ggsw_bit(module, params, ggsw_layout, sk_glwe_dft, b & 1 == 1, scratch));
        b >>= 1;
    }

    (selectors_glwe, selectors_ggsw)
}

/// Scratch bytes [`vpack_make_selectors`] needs.
///
/// Not just `ggsw_encrypt_sk_tmp_bytes`: `FheUintPrepared::encrypt_sk` takes a
/// `ScalarZnx` of `N` words for the bit it is encoding *before* it calls
/// `ggsw_encrypt_sk`, so the arena has to hold both at once. Callers used to get away
/// with the encryption term alone because the vertical-packing evaluation term happened to
/// be larger and dominated their `max`; it no longer is.
pub fn vpack_make_selectors_tmp_bytes<BE, E>(module: &Module<BE>, n: usize, ggsw_layout: &E) -> usize
where
    BE: Backend,
    Module<BE>: GGSWEncryptSk<BE>,
    E: poulpy_core::layouts::GGSWInfos,
{
    n * size_of::<i64>() + module.ggsw_encrypt_sk_tmp_bytes(ggsw_layout)
}

/// Builds the two Vertical Packing selectors for index `k`.
///
/// Returns the CMux-tree selectors = The log2(L/N) higher bits of `k`,
/// and the single GGSW encrypting the ring rotation  `X^{-(k % N)}` that collapses step 2 to one external product.
///
/// Size the arena with [`vpack_make_selectors_tmp_bytes`].
pub fn vpack_make_selectors<T, BE>(
    module: &Module<BE>,
    params: &mut MuxParams,
    ggsw_layout: &EncryptionLayout<GGSWLayout>,
    sk_glwe_dft: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    selector: T,
    scratch: &mut ScratchArena<BE>,
) -> (FheUintPrepared<BE::OwnedBuf, T, BE>, GGSWPrepared<BE::OwnedBuf, BE>)
where
    T: UnsignedInteger + ToBits + Into<u64>,
    BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64> + HostBackend + 'static,
    Module<BE>: GGSWEncryptSk<BE> + GGSWPreparedFactory<BE> + FheUintPreparedEncryptSk<T, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let mut tree_selector: FheUintPrepared<BE::OwnedBuf, T, BE> =
        FheUintPrepared::<BE::OwnedBuf, T, BE>::alloc_from_infos(module, ggsw_layout);
    tree_selector.encrypt_sk(
        module,
        selector,
        sk_glwe_dft,
        ggsw_layout,
        &mut params.source_xe,
        &mut params.source_xa,
        scratch,
    );

    // Only the low log2(N) bits are left after the CMux tree; encode them as X^{-r}.
    let r: i64 = (selector.into() % params.n_rlwe as u64) as i64;
    let rot_selector: GGSWPrepared<BE::OwnedBuf, BE> =
        make_ggsw_selector(module, params, ggsw_layout, sk_glwe_dft, Some(r), scratch);

    (tree_selector, rot_selector)
}
