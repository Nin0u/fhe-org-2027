// Same rationale as the crate-level allow in `src/lib.rs`: poulpy threads the module,
// scratch, layouts and keys through explicitly, so these signatures are long by design.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

use ohvpack::{params::{MuxParams, init_toy}, selector::*};
use poulpy_core::{
    EncryptionLayout, GGSWEncryptSk, GLWEDecrypt, GLWEEncryptSk, ScratchArenaTakeCore,
    layouts::{
        GLWESecretSampling,
        BackendGGSW, BackendGLWEPlaintext, BackendGLWESecret,
        Dsize, GGSWAtViewRef,GGSWLayout,
        GLWEInfos, GLWELayout, GLWEPlaintextLayout, GLWESecretPrepared, GLWESecretPreparedFactory, GLWEToBackendRef, ModuleCoreAlloc,
    }
};
use poulpy_cpu_ref::FFT64Ref;
use poulpy_hal::{api::{ScratchOwnedAlloc, ScratchOwnedBorrow}, layouts::{Backend, Module, ScalarZnx, ScratchArena, ScratchOwned, ZnxViewMut}};

type B = FFT64Ref;

fn setup(params: &mut MuxParams) -> (
    Module<B>,
    ScratchOwned<B>,
    GLWESecretPrepared<Vec<u8>, B>,
    EncryptionLayout<GLWELayout>,
    EncryptionLayout<GGSWLayout>,
) {
    let glwe_ct_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
        n: params.n_rlwe.into(),
        base2k: params.base2k.into(),
        k: params.k_rlwe.into(),
        rank: params.rank.into(),
    }).unwrap();
    let ggsw_ct_infos = EncryptionLayout::new_from_default_sigma(GGSWLayout {
        n: params.n_rlwe.into(),
        base2k: params.bg_bits.into(),
        k_aux: params.k_rlwe.into(),
        rank: params.rank.into(),
        dnum: params.l.into(),
        dsize: Dsize(1),
    }).unwrap();

    let module: Module<B> = Module::<B>::new(params.n_rlwe as u64);

    let scratch: ScratchOwned<B> = ScratchOwned::<B>::alloc(
        module.glwe_encrypt_sk_tmp_bytes(&glwe_ct_infos)
            .max(module.glwe_decrypt_tmp_bytes(&glwe_ct_infos))
            .max(module.ggsw_encrypt_sk_tmp_bytes(&ggsw_ct_infos)),
    );

    let prob: f64 = 0.5;
    let mut sk_glwe: BackendGLWESecret<B> = module.glwe_secret_alloc_from_infos(&glwe_ct_infos);
    module.glwe_secret_fill_ternary_prob(&mut sk_glwe, prob, &mut params.source_xs); // 1/2 0 and 1/4 of 1, -1
    let mut sk_glwe_dft: GLWESecretPrepared<Vec<u8>, B> =
        module.glwe_secret_prepared_alloc_from_infos(&glwe_ct_infos);
    module.glwe_secret_prepare(&mut sk_glwe_dft, &sk_glwe);

    (module, scratch, sk_glwe_dft, glwe_ct_infos, ggsw_ct_infos)
}

fn make_ggsw_selector_noprep<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &mut MuxParams,
    layout: &EncryptionLayout<GGSWLayout>,
    sk_glwe_dft: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    select: Option<i64>,
    scratch: &mut ScratchArena<BE>,
) -> BackendGGSW<B>
where
    Module<BE>: GGSWEncryptSk<BE>,
    for <'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    // Encode the selector as a ring rotation (same encoding as `make_ggsw_selector`);
    // `None` leaves the plaintext at 0, a one-hot vector's non-selected entry.
    let mut selector_pt: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(1);
    selector_pt.raw_mut().fill(0);
    if let Some(r) = select {
        encode_neg_rotation(selector_pt.raw_mut(), r, params.n_rlwe);
    }

    // Encryption that outputs a non-preprared ciphertext
    let mut ggsw_selector: BackendGGSW<BE> = module.ggsw_alloc_from_infos(layout);
    module.ggsw_encrypt_sk(
        &mut ggsw_selector,
        &selector_pt,
        sk_glwe_dft,
        layout,
        &mut params.source_xe,
        &mut params.source_xa,
        scratch,
    );
    ggsw_selector
}

/// Non-preparing twin of [`ohvpack::selector::make_ggsw_bit`], so the plaintext it encodes
/// can be decrypted. Same relationship to it as `make_ggsw_selector_noprep` has to
/// `make_ggsw_selector`.
fn make_ggsw_bit_noprep<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &mut MuxParams,
    layout: &EncryptionLayout<GGSWLayout>,
    sk_glwe_dft: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    bit: bool,
    scratch: &mut ScratchArena<BE>,
) -> BackendGGSW<BE>
where
    Module<BE>: GGSWEncryptSk<BE>,
    for <'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
{
    // A CMux consumes `[b]` as a *constant* polynomial, not a monomial.
    let mut bit_pt: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(1);
    bit_pt.raw_mut().fill(0);
    bit_pt.raw_mut()[0] = bit as i64;

    let mut ggsw: BackendGGSW<BE> = module.ggsw_alloc_from_infos(layout);
    module.ggsw_encrypt_sk(
        &mut ggsw,
        &bit_pt,
        sk_glwe_dft,
        layout,
        &mut params.source_xe,
        &mut params.source_xa,
        scratch,
    );
    ggsw
}

/// Decrypt `ct` into a plaintext of the given `base2k` (must match the
/// ciphertext) and full torus precision `full_k`, then decode the message at
/// `decode_k` (the scale at which the ±1 selector value actually lives).
fn decode_selector<R>(
    module: &Module<B>,
    ct: &R,
    sk: &GLWESecretPrepared<Vec<u8>, B>,
    base2k: usize,
    full_k: usize,
    decode_k: usize,
    n: usize,
    scratch: &mut ScratchArena<B>,
) -> Vec<i64>
where
    R: GLWEToBackendRef<B> + GLWEInfos,
{
    let pt_infos: GLWEPlaintextLayout = GLWEPlaintextLayout {
        n: n.into(),
        base2k: base2k.into(),
        k: full_k.into(),
    };
    let mut pt: BackendGLWEPlaintext<B> = module.glwe_plaintext_alloc_from_infos(&pt_infos);
    module.glwe_decrypt(ct, &mut pt, sk, scratch);
    let mut out: Vec<i64> = vec![0i64; n];
    pt.decode_vec_i64(&mut out, decode_k.into());
    out
}

/// Schoolbook multiply in `Z[X]/(X^N + 1)`.
fn ring_mul(a: &[i64], b: &[i64], n: usize) -> Vec<i64> {
    let mut r: Vec<i64> = vec![0i64; n];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if bj == 0 {
                continue;
            }
            // X^{i+j} folds back with a sign flip per wrap, since X^N = -1.
            let (mut k, mut sign) = (i + j, 1i64);
            while k >= n {
                k -= n;
                sign = -sign;
            }
            r[k] += sign * ai * bj;
        }
    }
    r
}

/// `encode_neg_rotation(r)` must produce `X^{-r}` over the whole exponent range, not
/// just the `r < N` range the tree evaluators happen to use. Vertical packing feeds it
/// `r` values across all of `[0, 2N)`, where the old `r - N` formula reflected the
/// rotation about `r = 3N/2` and silently returned the wrong coefficient.
#[test]
pub fn test_encode_neg_rotation() {
    let n: usize = 8;
    // Every coefficient distinct and non-zero, so picking the wrong one is visible.
    let f: Vec<i64> = (0..n).map(|c| c as i64 + 1).collect();

    for r in 0..(4 * n as i64) {
        let mut sel: Vec<i64> = vec![0i64; n];
        encode_neg_rotation(&mut sel, r, n);

        // f * X^{-r} brings coefficient r of f down to the constant term, with the
        // X^N = -1 sign flip once r has wrapped past N.
        let rr: usize = r.rem_euclid(2 * n as i64) as usize;
        let want: i64 = if rr < n { f[rr] } else { -f[rr - n] };
        let got: i64 = ring_mul(&f, &sel, n)[0];
        assert_eq!(got, want, "X^-{r}: constant term {got} != expected {want}");
    }

    // Negative exponents are rotations too, not an "encrypt 0" sentinel: the zero entries
    // of a one-hot vector are `None` at the `make_*_selector` level. `r = -m` must give
    // X^{m}, including at the m = N boundary where the exponent folds to -X^0.
    for r in -(4 * n as i64)..0 {
        let mut sel: Vec<i64> = vec![0i64; n];
        encode_neg_rotation(&mut sel, r, n);
        let m: usize = (-r).rem_euclid(2 * n as i64) as usize;
        let want: i64 = if m < n { f[0] } else { -f[0] };
        // f * X^{m} sends f's constant term to coefficient `m mod 2N`, with a sign flip
        // once it has wrapped past N.
        let got: i64 = ring_mul(&f, &sel, n)[m % n];
        assert_eq!(got, want, "X^{}: coefficient {} is {got} != expected {want}", m, m % n);
    }
}


/// `make_ggsw_bit` encodes the *constant* `b in {0, 1}` a CMux gate consumes, which is not
/// `make_ggsw_selector`'s monomial `X^{-r}`. The two agree only at `r = 0`, and the point
/// of the separate entry point is that the disagreement everywhere else is silent: a CMux
/// handed `X^{-1}` multiplies by a rotation instead of selecting.
#[test]
pub fn test_make_ggsw_bit() {
    let mut params: MuxParams = init_toy();
    let (module, mut scratch, sk_glwe_dft, _glwe_ct_infos, ggsw_ct_infos) = setup(&mut params);
    let n: usize = params.n_rlwe;

    let decode = |ggsw: &BackendGGSW<B>, params: &MuxParams, scratch: &mut ScratchOwned<B>| -> Vec<i64> {
        decode_selector(
            &module, &ggsw.at_view(0, 0), &sk_glwe_dft,
            params.bg_bits, params.k_rlwe, params.bg_bits, n, &mut scratch.borrow(),
        )
    };

    let mut expect_one: Vec<i64> = vec![0i64; n];
    expect_one[0] = 1;
    let expect_zero: Vec<i64> = vec![0i64; n];

    let ggsw_true = make_ggsw_bit_noprep(&module, &mut params, &ggsw_ct_infos, &sk_glwe_dft, true, &mut scratch.borrow());
    let got = decode(&ggsw_true, &params, &mut scratch);
    assert_eq!(got, expect_one, "bit = 1 decoded {got:?} != [1, 0, ...]");

    let ggsw_false = make_ggsw_bit_noprep(&module, &mut params, &ggsw_ct_infos, &sk_glwe_dft, false, &mut scratch.borrow());
    let got = decode(&ggsw_false, &params, &mut scratch);
    assert_eq!(got, expect_zero, "bit = 0 decoded {got:?} != all-zero");

    // Agrees with the rotation encoding at r = 0 ...
    let rot_zero = make_ggsw_selector_noprep(&module, &mut params, &ggsw_ct_infos, &sk_glwe_dft, Some(0), &mut scratch.borrow());
    let got = decode(&rot_zero, &params, &mut scratch);
    assert_eq!(got, expect_one, "X^0 decoded {got:?} != [1, 0, ...]");

    // ... and differs everywhere else, which is why the bit case is not spelled `Some(1)`.
    let rot_one = make_ggsw_selector_noprep(&module, &mut params, &ggsw_ct_infos, &sk_glwe_dft, Some(1), &mut scratch.borrow());
    let got = decode(&rot_one, &params, &mut scratch);
    let mut expect_rot_one: Vec<i64> = vec![0i64; n];
    expect_rot_one[n - 1] = -1; // X^{-1} = -X^{N-1}
    assert_eq!(got, expect_rot_one, "X^-1 decoded {got:?} != -X^(N-1)");
    assert_ne!(got, expect_one, "`Some(1)` must not encode the bit 1");
}

/// `ohvpack_make_selectors` builds the first stage's one-hot vector on the paper's uniform
/// `G = nu/alpha` rule. This pins the width, which entry is hot, and that every other
/// entry really encrypts zero rather than a stray rotation.
///
/// The `log2(alpha)` tree bits it returns are prepared, so they are not decryptable here;
/// `tests/test_ohvpack.rs` covers them end-to-end. Only their count is checked.
#[test]
pub fn test_ohvpack_make_selectors() {
    let mut params: MuxParams = init_toy();
    params.alpha = 32;
    let (module, mut scratch, sk_glwe_dft, glwe_ct_infos, ggsw_ct_infos) = setup(&mut params);

    let n: usize = params.n_rlwe;
    // nu = N^2 = 65536 polynomials in groups of 32 -> G = 2048, and
    // selector = (5*alpha + 7)*N + 130 sits in group 5 at b = 7 with i0 = 130.
    let selector: i64 = (5 * 32 + 7) * n as i64 + 130;
    let hot: usize = 5;
    let rot: usize = 130;

    let (selectors_glwe, tree_bits) = ohvpack_make_selectors::<B>(
        &module,
        &mut params,
        &glwe_ct_infos,
        &ggsw_ct_infos,
        &sk_glwe_dft,
        selector,
        &mut scratch.borrow(),
    );

    assert_eq!(selectors_glwe.len(), 2048, "G = nu/alpha, not the collapsed nu");
    assert_eq!(selectors_glwe.len(), params.ohcollapse_n_groups());
    assert_eq!(tree_bits.len(), 5, "log2(alpha) = 5 CMux levels, one TRGSW bit each");

    let mut expect_hot: Vec<i64> = vec![0i64; n];
    expect_hot[n - rot] = -1; // X^{-130} = -X^{N-130}
    let zero: Vec<i64> = vec![0i64; n];

    // Every entry, not just the hot one: a one-hot vector whose cold entries are not zero
    // leaks other groups into the accumulators and the sum decodes to garbage.
    for (i, sel) in selectors_glwe.iter().enumerate() {
        let got = decode_selector(
            &module, sel, &sk_glwe_dft,
            params.base2k, params.k_rlwe, params.log_p, n, &mut scratch.borrow(),
        );
        let expect = if i == hot { &expect_hot } else { &zero };
        assert_eq!(&got, expect, "GLWE[{i}] decoded {got:?} != {expect:?}");
    }
}

/// `alpha` below the balanced point `sqrt(L/N)`: the first stage's one-hot vector is
/// `L/(alpha*N)` wide, so it grows as `alpha` shrinks. Capping it at `alpha` entries (as
/// the code did before) left every group beyond the `alpha`-th unaddressable, which for a
/// LUT of `N^d` entries meant no index past `alpha^2 * N` could ever be selected.
#[test]
pub fn test_ohvpack_make_selectors_wide_first_stage() {
    let mut params: MuxParams = init_toy();
    params.alpha = 16; // sqrt(L/N) is 256 here, so the first stage is 16x wider than alpha
    let (module, mut scratch, sk_glwe_dft, glwe_ct_infos, ggsw_ct_infos) = setup(&mut params);

    let n: i64 = params.n_rlwe as i64;
    let n_groups: usize = params.lut_n_polys() / params.alpha; // 65536 / 16 = 4096

    // Pick an index in the last group -- exactly what the old alpha-wide vector could not
    // address: group 4095, position 7 within it, coefficient 130 within the polynomial.
    let group_ind: usize = n_groups - 1;
    let pos: i64 = 7;
    let rot: i64 = 130;
    let selector: i64 = (group_ind as i64 * params.alpha as i64 + pos) * n + rot;

    let mut expect_rot: Vec<i64> = vec![0i64; params.n_rlwe];
    expect_rot[params.n_rlwe - rot as usize] = -1;
    let expect_zero: Vec<i64> = vec![0i64; params.n_rlwe];

    let (selectors_glwe, tree_bits) = ohvpack_make_selectors::<B>(
        &module,
        &mut params,
        &glwe_ct_infos,
        &ggsw_ct_infos,
        &sk_glwe_dft,
        selector,
        &mut scratch.borrow(),
    );
    assert_eq!(selectors_glwe.len(), n_groups, "first stage must be L/(alpha*N) wide");
    assert_eq!(tree_bits.len(), 4, "log2(alpha) = 4 CMux levels, one TRGSW bit each");

    let got_sel = decode_selector(
        &module, &selectors_glwe[group_ind], &sk_glwe_dft,
        params.base2k, params.k_rlwe, params.log_p, params.n_rlwe, &mut scratch.borrow(),
    );
    assert_eq!(got_sel, expect_rot, "GLWE[{group_ind}] decoded {got_sel:?} != {expect_rot:?}");

    let got_zero = decode_selector(
        &module, &selectors_glwe[0], &sk_glwe_dft,
        params.base2k, params.k_rlwe, params.log_p, params.n_rlwe, &mut scratch.borrow(),
    );
    assert_eq!(got_zero, expect_zero, "GLWE[0] decoded {got_zero:?} != 0");
}
