//! Shared ciphertext layouts, scratch sizing and key generation.
//!
//! Every benchmark and every integration test needs the same thing before it can evaluate
//! anything: the layouts derived from a [`MuxParams`], a scratch arena sized against all
//! the operations that will run, and a GLWE secret with its DFT and LWE images. Neither
//! construction packs, so there are no evaluation keys beyond the selectors themselves.
//!
//! It lives here once instead, generic over the backend so the benchmarks can drive it
//! with the feature-selected AVX/AVX-512 module and the tests with the reference one.

use poulpy_core::{
    EncryptionLayout, GLWEDecrypt, GLWEEncryptSk, GLWEExternalProduct,
    GLWEMulPlain, GGSWEncryptSk, LWEDecrypt,
    layouts::{
        BackendGLWESecret, BackendLWESecret, Dsize, GGSWLayout, GLWELayout,
        GGSWPreparedFactory, GLWEPlaintextLayout, GLWESecretPrepared, GLWESecretPreparedFactory,
        GLWESecretSampling,
        LWELayout,
        ModuleCoreAlloc, SecretConversion,
    },
};
use poulpy_hal::layouts::{Backend, Module, NoiseInfos};

use crate::params::MuxParams;

/// Ternary secret density: half the coefficients are zero, the rest split between ±1.
const SECRET_ZERO_PROB: f64 = 0.5;

/// The ciphertext layouts one [`MuxParams`] gives rise to, all at GLWE precision `k_rlwe`.
pub struct MuxLayouts {
    /// Extracted LWE ciphertexts (the tree layers hand these between rounds).
    pub lwe: LWELayout,
    /// GLWE ciphertexts, at `k_rlwe`.
    pub glwe: EncryptionLayout<GLWELayout>,
    /// GLWE plaintexts, used to size `glwe_mul_plain` scratch.
    pub glwe_pt: GLWEPlaintextLayout,
    /// GGSW selectors, on the `bg_bits` × `l` gadget.
    pub ggsw: EncryptionLayout<GGSWLayout>,
}

impl MuxLayouts {
    pub fn new(params: &MuxParams) -> Self {
        let noise: NoiseInfos = NoiseInfos {
            k: params.k_rlwe,
            sigma: params.sigma,
            bound: params.bound,
        };
        let lwe: LWELayout = LWELayout {
            n: params.n_rlwe.into(),
            k: params.k_rlwe.into(),
            base2k: params.base2k.into(),
        };
        let glwe = EncryptionLayout::new(
            GLWELayout {
                n: params.n_rlwe.into(),
                base2k: params.base2k.into(),
                k: params.k_rlwe.into(),
                rank: params.rank.into(),
            },
            noise,
        )
        .unwrap();
        let glwe_pt: GLWEPlaintextLayout = GLWEPlaintextLayout {
            n: params.n_rlwe.into(),
            base2k: params.base2k.into(),
            k: params.k_rlwe.into(),
        };
        let ggsw = EncryptionLayout::new(
            GGSWLayout {
                n: params.n_rlwe.into(),
                base2k: params.bg_bits.into(),
                k_aux: params.k_rlwe.into(),
                rank: params.rank.into(),
                dnum: params.l.into(),
                dsize: Dsize(1),
            },
            noise,
        )
        .unwrap();
        Self { lwe, glwe, glwe_pt, ggsw }
    }
}

/// Scratch bytes needed by every operation an evaluation (and its setup and its
/// verification) invokes -- the max over all of them, since the arena is reused.
pub fn scratch_bytes<BE: Backend>(module: &Module<BE>, layouts: &MuxLayouts) -> usize
where
    Module<BE>: GLWEEncryptSk<BE>
        + GLWEDecrypt<BE>
        + LWEDecrypt<BE>
        + GLWEMulPlain<BE>
        + GGSWEncryptSk<BE>
        + GGSWPreparedFactory<BE>
        + GLWEExternalProduct<BE>,
{
    module
        .glwe_encrypt_sk_tmp_bytes(&layouts.glwe)
        .max(module.glwe_decrypt_tmp_bytes(&layouts.glwe))
        .max(module.lwe_decrypt_tmp_bytes(&layouts.lwe))
        .max(module.glwe_mul_plain_tmp_bytes(&layouts.glwe, &layouts.glwe, &layouts.glwe_pt))
        .max(module.ggsw_encrypt_sk_tmp_bytes(&layouts.ggsw))
        .max(module.ggsw_prepare_tmp_bytes(&layouts.ggsw))
        .max(module.glwe_external_product_tmp_bytes(&layouts.glwe, &layouts.glwe, &layouts.ggsw))
}

/// A GLWE secret in the three shapes the evaluations need it.
pub struct SecretKeys<BE: Backend> {
    /// Coefficient-domain GLWE secret; the noise probes decrypt under this one.
    pub glwe: BackendGLWESecret<BE>,
    /// DFT image, taken by every encryption.
    pub glwe_dft: GLWESecretPrepared<BE::OwnedBuf, BE>,
    /// The secret `lwe_sample_extract` produces ciphertexts under: `sk_lwe[0] =
    /// sk_glwe[0]` and `sk_lwe[k] = -sk_glwe[N-k]` for `k >= 1`, i.e. the `X -> X^{-1}`
    /// image of `glwe`.
    pub lwe: BackendLWESecret<BE>,
}

/// Samples a ternary GLWE secret and derives its DFT and LWE images.
pub fn secret_keys<BE: Backend<OwnedBuf = Vec<u8>, ZnxWord = i64>>(
    module: &Module<BE>,
    params: &mut MuxParams,
    layouts: &MuxLayouts,
) -> SecretKeys<BE>
where
    Module<BE>: GLWESecretSampling<BE> + GLWESecretPreparedFactory<BE> + SecretConversion<BE>,
{
    let mut glwe: BackendGLWESecret<BE> = module.glwe_secret_alloc_from_infos(&layouts.glwe);
    module.glwe_secret_fill_ternary_prob(&mut glwe, SECRET_ZERO_PROB, &mut params.source_xs);

    let mut glwe_dft: GLWESecretPrepared<BE::OwnedBuf, BE> =
        module.glwe_secret_prepared_alloc_from_infos(&layouts.glwe);
    module.glwe_secret_prepare(&mut glwe_dft, &glwe);

    let lwe: BackendLWESecret<BE> = module.lwe_secret_from_glwe_secret(&glwe, params.n_rlwe.into());

    SecretKeys { glwe, glwe_dft, lwe }
}
