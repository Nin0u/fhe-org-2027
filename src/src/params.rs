use poulpy_hal::source::Source;
use poulpy_core::{DEFAULT_SIGMA_XE, DEFAULT_BOUND_XE} ;
use rand::{RngExt};
use rand::rngs::ThreadRng;

use serde_json::Value;
use std::fs::File;
use std::io::Read;

pub struct MuxParams {
    pub n_rlwe: usize,     // RLWE degree
    pub rank: usize,       // GLWE rank
    pub log_p: usize,      // Plaintext modulus bit size
    pub p: usize,          // Plaintext Modulus
    pub base2k: usize,     // Base2K
    pub k_rlwe: usize,     // Torus Precision for the RLWE
    pub bg_bits: usize,    // Base2k for GGSW encryption
    pub l: usize,          // Dnum for GGSW encryption
    pub sigma: f64,        // Standard deviation
    pub bound: f64,        // Distribution bound, often set at 6*sigma
    pub depth: usize,      // LUT decomposition depth `d`: the table holds `N^d` entries
    pub alpha: usize,      // OHVPack collapsing factor (paper, Sec. 4.6)
    pub source_xs: Source, // PRNG for the private key
    pub source_xe: Source, // PRNG for the error
    pub source_xa: Source, // PRNG for the mask
}

impl MuxParams {
    /// Number of degree-`N` polynomials the LUT splits into.
    ///
    /// The LUT holds `N^depth` entries packed `N` per polynomial, so this is `N^(depth-1)`.
    /// (`depth` here is the LUT's decomposition depth `d`, not the depth of the tree.)
    pub fn lut_n_polys(&self) -> usize {
        self.n_rlwe.checked_pow((self.depth - 1) as u32).unwrap_or(usize::MAX)
    }

    /// Number of groups OHCollapse splits the table into, i.e. the width
    /// `G = L/(alpha*N)` of its one-hot TRLWE selector vector (paper, §4.3).
    ///
    /// The `L/N` polynomials are tiled into groups of `alpha`, so this is
    /// `ceil(n_polys / alpha)` -- *not* `alpha`. The two widths coincide at the balanced
    /// point `alpha = sqrt(nu)`, which is exactly the collapsing factor §4.6 settles on;
    /// below it the first stage is the *wide* one.
    pub fn ohcollapse_n_groups(&self) -> usize {
        self.lut_n_polys().div_ceil(self.alpha.max(1))
    }

    /// Number of candidates OHCollapse leaves for the CMux tree, i.e. `min(alpha, nu)`.
    ///
    /// The stage produces one accumulator per position inside a group, and a group holds
    /// `alpha` polynomials -- but only `nu` of them exist, so at `alpha > nu` the trailing
    /// `alpha - nu` positions are encryptions of 0 no group ever writes to. Clamping keeps
    /// the tree over the candidates that carry a table entry.
    pub fn ohcollapse_n_candidates(&self) -> usize {
        self.alpha.max(1).min(self.lut_n_polys())
    }
}

/// Toy parameters exercising both constructions: OHVPack and Vertical Packing.
///
/// Vertical Packing is what sets the floor here: it runs the deepest CMux tree, and a CMux
/// compounds two errors:
///
///  - The external product drops everything below `base2k * dnum` bits of its operand, so
///    the GGSW gadget must span its modulus: `bg_bits * l >= k_rlwe`. The previous values
///    (`bg_bits = 5, l = 2`) covered 10 bits of a 27-bit torus, which one CMux already
///    destroys.
///  - The GLWE torus needs headroom above the `log_p`-bit message for the noise the tree
///    accumulates. Widening the gadget alone does not buy this: at `k_rlwe = 26` an
///    8-level tree still corrupts ~4% of indices even with a 65-bit gadget. `k_rlwe = 39`
///    is what clears it.
///
/// So the gadget below is sized to span its modulus at a uniform `base2k = 13`.
/// `n_rlwe` stays at 256 to keep the toy tests small; the cost is in torus precision
/// (3-limb GLWE), not in ring degree.
pub fn init_toy() -> MuxParams
{
    MuxParams {
        n_rlwe: 256,
        rank: 1,
        log_p: 4,
        p: 16,
        base2k: 13,
        k_rlwe: 39,  // 3 limbs of 13
        bg_bits: 13,
        l: 3,        // 13 * 3 = 39 = k_rlwe -> gadget spans the GGSW modulus
        sigma: DEFAULT_SIGMA_XE,
        bound: DEFAULT_BOUND_XE,
        depth: 3,
        alpha: 256,
        source_xs: Source::new([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        source_xe: Source::new([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
        source_xa: Source::new([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
    }
}

pub fn generate_seed(rng: &mut ThreadRng) -> [u8; 32] {
    let arr: [u8; 32] = std::array::from_fn(|_| rng.random());
    arr
}

pub fn params_from_json(filepath: &str) -> Vec<MuxParams> {
    let mut file = File::open(filepath).unwrap();
    let mut data = String::new();
    file.read_to_string(&mut data).unwrap();
    let read_json: Value = serde_json::from_str(&data).unwrap();

    let mut rng = rand::rng();

    let mut result: Vec<MuxParams> = Vec::new();
    if let Value::Array(items) = read_json {
        for item in items {
            let mut params: MuxParams = MuxParams {
                n_rlwe: item.get("N").and_then(|v| v.as_u64()).expect("Error while converting N") as usize,
                rank: 1,
                log_p: 0,
                p: item.get("p").and_then(|v| v.as_u64()).expect("Error while converting p") as usize,
                base2k: 19,
                k_rlwe: item
                    .get("powq")
                    .and_then(|v| v.as_u64())
                    .expect("Error while converting powq") as usize,
                bg_bits: item.get("Bg").and_then(|v| v.as_u64()).expect("Error while converting b_g") as usize,
                l: item.get("l").and_then(|v| v.as_u64()).expect("Error while converting l") as usize,
                sigma: item
                    .get("std_dev")
                    .and_then(|v| v.as_f64())
                    .expect("Error while converting std_dev"),
                bound: 0.0,
                depth: item.get("d").and_then(|v| v.as_u64()).expect("Error while converting d") as usize,
                alpha: item.get("alpha").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
                source_xs: Source::new(generate_seed(&mut rng)),
                source_xe: Source::new(generate_seed(&mut rng)),
                source_xa: Source::new(generate_seed(&mut rng)),
            };
            params.log_p = (params.p as f64).log2().floor() as usize;
            params.bg_bits = (params.bg_bits as f64).log2().floor() as usize;
            // `std_dev` is a fraction of the torus; poulpy's `NoiseInfos` wants it relative to
            // `2^-k`, so scale by `2^k_rlwe`. `exp2` rather than `1 << k_rlwe`, which is an
            // `i32` shift: it panics past `k_rlwe = 30` in debug and wraps to a *negative*
            // sigma in release, and `MuxLayouts` builds `NoiseInfos` by struct literal, so
            // poulpy's `sigma >= 1.0` check never sees it.
            params.sigma *= (params.k_rlwe as f64).exp2();
            params.bound = 6.0 * params.sigma;

            result.push(params);
        }
    }
    result
}
