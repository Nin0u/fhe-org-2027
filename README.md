# OHVPack : Evaluating Large LUTs in TFHE
This project implements two constructions, Vertical Packing (VPack) and OHVPack, using [Poulpy](https://github.com/poulpy-fhe/poulpy/tree/main) v0.8.1.

## Description
The [TFHE](https://eprint.iacr.org/2018/421.pdf) cryptosystem introduces the Controlled Multiplexer (CMux) that homomorphically selects one of the inputs depending on an encrypted condition, and outputs a new ciphertext indistinguishable from the inputs.
With the CMux, we're able to compute sophisticated functions homomorphically with VPack (Section 5.1 of the [TFHE](https://eprint.iacr.org/2018/421.pdf) paper): a binary CMux tree over the `L/N` polynomials of the table, followed by a rotation.

OHVPack replaces the bulk of that tree with OHCollapse, a first stage driven by one-hot encrypted selectors: it reduces the `L/N` plaintext polynomials to `alpha` ciphertexts using only plaintext x ciphertext products, with no external product and no packing. A CMux tree of depth `log2(alpha)` then selects among those `alpha` candidates.

This project compares the two approaches in practice.

## Project architecture
This project contains 2 "packages" :
- `paramux` : A Python tool that finds secure TFHE parameters for VPack and OHVPack.
- `src` : Contains the Rust implementation of VPack and OHVPack using [Poulpy](https://github.com/poulpy-fhe/poulpy/tree/main) v0.8.1.
