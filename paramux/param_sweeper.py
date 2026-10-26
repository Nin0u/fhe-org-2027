# =============================================================================
# |                                                                           |
# |                          PARAM SWEEPERER                                  |
# |                                                                           |
# |                   Using Lattice Estimator, Find secure                    |
# |          noise standard deviation depending on the parameter n/N.         |
# =============================================================================

import argparse
import math
import multiprocessing
import json
from estimator import ND, LWE
from sympy import isprime


TLWE_OUTPUT_FILENAME = "sweeper_output_tlwe.json"
TRLWE_OUTPUT_FILENAME = "sweeper_output_trlwe.json"
N_list = [1024, 2048]

def last_prime(q: int, n: int):
    """
    This function returns the prime modulus Q for NTT.
    We have precomputed its values for usual values N = 1024 and N = 2048.
    """
    if n == 1024 : return 1073707009
    elif n == 2048 : return 1073692673
    else : 
        nBits = q.bit_length()
        m = n << 1
        r = q % m
        q_new = q + 1 - r

        if r < 2:
            q_new -= m
        while not isprime(q_new):
            q_new -= m
        if q_new.bit_length() != nBits:
            raise ValueError("Bit length mismatch")

        return q_new


class ParamSweeper:
    def __init__(self, q):
        self.cpu_count = multiprocessing.cpu_count()
        self.q = q

    def __str__(self):
        return f"ParamSweeper with {self.cpu_count} CPUs and q = {self.q}"

    def __repr__(self):
        return f"ParamSweeper with {self.cpu_count} CPUs and q = {self.q}"

    def sweep(self, n_list: list) -> list:
        """
        For each n given, dichotomic search the stddev such as its security is between 128 and 129 bits.

        :param n_list: The list of LWE/Ring dimensions n/N
        :type n_list: list
        :return: A list containing secure parameters for each n with the associated security.
        :rtype: list
        """

        result = []
        cmod = 1 << self.q

        # Arbitrarily choose starting point e = 1.0 for dichotomy
        # User can modify it for faster convergence.
        e = ND.DiscreteGaussian(1.0)
        for n in n_list :
            s = ND.SparseTernary(p=n // 4, m=n // 4, n=n)
            params = LWE.Parameters(n, cmod, s, e)
            estimate = LWE.dual_hybrid(params)
            security = math.log2(estimate["rop"])
            while math.isinf(security) or security < 128.0 or security >= 129.0:
                m = e.stddev / 2.0
                if security < 128.0:
                    e.stddev += m
                else:
                    e.stddev -= m
                estimate = LWE.dual_hybrid(params)
                security = math.log2(estimate["rop"])

            d = dict()
            d["n"] = n
            d["q"] = cmod
            d["e"] = float(e.stddev / cmod)
            d["security"] = float(security)
            result.append(d)

            print(n, "=>", d["e"])

        return result

if __name__ == "__main__":
    print("N_list = ", N_list)
    parser = argparse.ArgumentParser(description="T(R)LWE Param Sweeper. Given list of LWE/Ring dimensions, computes dual-hybrid 128-bits secured noise standard deviation")
    parser.add_argument(
        "-q",
        type=int,
        default=32,
        help="The exponent of the ciphertext modulus",
    )
    parser.add_argument(
        "-y",
        action="store_true",
        help="Accept to write the outputs in files"
    )
    args = parser.parse_args()

    sweeper = ParamSweeper(args.q)
    print(f"Initilized {sweeper}")
    result = sweeper.sweep(N_list)

    if args.y :
        with open(TLWE_OUTPUT_FILENAME, "w") as f:
            json.dump(result, f, indent=4)
        for elem in result : elem["Q"] = last_prime(elem["n"])
        with open(TRLWE_OUTPUT_FILENAME, "w") as f:
            json.dump(result, f, indent=4)
        print("Parameters written to", TLWE_OUTPUT_FILENAME, "and", TRLWE_OUTPUT_FILENAME)
    else :
        write_res = input("Output as TLWE Params ? [y/n] ")
        while True :
            if write_res.lower() == "y" or write_res.lower() == "yes" :
                with open(TLWE_OUTPUT_FILENAME, "w") as f:
                    json.dump(result, f, indent=4)
                print("Parameters written to", TLWE_OUTPUT_FILENAME)
                break
            elif write_res.lower() == "n" or write_res.lower() == "no" :
                break
            else :
                write_res = input("Output as TLWE Params ? [y/n] ")
        
        write_res = input("Output as TRLWE Params ? [y/n] ")
        while True :
            if write_res.lower() == "y" or write_res.lower() == "yes" :
                # Rename n to N.
                # Compute the parameter Q which is the NTT prime modulus used in OpenFHE.
                for elem in result : elem["Q"] = last_prime(elem["n"])
                with open(TRLWE_OUTPUT_FILENAME, "w") as f:
                    json.dump(result, f, indent=4)
                print("Parameters written to", TRLWE_OUTPUT_FILENAME)
                break
            elif write_res.lower() == "n" or write_res.lower() == "no" :
                break
            else :
                write_res = input("Output as TRLWE Params ? [y/n] ")
