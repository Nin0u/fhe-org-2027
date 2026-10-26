# =============================================================================
# |                                                                           |
# |                    PARAMUX : MULTIPLEXER PARAMS FINDER                    |
# |                                                                           |
# |      Given a set of TFHE parameters (n, q, keydistribution, stddev),      |
# |        find adapted parameters for KeySwtich and External Product         |
# |              (Bks, t, Bg, l) follwing different strategies.               |
# |                                                                           |
# |   Covers OHVPack (Sec. 5) and Vertical Packing, the baseline it is        |
# |   compared against.                                                       |
# =============================================================================

import argparse
import math
import json
import os
from estimator import ND
from itertools import product

# Output tree: <OUTPUT_ROOT>/<mode name>/depth_<d>/s<strategy>.json, written next to this
# script so the location does not depend on the working directory (the Docker image runs
# it from /lattice-estimator).
OUTPUT_ROOT: str = "example_finder"
MODE_NAMES: dict[int, str] = {0: "vertical_packing", 1: "ohvpack"}

ERFC_THRESHOLD: float = 2 ** (-30)
ENABLE_NTT: bool = True
HALF_TORUS = False

class ParamsFinder():
    """Super class that derives to OHVPack/Vertical-Packing ParamFinder"""
    def __init__(
        self,
        filename: tuple[str, str] | None,
        d: int,
        p: int,
    ):
        self.paramsTLWE = {}
        self.paramsTRLWE = {}
        self.d = d
        self.p = p
        self.q = 0

        # Load parameters
        if filename is None:
            self.load_default()
        else:
            self.paramsTLWE = self.load_from_file(filename[0])
            self.paramsTRLWE = self.load_from_file(filename[1], True)
            if self.paramsTLWE is None or self.paramsTRLWE is None:
                print("An error occured. Loading default params")
                self.load_default()

    def load_default(self):
        """
        Hardcoded TLWE and TRLWE 128-bit secured parameters determined with lattice-estimator.
        """
        self.q = 30
        self.paramsTLWE = {
            1024: {"Xs":ND.SparseTernary(p=256, m=256, n=1024), "stddev": 4.027856448374223e-08},
            2048: {"Xs":ND.SparseTernary(p=512, m=512, n=2048), "stddev": 6.914273840974428e-13},
        }

        self.paramsTRLWE = {
            1024: {"Q": 1073707009, "Xs": ND.SparseTernary(p=256, m=256, n=1024), "stddev": 4.027856448374223e-08},
            2048: {"Q": 1073692673, "Xs": ND.SparseTernary(p=512, m=512, n=2048), "stddev": 6.914273840974428e-13},
        }

    def load_from_file(self, filename: str, rlwe: bool = False) -> dict | None:
        """
        Load parameters from a given file.

        @param filename: The path to the file.
        @param rlwe: A Boolean that tells if the file contains LWE or RLWE parameters
        @return A dictionary indexed by n (or N but we call it n anyway).
        """
        result = {}

        # The format of the JSON is a list of dictionnaries
        with open(filename) as f:
            L = json.load(f)

        # Each dictionnary of the file should have a "n" field
        if self.q == 0:
            self.q = int(math.log2(L[0]["q"]))
        for d in L:
            n = int(d["n"])
            p = int(n // 4)
            result[n] = {}
            if self.q != int(math.log2(d["q"])):
                print(f"Error while loading file : self.q = {self.q} vs {int(math.log2(d["q"]))}")
                return None

            if rlwe:
                result[n]["Q"] = int(d["Q"])
            result[n]["Xs"] = ND.SparseTernary(p=p, m=p, n=n)
            result[n]["stddev"] = float(d["e"])
        return result

    def build_params_set(self, q: int) -> list:
        """
        Computes the cartesian product of values of n and N.
        Takes only the results where N == n (specific to our case).

        :param q: The exponent of the ciphertext modulus.
        :type q: int
        :return: The list with all the pairs (n,N) with N > n.
        :rtype: list
        """

        # Here we set Bg and B_KS to 2^10 = 1024 max as we want t/l to be < 6 so Bg^l and B_KS^t don't exceed 64 bits
        params_l = [i for i in range(2, 7)]
        params_Bg = [1 << i for i in range(10, 0, -1)]
        params_t = [i for i in range(2, 7)]
        params_B_KS = [1 << i for i in range(10, 0, -1)]

        L = list(
            product(
                self.paramsTLWE.keys(),
                self.paramsTRLWE.keys(),
                params_t,
                params_B_KS,
                params_l,
                params_Bg,
            )
        )

        # We take results such as log2(Bg) * l is approximatively q
        # But is lesser than q
        result = [
            elem
            for elem in L
            if elem[1] == elem[0]
            and (0.5 <= math.log2(elem[5]) * elem[4] / q and math.log2(elem[5]) * elem[4] / q < 1.0)
            and (0.5 <= math.log2(elem[3]) * elem[2] / q and math.log2(elem[3]) * elem[2] / q < 1.0)
        ]
        return result

    def var_ext(self, params: dict, var_TRGSW: float, var_ms_ntt: float, var_ms_intt: float) -> float:
        """
        Additive noise E_ext of one External Product (paper, Sec. 2.3.4) :

            V_ext = ||mu||^2 * V_C + E_ext
            E_ext = 2 * l * N * (Bg^2 / 4) * V_G + ||mu||^2 * (N + 1) * Bg^(-2l) / 12

        Only the additive part E_ext is returned : the incoming V_C is the variance the
        callers already carry.

        Every selector this tool sizes for is an encrypted ring rotation X^(-c) (or a
        CMux bit for Vertical Packing), so the message is a monomial and ||mu||^2 = 1 --
        which is also why the incoming variance is carried rather than amplified.

        The 2l (and not l) comes from the decomposition being into 2l pieces at k = 1 :
        l per component of the TRLWE ciphertext.
        """
        N = params["N"]
        Bg = params["Bg"]
        l = params["l"]
        return (
            2.0 * l * N * ((Bg * Bg) / 4.0) * (var_TRGSW + var_ms_ntt)
            + var_ms_intt
            + ((N + 1) * (Bg ** (-2 * l))) / 12.0
        )

    def alpha(self, params: dict) -> int:
        """
        OHVPack collapsing factor (paper, Sec. 4.6) :

            alpha = sqrt(nu),   nu = L / N

        rounded to the nearest power of two dividing nu. Since nu = N^(d-1) with N a power
        of two, every power of two up to nu divides it, so the rounding is unconstrained.

        alpha is the only one of the three costs with an interior optimum : it minimizes
        the memory 2*nu/alpha + 2*alpha, balancing the one-hot selectors the client must
        send against the accumulators the first stage returns. Time alone would take
        alpha = 1 (inadmissible : the client would transmit twice the table) and noise
        alone the largest alpha available; inside the useful range the time cost is flat to
        within a few percent.

        The remark of Sec. 4.6 : rounding sqrt(nu) *up* onto a power of N is avoided, and
        we round down instead.

        That rule is inert at the LUT sizes this tool reports on : with nu = N^(d-1) and N
        a power of two, log2(sqrt(nu)) = (d-1) * log2(N) / 2, which for the supported
        (N, d) is an integer, so alpha is already a power of two and nothing is rounded. It
        is implemented because the paper states it and it starts to bite as soon as the
        table is not exactly N^d.
        """
        nu = (params["N"] ** self.d) / params["N"]
        log_n = int(math.log2(params["N"]))
        exponent = math.log2(math.sqrt(nu))
        alpha_pow = max(0, round(exponent))

        # Rounding up landed exactly on a power of N : take the floor instead. `ceil - x`
        # is under 1, so a boundary can only be crossed by landing on it.
        if alpha_pow > exponent and alpha_pow > 0 and alpha_pow % log_n == 0:
            alpha_pow = max(0, math.floor(exponent))

        return 1 << int(alpha_pow)

    def _strategy1(self, elem: dict):
        ''' Closest to Threshold '''
        return abs(ERFC_THRESHOLD - elem["erfc(Dec)"])

    def _strategy2(self, elem: dict):
        ''' Smallest t and l '''
        return elem["t"] + elem["l"]

    def _strategy3(self, elem: dict):
        ''' Smallest Decryption erfc '''
        return elem["erfc(Dec)"]
    
    def objective() : 
        """
        objective function to minimize which is 1 - erf(a / sqrt(2) * sigma)
        We compute the bootstrapping erfc at the same time

        :param params: A Dictionnary containing the parameters on which the variance depends on.
        :type params: dict
        :return: The decryption erfc of the output with the bootstrapping erfc at each d.
        :rtype: tuple(float, list)
        """
        pass

    def find_params(self, strategy: int) -> list:
        """
        Find the best parameters with the given data.
        """

        # We're going to compute the parameters set
        # And extract candidates that satisfy the threshold
        candidates = []
        # Iterate over the parameter sets
        for elem in self.build_params_set(self.q):
            params = {
                "n": elem[0],
                "N": elem[1],
                "t": elem[2],
                "Bks": elem[3],
                "l": elem[4],
                "Bg": elem[5],
            }

            # If the parameters gives an output with an erfc < ERFC_THRESHOLD
            # The set of parameters is considered as a candidate.
            # obj == 0.0 is a legitimate underflow (failure probability below the
            # smallest representable float), i.e. an excellent parameter set, so keep it.
            obj, erfc_boot = self.objective(params)
            if obj < ERFC_THRESHOLD:
                params["erfc(Dec)"] = obj
                params["erfc(Boot)"] = erfc_boot
                candidates.append(params)

        # Here we have a list of candidates for our parameters
        # We choose the one with the smallest t and l
        result = []

        key = self._strategy1 if strategy == 1 else self._strategy2 if strategy == 2 else self._strategy3
        for n in self.paramsTLWE.keys():
            c = [elem for elem in candidates if elem["n"] == n]
            if len(c) == 0:
                print(n, ": No candidate.")
                result.append(None)
            else:
                result.append(min(c, key=key))

        return result

    def find_max_depth(self, params):
        """
        Find the maximum LUT depth d with the given data.
        """
        pass

class ParamsFinderOHVPack(ParamsFinder):
    """
    OHVPack (paper, Sec. 5) : a first stage with OHCollapse 
    followed by a CMux tree of depth ceil(log_2(alpha)) over the alpha ciphertexts it leaves.
    """

    def __init__(
        self,
        filename: tuple[str, str] | None,
        d: int,
        p: int,
    ): ParamsFinder.__init__(self, filename, d, p)

    def objective(self, params: dict) -> tuple:
        """
        objective function to minimize which is 1 - erf(a / sqrt(2) * sigma)
        We compute the bootstrapping erfc at the same time

        In OHVPack, we additionnaly find the collapsing factor alpha cf the paper.

        :param params: A Dictionnary containing the parameters on which the variance
                        depends on : n, N, t, Bks, l, Bg, alpha
        :type params: dict
        :return: The decryption erfc of the output with the bootstrapping erfc at each d.
        :rtype: tuple(float, list)
        """

        # NTT and INTT moduli
        Q = self.paramsTRLWE[params["N"]]["Q"]
        q = 1 << self.q

        # Variance formulas
        var_TRLWE = self.paramsTRLWE[params["N"]]["stddev"] ** 2
        var_TRGSW = var_TRLWE
        var_modswitch_ntt = (params["N"] + 1.0) / (12.0 * Q * Q) if ENABLE_NTT else 0
        var_modswitch_intt = (params["N"] + 1.0) / (12.0 * q * q) if ENABLE_NTT else 0
        var_ext = self.var_ext(params, var_TRGSW, var_modswitch_ntt, var_modswitch_intt)
        var_bs = (params["N"] + 1) / (48.0 * params["N"] * params["N"])

        # Collapsing factor and the number of groups the first stage sums (paper, Sec. 4.6)
        nu = (params["N"] ** self.d) / params["N"]
        alpha = self.alpha(params)

        # Depth = 1 : Plain x TRLWE 
        variance = (float(nu) / alpha) * (
            params["N"] * ((self.p >> 1) ** 2) * (var_TRLWE + var_modswitch_ntt)
        ) + (var_modswitch_intt)

        # Depth > 1 : one external product per level of the log2(alpha)-deep CMux tree
        params["d"] = self.d
        erfc_boot = []
        for _ in range(int(math.ceil(math.log2(alpha)))) :
            variance += var_ext
            # Add Bootstrapping's erfc
            erfc_boot.append(math.erfc(1.0 / ((4 if HALF_TORUS else 2) * math.sqrt(2 * (variance + var_bs)) * self.p)))

        # Here variance is computed and we want to minimize erfc(a / sqrt(2) * stddev)
        # a is the decryption radius which equals to
        # 1 / (2 * p) on the full torus
        # 1 / (4 * p) on the half torus
        params["alpha"] = alpha
        return((
            math.erfc(1.0 / ((4 if HALF_TORUS else 2)* math.sqrt(2 * variance) * self.p)),
            erfc_boot)
        )
        
    def find_max_depth(self, params):
        if params is None : return None
        # NTT and INTT moduli
        Q = self.paramsTRLWE[params["N"]]["Q"]
        q = 1 << self.q

        # Variance formulas
        var_TRLWE = self.paramsTRLWE[params["N"]]["stddev"] ** 2
        var_TRGSW = var_TRLWE
        var_modswitch_ntt = (params["N"] + 1.0) / (12.0 * Q * Q) if ENABLE_NTT else 0
        var_modswitch_intt = (params["N"] + 1.0) / (12.0 * q * q) if ENABLE_NTT else 0
        var_ext = self.var_ext(params, var_TRGSW, var_modswitch_ntt, var_modswitch_intt)
        var_bs = (params["N"] + 1) / (48.0 * params["N"] * params["N"])

        # Collapsing factor and the number of groups the first stage sums (paper, Sec. 4.6)
        nu = (params["N"] ** self.d) / params["N"]
        alpha = self.alpha(params)

        # Depth = 1 : Plain x TRLWE 
        variance = (float(nu) / alpha) * (
            params["N"] * ((self.p >> 1) ** 2) * (var_TRLWE + var_modswitch_ntt)
        ) + (var_modswitch_intt)
        erfc = math.erfc(1.0 / ((4 if HALF_TORUS else 2) * math.sqrt(2 * variance) * self.p))

        d = 0
        while erfc < ERFC_THRESHOLD:
            d += 1
            # Each extra level of the CMux tree adds one external product: no packing.
            variance += var_ext
            erfc = math.erfc(1.0 / ((4 if HALF_TORUS else 2) * math.sqrt(2 * variance) * self.p))

        params["max_depth"] = (1 + d) if d > 0 else 0
        return 0

class ParamsFinderVerticalPacking(ParamsFinder):
    def __init__(
        self,
        filename: tuple[str, str] | None,
        d: int,
        p: int,
    ): ParamsFinder.__init__(self, filename, d, p)

    def objective(self, params: dict) -> tuple:
        """
        objective function to minimize which is 1 - erf(a / sqrt(2) * sigma)
        We compute the bootstrapping erfc at the same time

        :param params: A Dictionnary containing the parameters on which the variance
                        depends on : n, N, t, Bks, l, Bg, alpha
        :type params: dict
        :return: The decryption erfc of the output with the bootstrapping erfc at each d.
        :rtype: tuple(float, list)
        """

        # NTT and INTT moduli
        Q = self.paramsTRLWE[params["N"]]["Q"]
        q = 1 << self.q

        # Variance formulas
        var_TRLWE = self.paramsTRLWE[params["N"]]["stddev"] ** 2
        var_TRGSW = var_TRLWE
        var_modswitch_ntt = (params["N"] + 1.0) / (12.0 * Q * Q) if ENABLE_NTT else 0
        var_modswitch_intt = (params["N"] + 1.0) / (12.0 * q * q) if ENABLE_NTT else 0
        var_ext = self.var_ext(params, var_TRGSW, var_modswitch_ntt, var_modswitch_intt)
        var_bs = (params["N"] + 1) / (48.0 * params["N"] * params["N"])
        lut_size = params["N"] ** self.d

        variance = var_TRLWE + var_modswitch_ntt
        erfc_boot = []

        # Selection step
        for _ in range(int(math.log2(lut_size / params["N"]))) :
            variance = max(variance, var_TRGSW) + var_ext
            # Add Bootstrapping's erfc
            erfc_boot.append(math.erfc(1.0 / ((4 if HALF_TORUS else 2) * math.sqrt(2 * (variance + var_bs)) * self.p)))

        # Rotation step
        params["d"] = self.d
        variance += var_ext
        erfc_boot.append(math.erfc(1.0 / ((4 if HALF_TORUS else 2) * math.sqrt(2 * (variance + var_bs)) * self.p)))

        return (
            math.erfc(1.0 / ((4 if HALF_TORUS else 2)* math.sqrt(2 * variance) * self.p)),
            erfc_boot
        )
        
    def find_max_depth(self, params):
        if params is None : return None
        # NTT and INTT moduli
        Q = self.paramsTRLWE[params["N"]]["Q"]
        q = 1 << self.q

        var_TRLWE = self.paramsTRLWE[params["N"]]["stddev"] ** 2
        var_TRGSW = var_TRLWE
        var_modswitch_ntt = (params["N"] + 1.0) / (12.0 * Q * Q) if ENABLE_NTT else 0
        var_modswitch_intt = (params["N"] + 1.0) / (12.0 * q * q) if ENABLE_NTT else 0
        var_ext = self.var_ext(params, var_TRGSW, var_modswitch_ntt, var_modswitch_intt)

        log_n = int(math.log2(params["N"]))

        # Depth 1 (LUT of size N): only the X^{-r} ring-rotation external product runs.
        variance = max(var_TRLWE + var_modswitch_ntt, var_TRGSW) + var_ext
        erfc = math.erfc(1.0 / ((4 if HALF_TORUS else 2) * math.sqrt(2 * variance) * self.p))

        d = 1
        while erfc < ERFC_THRESHOLD:
            d += 1
            # Each extra d multiplies the LUT size by N, i.e. log2(N) more CMux-tree
            # external products on the noise path (additive, as in objective()).
            for _ in range(log_n):
                variance = max(variance, var_TRGSW) + var_ext
            erfc = math.erfc(1.0 / ((4 if HALF_TORUS else 2) * math.sqrt(2 * variance) * self.p))

        # The loop stops at the first d that fails to decrypt.
        params["max_depth"] = d - 1
        return 0

# =============================================================================
# |                                                                           |
# |                                 MAIN                                      |
# |                                                                           |
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paramux : TFHE Multiplexer Parameters Finder")
    parser.add_argument(
        "-d",
        type=int,
        help="log_N Lut size",
        required=True,
        choices=[2, 3],
    )
    parser.add_argument(
        "-p",
        type=int,
        help="The Plaintext Modulus",
        required=True,
        choices=[2, 4, 8, 16, 32, 64, 128, 256]
    )
    parser.add_argument(
        "-f",
        nargs=2,
        metavar=("tlwe_param_file", "trlwe_param_file"),
        type=str,
        help="Files to TLWE and TRLWE parameters ouputted by param_sweeper.",
    )
    parser.add_argument(
        "-s",
        type=int,
        default=3,
        help="Define the strategy to use to select the right parameters."
        "1 = erfc closest to threshold."
        "2 = Smallest t and l(scaling with N)."
        "3 = Smallest erfc.",
        choices=[1, 2, 3],
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        required = True,
        choices=[0, 1],
        help="The mode represents the construction we want."
        "0 = Vertical Packing"
        "1 = OHVPack"
    )
    args = parser.parse_args()

    print("==== Static Values ====")
    print(f"\terfc Threshold : 2^({math.log2(ERFC_THRESHOLD)})\n\
            \tNTT : {ENABLE_NTT}\n\
            \tHalf Torus : {HALF_TORUS}"
        )
    print("=======================", end="\n\n")

    if args.mode == 0 :
        print("Evaluating Vertical Packing parameters")
        finder = ParamsFinderVerticalPacking(filename=args.f, d=args.d, p=args.p)
    else :
        print("Evaluating OHVPack parameters")
        finder = ParamsFinderOHVPack(filename=args.f, d=args.d, p=args.p)
    result = finder.find_params(args.s)

    # Once params found, build a JSON  with all the parameters for benchmarking
    output_json = []
    for elem in result:
        if finder.find_max_depth(elem) is None:
            exit(0)
        erfc_dec = elem.pop("erfc(Dec)", None)
        erfc_boot = elem.pop("erfc(Boot)", None)
        max_depth = elem.pop("max_depth", None)
        d = elem.pop("d", None)
        print(elem)
        print(f"\tstddev : {finder.paramsTLWE[elem["n"]]["stddev"]}")
        print(f"\terfc(Dec) : {erfc_dec} => log2 = " + ("< -1024" if erfc_dec == 0 else str(math.log2(erfc_dec))))
        print("\terfc(Boot):")
        for e in erfc_boot:
            if e == 0:
                print(f"\t\t{e} => log2 < -1024")
            else:
                print(f"\t\t{e} => log2 = {math.log2(e)}")
        print(f"\ttarget d : {d}")
        print(f"\tmax_depth : {max_depth}")
        print("\n")

        elem["p"] = args.p
        elem["d"] = d
        elem["powq"] = finder.q
        elem["Q"] = finder.paramsTRLWE[elem["N"]]["Q"]
        elem["std_dev"] = finder.paramsTLWE[elem["N"]]["stddev"]
        output_json.append(elem)

    # The output location is derived from the run rather than asked for: the mode names the
    # construction folder, the depth the one below it, and the strategy the file itself.
    # Anchored on this script's directory so the tree is the repository's `example_finder/`
    # whatever the working directory is (the Docker image runs from /lattice-estimator).
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        OUTPUT_ROOT,
        MODE_NAMES[args.mode],
        "depth_" + str(args.d),
    )
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "s" + str(args.s) + ".json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
        f.write("\n")
    print("Data written in " + output_file)