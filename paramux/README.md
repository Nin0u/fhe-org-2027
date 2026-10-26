# Paramux : Multiplexer Parameters Finder
Paramux finds secure TFHE parameters for OHVPack (Section 5) and for Vertical Packing, the standard TFHE baseline it is compared against, using [lattice-estimator](https://github.com/malb/lattice-estimator/tree/main/estimator).

## Requirements
We highly recommend you to run the code inside a docker container so that you only have to install [Docker](https://docs.docker.com/).

If you still want to run the code without Docker, you'll need :
- `git`
- `python` >= `3.2` and `python-venv`
- [Sagemath](https://doc.sagemath.org/html/en/installation/index.html) with its dependencies.

## Setup with Docker
At the root of this directory execute `run.sh` or the following commands :
```bash
docker build -t paramux .
docker run --rm -it paramux:latest
```

You should have an interactive terminal inside the docker container in which you can run the code.

## Setup without Docker
Make sure you have installed Sagemath (this is the hardest part of the setup).

- Clone `lattice-estimator` :
```bash
git clone https://github.com/malb/lattice-estimator.git
```

- Copy The source `.py` files to the root of `lattice-estimator`
- Install the requirements in `requirements.txt`
```bash
sage -pip install -r /lattice-estimator/requirements.txt
```

## About the programs
No matter the setup, you should now be able to run the `sage` command.
Run it on the source `.py` files with `-h`(help) option :
```bash
sage ./param_sweeper.py -h
sage ./param_finder.py -h
```
Here are some examples of commands you can run :
```bash
    sage ./param_sweeper.py -q 30 -y
    sage ./param_finder.py -d 2 -p 16 -m 1 -s 1 -f sweeper_output_tlwe.json sweeper_output_trlwe.json
```

### Modes
`param_finder.py` selects the construction with `-m` :

| `-m` | construction | paper |
| ---- | ------------ | ----- |
| `0`  | Vertical Packing | Section 2.3.7 |
| `1`  | OHVPack | Section 5 |

`-d` is the LUT's decomposition depth, i.e. `L = N^d`, and is restricted to `2` and `3`.

## Output Example
We give example JSON files ouputted from `param_sweeper.py` in `example_sweeper/`. These files are used as input for `param_finder.py`.

`param_finder.py` no longer asks whether to write its result: the output path is derived from the run and is always

```
example_finder/<vertical_packing|ohvpack>/depth_<d>/s<strategy>.json
```

relative to this directory, whatever the working directory is. `example_finder/` therefore contains output examples that are ready-to-use in our OpenFHE and Poulpy implementations. We used the example parameter files for our benchmarks.
