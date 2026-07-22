# DCPuls-LPV-Identification

MATLAB implementations for low-rank subspace identification of linear
parameter-varying (LPV) systems. The repository contains the DCPuls
identification routine together with the simulation cases used to evaluate
it.

## Contents

| File | Purpose |
| --- | --- |
| `idf_yu.m` | DCPuls identification for LPV systems with known scheduling variables |
| `idf_simo.m` | SIMO identification routine used by Case 3 |
| `DCPuls_case1.m` | Case 1: known-scheduling LPV benchmark over several SNR levels |
| `case2.m` | Case 2: fourth-order LPV benchmark |
| `case3.m` | Case 3: multi-output LPV benchmark |
| `mu_unknown.m` | Unknown-scheduling experiment using a known basis-function dictionary |

The experiment scripts include their non-`idf` helper functions locally so
that each case can be inspected and run without additional project files.

## Requirements

- MATLAB R2025b or a compatible recent MATLAB release
- Statistics and Machine Learning Toolbox for `boxplot`
- Parallel Computing Toolbox is optional. `idf_yu` uses a thread pool when
  available and falls back to serial execution otherwise.

No third-party MATLAB packages or external datasets are required.

## Running the experiments

Open MATLAB, set this repository as the current folder, and run one of:

```matlab
DCPuls_case1
case2
case3
mu_unknown
```

The default scripts use 100 Monte Carlo trials. For a quick installation
check, temporarily set `Nmc = 1` near the beginning of the selected script.

`DCPuls_case1.m` and `mu_unknown.m` save summary results as MAT files in the
current folder. These generated files are excluded from version control.

## Reproducibility notes

- The experiments generate independent Monte Carlo data
  on each run. Set `rng(seed)` before their simulation loops when an exactly
  repeatable run is required.
- The scripts report VAF, output NRMSE, and identification time.
- The known-scheduling cases use LPV Ho-Kalman realization after estimating the
  predictor parameters. 

## License

This project is licensed under the MIT License. See `LICENSE` for details.
