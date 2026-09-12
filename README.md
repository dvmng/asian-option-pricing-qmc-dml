# Pricing and Hedging Asian Options on GPU Compute: RQMC and Differential Machine Learning

Code, datasets, trained models and results for a master's thesis in quantitative finance.

The study prices arithmetic Asian options on GPU compute capacity under a log-Ornstein–Uhlenbeck model. Randomized quasi-Monte Carlo (RQMC) produces price and Delta labels for two neural network approaches: a price-only multilayer perceptron (MLP) and differential machine learning (DML). The experiments compare out-of-sample accuracy, errors near fixing dates and extreme scenarios, and local and dynamic hedging performance.

**[Read the thesis (PDF, in Spanish)](thesis/main.pdf)**

## Repository contents

| Directory | Contents |
|---|---|
| `data/raw/`, `data/processed/` | Market observations and processed data |
| `data/frozen/` | Pricing datasets, training order and validation splits used in the thesis |
| `models/final/` | The 40 final model checkpoints and their training records |
| `results/` | Calibration, pricing, model selection, error diagnostics and hedging results |
| `protocols/` | Fixed experiment settings |
| `src/` | Pricing, simulation and learning code |
| `experiments/` | Eight experiment entry points |
| `checks/` | Integrity checks, regression comparisons and additional diagnostics |
| `archive/recovered_scripts/` | Recovered analysis scripts and provenance records |
| `figures/`, `tables/` | Figures and tables used in the thesis |
| `thesis/` | Compiled thesis PDF |
| `docs/` | Reproduction notes and documented corrections, in Spanish |

New runs write to `data/reproduced/`, `models/reproduced/` and `results/reproduced/`. The frozen datasets, final checkpoints and published results are kept as the reference for comparison.

## Setup

Python 3.11 or later is required. The final evaluation used PyTorch 2.12.1 with CUDA 12.6. A CUDA GPU is recommended for regenerating the RQMC dataset and training all 40 models.

Run these commands from the repository root:

```shell
python -m pip install -r requirements.txt
python -m pip install -e . --no-build-isolation
```

For GPU runs, install a PyTorch build compatible with your CUDA environment.

## Check the saved results

```shell
python main.py --checks
python main.py --layout
```

`--checks` verifies the frozen pricing dataset, the 40-model grid, the hedging protocols and the main result files. `--layout` displays the project structure. Neither command retrains the models.

To rebuild the thesis figures and tables from the saved results:

```shell
python main.py --outputs
```

## Experiments

The workflow runs from market data and Log-OU calibration through RQMC label generation, neural network training, evaluation and hedging.

| Stage | Script in `experiments/` | Purpose |
|---:|---|---|
| 1 | `01_market_data.py` | Inspect saved market data or fetch a new Ornn observation |
| 2 | `02_market_analysis.py` | Compare market observations across GPU types |
| 3 | `03_model_calibration.py` | Calibrate Log-OU dynamics under the physical measure |
| 4 | `04_pricing_dataset.py` | Generate RQMC price and Delta labels |
| 5 | `05_ml_training.py` | Train MLP and DML models |
| 6 | `06_ml_evaluation.py` | Evaluate on the test and reference datasets |
| 7 | `07_local_hedging.py` | Run local hedging experiments |
| 8 | `08_dynamic_hedging.py` | Run the dynamic hedging stress test |

Each stage has its own command-line help. For example:

```shell
python main.py --experiment 5 -- --help
```

## Reproduce the final experiment

Generate the pricing dataset:

```shell
python main.py --experiment 4 -- --preset tfm --device cuda --output data/reproduced/pricing_dataset
```

Train the full grid of 40 models:

```shell
python main.py --experiment 5 -- --data-dir data/reproduced/pricing_dataset --prepared-dir data/reproduced/ml_prepared --output-dir models/reproduced --full-grid --device cuda
```

If `data/reproduced/ml_prepared/train_order.npy` is missing, the training order is rebuilt using the protocol seed and checked against the frozen order.

Evaluate the new models:

```shell
python main.py --experiment 6 -- --data-dir data/reproduced/pricing_dataset --results-dir models/reproduced --output-dir results/reproduced/ml_evaluation --device cuda --save-predictions
```

Run both hedging experiments with those models:

```shell
python main.py --experiment 7 -- --models-dir models/reproduced --output-dir results/reproduced/hedging/local --device cuda
python main.py --experiment 8 -- --models-dir models/reproduced --output-dir results/reproduced/hedging/dynamic_forward --device cuda
```

Compare the new metrics with the saved results:

```shell
python checks/regression_check.py results/ml/final_evaluation/aggregate_metrics.csv results/reproduced/ml_evaluation/aggregate_metrics.csv
```

## Data and model selection

The market observations used in the thesis are included in `data/raw/` and `data/processed/`, so the analysis does not depend on a web source returning the same historical data later. Stage 1's `--refresh-ornn` option saves new observations to `data/staging/`.

Hyperparameter selection and differential-loss weighting records are stored in `results/model_selection/` and `data/frozen/model_selection_validation/`. The commands above reproduce the final experiment with its fixed protocol; the full exploratory hyperparameter search is not rerun.

## Additional checks and corrections

[Reproduction notes](docs/REPRODUCIBILIDAD.md) document the four recovered scripts, their original paths and the verified environment. To run the recovered analyses and the forward present-value diagnostic:

```shell
python -B checks/reproduce_recovered.py
python -B checks/forward_pv_diagnostic.py
```

These outputs are saved under `results/reproduced/`. The thesis retains the original forward-hedging results in its main tables and reports the present-value discounting correction separately.
