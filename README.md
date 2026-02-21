# ResidualPlannerPlus

A differentially private query answering system based on residual planner decomposition. ResidualPlannerPlus answers sets of marginal queries over tabular data while minimizing total error under differential privacy constraints.

## Features

- **Residual Basis Decomposition** -- decomposes marginal queries into residual components for noise-optimal measurement
- **Sum-of-Variances Optimization** (`ResPlanSum`) -- minimizes total variance across all queries using a closed-form Cauchy--Schwarz solution
- **Max-Variance Optimization** (`ResPlanMax`) -- minimizes the maximum per-query variance via convex programming (CVXPY or Gurobi)
- **Multiple Workload Bases** -- supports Identity (`I`), Prefix-sum (`P`), and Range (`R`) query bases
- **Scalable Reconstruction** -- efficient Kronecker-product-based reconstruction for high-dimensional workloads

## Project Structure

```
ResidualPlannerPlus/
├── resplan/                 # Core library package
│   ├── __init__.py
│   ├── ResPlan.py           # Main planner classes (ResPlanSum, ResPlanMax)
│   ├── utils.py             # Optimization solvers and matrix utilities
│   ├── workload.py          # Workload configuration helpers
│   ├── hdmm_convex.py       # McKenna convex optimization (adapted from HDMM, AGPL-3.0)
│   ├── softmax.py           # Softmax-based optimization
│   ├── bounded_dp.py        # Bounded DP utilities
│   └── parameter.py         # Gurobi license config (not tracked in git)
├── hdmm/                    # HDMM (git submodule, github.com/dpcomp-org/hdmm)
├── tests/                   # Test files
├── experiments/             # Experiment and benchmark scripts
│   └── HDMM_baseline/      # HDMM baseline comparison experiments
│       ├── MaxVar/
│       ├── RMSE/
│       ├── RMSE_query/
│       └── journal/
├── data/                    # Datasets (adult.csv, simple_adult.csv, etc.)
├── docs/                    # Design documents and documentation
├── figures/                 # Paper figures
├── scripts/                 # Utility scripts
├── backup/                  # Legacy code
├── demo.ipynb               # Demo notebook
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone --recurse-submodules https://github.com/<your-org>/ResidualPlannerPlus.git
cd ResidualPlannerPlus
pip install -r requirements.txt
```

If you already cloned without `--recurse-submodules`, initialize the HDMM submodule:

```bash
git submodule update --init
```

**Gurobi (optional):** `ResPlanMax` can use Gurobi as an alternative solver. If you have a Gurobi license, create `resplan/parameter.py` with your credentials:

```python
options = {
    "WLSACCESSID": "<your-access-id>",
    "WLSSECRET": "<your-secret>",
    "LICENSEID": <your-license-id>,
}
```

## Quick Start

```python
import numpy as np
import pandas as pd
from resplan import ResPlanSum

# Define domain sizes and basis types for each attribute
domains = [2, 2, 3]
bases = ['I', 'I', 'I']

# Create the planner
system = ResPlanSum(domains, bases)

# Load data and register it with the planner
data = pd.read_csv("data/simple_adult.csv")
system.input_data(data, ['education', 'marital', 'gender'])

# Add marginal queries (e.g., all 2-way marginals)
import itertools
att = tuple(range(len(domains)))
for subset in itertools.combinations(att, 2):
    system.input_mech(subset)

# Compute noise allocation, measure, and reconstruct
system.get_noise_level()
system.measurement()
system.reconstruction()

# Evaluate mean L1 error
print("Mean Error:", system.get_mean_error(ord=1))
```

## Running Tests

```bash
# Run a specific test
python -m pytest tests/test_resplan.py

# Run all tests in the tests/ directory
python -m pytest tests/
```

## Running Experiments

```bash
# Synthetic dataset benchmarks
python -m experiments.exp_synthetic_dataset

# Large dataset benchmarks (CPS, Adult, Loans)
python -m experiments.exp_large_dataset

# Reconstruction timing
python -m experiments.exp_reconstruct
```

## Dependencies

### HDMM (git submodule)

This project uses [HDMM](https://github.com/dpcomp-org/hdmm) for baseline comparisons. It is included as a git submodule under `hdmm/`.

`resplan/hdmm_convex.py` contains the `McKennaConvex` optimizer adapted from HDMM's `templates.py`, originally authored by Ryan McKenna. This code is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html).

```bash
# Clone with submodule:
git clone --recurse-submodules https://github.com/<your-org>/ResidualPlannerPlus.git

# Or if already cloned:
git submodule update --init
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html). See [LICENSE](LICENSE) for details.

`resplan/hdmm_convex.py` is adapted from [HDMM](https://github.com/dpcomp-org/hdmm) (AGPL-3.0).
