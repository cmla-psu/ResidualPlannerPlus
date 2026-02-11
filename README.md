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
│   ├── hdmm_convex.py       # McKenna convex optimization strategy
│   ├── softmax.py           # Softmax-based optimization
│   ├── bounded_dp.py        # Bounded DP utilities
│   └── parameter.py         # Gurobi license config (not tracked in git)
├── tests/                   # Test files
├── experiments/             # Experiment and benchmark scripts
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
git clone https://github.com/<your-org>/ResidualPlannerPlus.git
cd ResidualPlannerPlus
pip install -r requirements.txt
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

## License

MIT License. See [LICENSE](LICENSE) for details.
