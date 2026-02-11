# ResPlan.py Documentation

## Overview

ResPlan.py implements a **Residual Planner** system for differential privacy mechanisms. It provides a framework for answering statistical queries (marginals) on datasets while maintaining privacy through carefully calibrated noise addition. The system uses residual basis matrices to optimize the privacy-utility tradeoff.

## Core Concepts

### Residual Planning
The system decomposes complex marginal queries into simpler residual measurements, then reconstructs the desired answers. This approach allows for:
- Better privacy budget allocation
- Reduced noise in final answers
- Support for various query workloads (prefix sums, range queries, identity)

### Key Mathematical Foundation
- **Marginal Queries**: Statistical summaries over subsets of attributes
- **Residual Matrices**: Transform data to reduce correlation and optimize noise
- **Kronecker Products**: Combine transformations across multiple attributes
- **Variance Optimization**: Minimize total variance subject to privacy constraints

## Essential Classes

### 1. MargMech (Marginal Mechanism)
**Purpose**: Represents a marginal query mechanism that stores query results and metadata.

**Key Attributes**:
- `domains`: Domain sizes for each attribute
- `att`: Tuple of attributes involved in the marginal
- `noisy_answer_vector`: Final noisy query answers
- `variance`: Variance of the mechanism
- `csv_data`: Original data for the marginal

**Key Methods**:
```python
def input_noisy_answer_vector(self, answer)  # Store noisy results
def get_noisy_answer_vector(self)            # Retrieve noisy results
def input_variance(self, var)                # Set variance
def output_variance(self)                    # Get variance
```

### 2. ResMech (Residual Mechanism)
**Purpose**: Implements residual measurements on data subsets using residual matrices.

**Key Attributes**:
- `domains`: Domain sizes for attributes
- `att`: Attributes in this residual mechanism
- `res_mat_list`: List of residual matrices for each attribute
- `noise_level`: Calibrated noise level for this mechanism

**Key Methods**:
```python
def measure(self, input_base=None, debug=False, n_runs=None)
    # Performs measurement with noise addition
    # Returns covariance matrix if debug=True with multiple runs

def get_recon_answer(self, mat_list)
    # Reconstructs answer using provided transformation matrices

def measure_gaussian(self, n_runs, domains, debug=False)
    # Special method for zero-dataset simulation
    # Generates Gaussian noise with proper scaling
```

### 3. ResidualPlanner (Abstract Base Class)
**Purpose**: Base class defining the interface for residual planning systems.

**Key Attributes**:
- `domains`: Domain sizes for all attributes
- `bases`: Basis types for each attribute ('P'=Prefix, 'I'=Identity, 'R'=Range)
- `marg_dict`: Dictionary mapping attribute tuples to MargMech objects
- `res_dict`: Dictionary mapping attribute tuples to ResMech objects
- `residual_matrix`: Precomputed residual matrices for each domain size

**Key Methods**:
```python
def input_mech(self, att)                    # Add a marginal query
def measurement(self)                        # Perform all measurements
def reconstruction(self, debug=False)        # Reconstruct all marginals
def get_mean_error(self, ord=1)             # Calculate reconstruction error
```

### 4. ResPlanSum (Concrete Implementation)
**Purpose**: Implements residual planning for sum queries with variance minimization.

**Key Features**:
- Optimizes total variance across all queries
- Uses Cauchy-Schwarz optimization for noise allocation
- Supports multiple basis types (Prefix, Identity, Range)

**Key Methods**:
```python
def preprocessing(self)
    # Precomputes residual matrices and variance coefficients

def get_noise_level(self)
    # Solves optimization problem to find optimal noise levels
    # Uses find_var_sum_cauchy() from utils

def get_random_noise_level(self, seed=None)
    # Alternative: assigns random noise levels (for testing)

def get_zero_noise_level(self)
    # Alternative: sets all noise to zero (for baseline)
```

### 5. ResPlanMax (Concrete Implementation)
**Purpose**: Implements residual planning for max queries with per-query variance bounds.

**Key Features**:
- Bounds variance for each individual query
- Uses convex optimization (CVXPY or Gurobi)
- More complex constraint handling

## Essential Functions

### Utility Functions
```python
def find_rational_approximation(x, max_denominator=1000)
    # Converts float to rational number for exact arithmetic

def rational_approx_sqrt(x, max_denominator=100)
    # Finds rational approximation of square root
    # Returns (numerator, denominator) tuple
```

### Test Functions
```python
def test_simple_adult()
    # Tests on simplified adult dataset (3 attributes)
    # Good for understanding basic functionality

def test_paper()
    # Implements the range queries example from research paper
    # Demonstrates range query capabilities

def test_age_income()
    # Tests on age-income dataset with prefix queries
    # Shows handling of different domain sizes
```

## Detailed Workflow

### 1. System Initialization - Deep Dive

```python
domains = [2, 3]  # Domain sizes: attribute 0 has 2 values, attribute 1 has 3 values
bases = ['I', 'R']  # Basis types: Identity for attr 0, Range for attr 1
system = ResPlanSum(domains, bases)
```

**What happens during initialization:**

#### A. Base Class Setup (`ResidualPlanner.__init__`)
```python
self.domains = domains           # Store domain sizes [2, 3]
self.bases = bases              # Store basis types ['I', 'R']
self.num_of_mech = 0           # Counter for marginal mechanisms
self.num_of_res = 0            # Counter for residual mechanisms

# Key data structures initialized as empty
self.marg_dict = {}            # Maps attribute tuples to MargMech objects
self.res_dict = {}             # Maps attribute tuples to ResMech objects
self.marg_index = {}           # Maps attribute tuples to mechanism IDs
self.res_index = {}            # Maps attribute tuples to residual IDs
self.id2res = {}               # Maps residual IDs back to attribute tuples

# Coefficient storage for optimization
self.pcost_coeff = {}          # Privacy cost coefficients per residual
self.var_coeff_sum = defaultdict(int)  # Variance coefficients per residual
```

#### B. Preprocessing (`ResPlanSum.preprocessing()`)
For each attribute domain, the system precomputes essential matrices:

```python
for i, k in enumerate(self.domains):  # i=0,k=2 then i=1,k=3
    base = self.bases[i]  # 'I' for attr 0, 'R' for attr 1
    
    # Find residual basis matrices using utility functions
    Bs, R, Us, Ur = find_residual_basis_sum(k, base)
    
    if base == 'I':
        # For Identity basis, use subtract matrix
        self.residual_matrix[k] = subtract_matrix(k)
    else:
        # For other bases, use computed residual matrix
        self.residual_matrix[k] = R
    
    # Precompute pseudo-inverse for reconstruction
    self.residual_pinv[k] = np.linalg.pinv(self.residual_matrix[k])
    
    # Create one-vector for sum operations
    self.one_mat[k] = np.ones([k,1])/k
    
    # Store privacy and variance coefficients
    self.pcost_sum[i] = 1
    self.pcost_res[i] = np.max(np.diag(R.T @ R))
    self.var_sum[i] = np.trace(Us @ Us.T)
    self.var_res[i] = np.trace(Ur @ Ur.T)
```

**Example with domains=[2,3], bases=['I','R']:**
- For attribute 0 (domain=2, basis='I'):
  - `residual_matrix[2]` = 2×2 subtract matrix
  - `residual_pinv[2]` = pseudo-inverse of subtract matrix
  - `one_mat[2]` = [0.5, 0.5]ᵀ
- For attribute 1 (domain=3, basis='R'):
  - `residual_matrix[3]` = 3×3 range-optimized matrix
  - `residual_pinv[3]` = pseudo-inverse for reconstruction
  - `one_mat[3]` = [0.33, 0.33, 0.33]ᵀ

### 2. Data Input - Deep Dive
```python
data = pd.read_csv("dataset.csv")
system.input_data(data, col_names)
```

**What happens during data input:**
```python
def input_data(self, data, col_names):
    self.data = data           # Store DataFrame
    self.col_names = col_names # Store column names for attribute mapping
```

This step simply stores the data reference. **No actual processing occurs yet** - data processing happens during measurement phase.

### 3. Query Specification - Deep Dive

```python
# Add marginal queries
system.input_mech((0,))     # 1-way marginal on attribute 0
system.input_mech((1,))     # 1-way marginal on attribute 1  
system.input_mech((0,1))    # 2-way marginal on attributes 0,1
```

**What happens for each `input_mech()` call:**

#### Step A: Create Marginal Mechanism
```python
def input_mech(self, att):  # att = (0,) for first call
    # Create marginal mechanism object
    mech = MargMech(self.domains, att)
    
    # Calculate number of queries for this marginal
    cur_domains = [self.domains[at] for at in att]  # [2] for att=(0,)
    mech.num_queries = np.prod(cur_domains)         # 2 queries
    
    # Store in dictionaries
    self.marg_dict[att] = mech              # marg_dict[(0,)] = mech
    self.marg_index[att] = self.num_of_mech # marg_index[(0,)] = 0
    self.num_of_mech += 1                   # num_of_mech = 1
```

#### Step B: Generate Required Residual Mechanisms
For marginal `att=(0,)`, the system needs all subsets: `[(), (0,)]`

```python
att_subsets = all_subsets(att)  # [(), (0,)] for att=(0,)

for subset in att_subsets:
    if subset not in self.res_dict:  # Create if doesn't exist
        # Calculate privacy cost coefficient
        pcost_res_list = [self.pcost_res[at] for at in subset]
        pcost_coeff = np.prod(pcost_res_list)
        self.pcost_coeff[subset] = pcost_coeff
        
        # Create residual mechanism
        res_mech = ResMech(self.domains, subset, self.residual_matrix)
        
        # Store residual mechanism
        self.res_dict[subset] = res_mech
        self.res_index[subset] = self.num_of_res
        self.id2res[self.num_of_res] = subset
        self.num_of_res += 1
```

#### Step C: Calculate Variance Coefficients
For each subset, calculate how it contributes to the variance of the target marginal:

```python
for subset in att_subsets:  # subset in [(), (0,)]
    # Variance from residual measurements
    var_res_list = [self.var_res[at] for at in subset]
    var_res_query = np.prod(var_res_list)
    
    # Variance from sum measurements (attributes not in subset)
    var_sum_list = []
    for c in att:  # c in [0] for att=(0,)
        if c not in subset:
            var_sum_list.append(self.var_sum[c])
    var_sum_query = np.prod(var_sum_list)
    
    # Total variance coefficient for this subset
    self.var_coeff_sum[subset] += var_sum_query * var_res_query
```

**Example: Processing `input_mech((0,1))`**

1. **Subsets generated**: `[(), (0,), (1,), (0,1)]`
2. **New residual mechanisms created**: Any subset not already in `res_dict`
3. **Variance coefficients calculated**: Each subset gets a coefficient showing how its noise affects the (0,1) marginal

**Complete state after three `input_mech()` calls:**

```python
# Marginal mechanisms
marg_dict = {
    (0,): MargMech_for_attr_0,
    (1,): MargMech_for_attr_1, 
    (0,1): MargMech_for_attr_0_1
}

# Residual mechanisms (all unique subsets)
res_dict = {
    (): ResMech_empty,      # For sum query
    (0,): ResMech_attr_0,   # For attribute 0 residual
    (1,): ResMech_attr_1,   # For attribute 1 residual
    (0,1): ResMech_attr_0_1 # For 2-way residual
}

# Variance coefficients (how each residual affects each marginal)
var_coeff_sum = {
    (): coeff_for_empty_subset,
    (0,): coeff_for_attr_0_subset,
    (1,): coeff_for_attr_1_subset,
    (0,1): coeff_for_both_attrs_subset
}
```

### 4. Understanding the Subset Generation Logic

**Why does marginal `(0,1)` need subsets `[(), (0,), (1,), (0,1)]`?**

This is the **inclusion-exclusion principle** for residual reconstruction:
- Answer for (0,1) = Sum over all subsets of measurements
- Each subset represents a different "perspective" on the data
- The reconstruction formula combines these perspectives optimally
- More subsets = more measurements = better noise averaging but higher privacy cost

### 5. ResMech Initialization Details

When `ResMech(self.domains, subset, self.residual_matrix)` is called, here's what happens internally:

```python
class ResMech:
    def __init__(self, domains, att, residual_matrix):
        self.domains = domains              # [2, 3] 
        self.att = att                      # e.g., (0,) or (0,1)
        self.res_mat_list = []             # Will store residual matrices
        
        # Build core matrix by selecting appropriate residual matrices
        self.get_core_matrix(residual_matrix)
        
        # Initialize state variables
        self.noise_level = None            # Will be set during optimization
        self.noisy_answer_vector = None    # Will store measurement results
        self.non_noisy_vector = None       # Will store true values (debug)
```

#### Core Matrix Construction (`get_core_matrix`)

```python
def get_core_matrix(self, residual_matrix, using_identity=False):
    att_set = set(list(self.att))  # Convert (0,1) to {0, 1}
    
    for i in range(0, self.num_att):  # i = 0, 1 for 2 attributes
        att_size = self.domains[i]    # 2 for attr 0, 3 for attr 1
        
        if i in att_set:
            # This attribute is in the residual measurement
            if using_identity:
                res_mat = subtract_matrix(att_size)
            else:
                res_mat = residual_matrix[att_size]  # Use precomputed matrix
            self.res_mat_list.append(res_mat)
```

**Example: Creating ResMech for subset (0,1)**
- `att_set = {0, 1}`
- For i=0: att_size=2, i in att_set → append `residual_matrix[2]`
- For i=1: att_size=3, i in att_set → append `residual_matrix[3]`
- Result: `res_mat_list = [residual_matrix[2], residual_matrix[3]]`

**Example: Creating ResMech for subset (1,)**
- `att_set = {1}`
- For i=0: att_size=2, i not in att_set → skip
- For i=1: att_size=3, i in att_set → append `residual_matrix[3]`
- Result: `res_mat_list = [residual_matrix[3]]`

**Example: Creating ResMech for subset ()**
- `att_set = {}` (empty)
- For i=0: att_size=2, i not in att_set → skip
- For i=1: att_size=3, i not in att_set → skip
- Result: `res_mat_list = []` (empty list)

### 6. MargMech Initialization Details

```python
class MargMech:
    def __init__(self, domains, att, var_bound=1.0):
        self.domains = domains                    # [2, 3]
        self.att = att                           # e.g., (0,1)
        self.num_att = len(domains)              # 2
        
        # Calculate how many queries this marginal represents
        cur_domains = [self.domains[at] for at in att]  # [2, 3] for att=(0,1)
        self.num_queries = np.prod(cur_domains)         # 2 × 3 = 6 queries
        
        # Initialize storage for results
        self.noisy_answer_vector = None          # Final noisy answers
        self.non_noisy_vector = None            # True answers (debug)
        self.variance = None                    # Total variance
        self.csv_data = None                    # Original data
```

**Query Count Examples:**
- `att=(0,)`: domains=[2] → num_queries = 2 (one for each value of attribute 0)
- `att=(1,)`: domains=[3] → num_queries = 3 (one for each value of attribute 1)  
- `att=(0,1)`: domains=[2,3] → num_queries = 6 (one for each combination)
- `att=()`: domains=[] → num_queries = 1 (just the total count)

### 7. Memory and Complexity Analysis

After processing `input_mech((0,)), input_mech((1,)), input_mech((0,1))`:

**Memory Usage:**
```python
# Marginal mechanisms: 3 objects
marg_dict = {(0,): MargMech, (1,): MargMech, (0,1): MargMech}

# Residual mechanisms: 4 objects  
res_dict = {
    (): ResMech_empty,      # res_mat_list = []
    (0,): ResMech_attr0,    # res_mat_list = [matrix_2x2]
    (1,): ResMech_attr1,    # res_mat_list = [matrix_3x3]  
    (0,1): ResMech_both     # res_mat_list = [matrix_2x2, matrix_3x3]
}
```

**Total Queries:** 2 + 3 + 6 = 11 queries across all marginals
**Total Residual Measurements:** 4 different residual mechanisms
**Matrix Storage:** 2×2 and 3×3 residual matrices plus their pseudo-inverses

### 4. Noise Calibration
```python
obj = system.get_noise_level()  # Optimize noise allocation
```

### 5. Measurement
```python
system.measurement()  # Perform noisy measurements
```

### 6. Reconstruction
```python
system.reconstruction()  # Reconstruct marginal answers
```

### 7. Error Analysis
```python
error = system.get_mean_error()  # Calculate mean reconstruction error
```

## Basis Types

### Identity ('I')
- **Use Case**: Small domains (≤ 10 values)
- **Properties**: No compression, direct measurement
- **Matrix**: Identity matrix

### Prefix ('P')
- **Use Case**: Large domains with prefix sum queries
- **Properties**: Hierarchical structure, good for cumulative queries
- **Matrix**: Lower triangular matrix

### Range ('R')
- **Use Case**: Range queries over ordered domains
- **Properties**: Optimized for interval queries
- **Matrix**: Custom range-optimized structure

## Key Dependencies

### External Libraries
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy.sparse`: Sparse matrix operations
- `fractions`: Rational number arithmetic

### Internal Modules
- `utils.py`: Utility functions for matrix operations and optimization
- Functions used: `mult_kron_vec`, `all_subsets`, `find_var_sum_cauchy`, etc.

## Usage Examples

### Basic Example
```python
# Simple 2-attribute example
domains = [3, 3]
bases = ['I', 'I']
system = ResPlanSum(domains, bases)

# Add queries
system.input_mech((0,))    # Marginal on attribute 0
system.input_mech((1,))    # Marginal on attribute 1

# Process
system.get_noise_level()
system.measurement()
system.reconstruction()
```

### Advanced Example with Real Data
```python
# Load and process adult dataset
system, total = test_simple_adult()
obj = system.get_noise_level()
system.measurement()
system.reconstruction()
error = system.get_mean_error()
```

## Performance Considerations

- **Memory**: Grows with domain sizes and number of attributes
- **Computation**: Optimization step is the bottleneck
- **Scalability**: Works well up to ~10-15 attributes with mixed domain sizes

## Debug Mode

The system supports debug mode for detailed analysis:
```python
debug_results = system.reconstruction(debug=True)
# Returns detailed covariance and trace information
```

This enables analysis of:
- Covariance matrices for each residual mechanism
- Trace calculations for variance analysis
- Comparison between different reconstruction strategies
