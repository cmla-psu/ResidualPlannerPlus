# ResPlan.py Structure Documentation

## Overview
ResPlan.py implements a residual planning mechanism for differential privacy, with two main variants: ResPlanSum and ResPlanMax. The code provides functionality for handling marginal queries and residual mechanisms with different basis types (Prefix, Identity, Range).

## Core Classes

### 1. MargMech
- Purpose: Handles marginal mechanism operations
- Key attributes:
  - domains: List of domain sizes
  - att: Attributes for the marginal
  - var_bound: Variance bound
  - noisy_answer_vector: Vector containing noisy answers
  - non_noisy_vector: Vector containing true answers
- Main methods:
  - input_noisy_answer_vector(): Sets noisy answer vector
  - input_non_noisy_vector(): Sets non-noisy vector
  - get_noisy_answer_vector(): Retrieves noisy answers
  - get_non_noisy_vector(): Retrieves true answers

### 2. ResMech
- Purpose: Implements residual mechanism functionality
- Key attributes:
  - domains: List of domain sizes
  - att: Attributes for the mechanism
  - residual_matrix: Matrix for residual calculations
  - noise_level: Level of noise to be added
- Main methods:
  - get_core_matrix(): Generates core matrix for calculations
  - measure(): Performs measurement operations
  - get_recon_answer(): Reconstructs answers
  - get_origin_answer(): Gets original answers

### 3. ResidualPlanner (Base Class)
- Purpose: Base class for residual planning implementations
- Key attributes:
  - domains: List of domain sizes
  - bases: List of basis types
  - marg_dict: Dictionary of marginal mechanisms
  - res_dict: Dictionary of residual mechanisms
- Main methods:
  - preprocessing(): Initial setup operations
  - input_mech(): Adds a mechanism
  - measurement(): Performs measurements
  - reconstruction(): Reconstructs answers
  - get_mean_error(): Calculates mean error

### 4. ResPlanSum
- Purpose: Implements sum-based residual planning
- Inherits from: ResidualPlanner
- Key methods:
  - preprocessing(): Sets up sum-based residual matrices
  - input_mech(): Adds mechanisms with sum-based calculations
  - get_noise_level(): Calculates noise levels for sum-based approach

### 5. ResPlanMax
- Purpose: Implements max-based residual planning
- Inherits from: ResidualPlanner
- Key methods:
  - preprocessing(): Sets up max-based residual matrices
  - input_mech(): Adds mechanisms with max-based calculations
  - get_noise_level(): Calculates noise levels for max-based approach

## Test Functions

### 1. test_prefixsum()
- Tests prefix sum functionality with single domain

### 2. test_prefixsum_max()
- Tests prefix sum functionality with max-based approach

### 3. test_Adult()
- Tests with Adult dataset
- Uses 14 attributes with mixed basis types

### 4. test_simple_adult()
- Simplified version of Adult dataset test
- Uses 3 attributes (education, marital, gender)

### 5. test_age_income()
- Tests with age and income attributes
- Uses prefix basis for both attributes

### 6. test_allkway_csv()
- Tests k-way marginals with configurable parameters
- Supports custom domain size and number of attributes

## Key Features
1. Support for multiple basis types:
   - Prefix (P)
   - Identity (I)
   - Range (R)

2. Flexible domain handling:
   - Supports multiple attributes
   - Variable domain sizes
   - Mixed basis types

3. Privacy mechanisms:
   - Sum-based residual planning
   - Max-based residual planning
   - Configurable noise levels

4. Error measurement:
   - Mean error calculation
   - Support for different error norms

## Dependencies
- numpy
- pandas
- scipy.sparse
- itertools
- collections.defaultdict
- functools.reduce 