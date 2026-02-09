# ResPlan Abstract Architecture Design Report

## Executive Summary

The ResPlan system has been redesigned with a modular, extensible architecture using abstract base classes and strategy patterns. This design separates concerns into distinct, pluggable components while maintaining the core differential privacy functionality. The new architecture enables easy extension with different optimization algorithms, residual computation methods, and noise mechanisms.

## Core Design Principles

### 1. **Strategy Pattern Implementation**
- Each major algorithmic component is abstracted into interchangeable strategies
- Enables mixing and matching different approaches without code modification
- Supports future research directions and algorithm experimentation

### 2. **Template Method Pattern**
- Common orchestration logic is centralized in the abstract base planner
- Algorithm-specific implementations are delegated to concrete classes
- Measurement and reconstruction are abstract methods requiring custom implementation

### 3. **Separation of Concerns**
- Data structures and algorithms are clearly separated
- Mechanism classes focus on data storage, planner classes handle processing
- Clear interfaces between components enable independent testing and development

## Architecture Overview

```
AbstractResidualPlanner (Main Orchestrator)
├── AbstractResidualBasisStrategy (Residual Matrix Computation)
├── AbstractOptimizationStrategy (Privacy Budget Allocation)  
├── AbstractMeasurementMechanism (Data Structure)
└── AbstractQueryMechanism (Data Structure)
```

## Core Abstract Classes

### 1. **AbstractResidualBasisStrategy**
**Purpose**: Computes residual matrices from workload matrices

**Key Responsibilities**:
- Generate residual matrices for different domain sizes
- Calculate privacy cost coefficients
- Compute variance coefficients for optimization

**Extensibility**: 
- `SubtractionResidualStrategy` (current approach)
- `OrthogonalResidualStrategy` (orthogonal complement)
- `SpectralResidualStrategy` (spectral decomposition)
- `CustomResidualStrategy` (user-defined functions)

**Interface**:
```python
@abstractmethod
def compute_residual_matrices(self, workload_matrix: np.ndarray) -> ResidualBasisResult
def get_privacy_cost_coefficient(self) -> float
def get_variance_coefficients(self) -> Dict[str, np.ndarray]
```

### 2. **AbstractOptimizationStrategy**
**Purpose**: Solves privacy budget allocation optimization problems

**Key Responsibilities**:
- Take privacy and variance coefficients as input
- Solve constrained optimization problem
- Return optimal noise level allocation

**Extensibility**:
- `SumVarianceOptimizer` (minimize L1 norm - current)
- `MaxVarianceOptimizer` (minimize L∞ norm - current)
- `WeightedVarianceOptimizer` (weighted query importance)
- `RobustOptimizer` (uncertainty-aware optimization)
- `AdaptiveOptimizer` (dynamic optimization with feedback)

**Interface**:
```python
@abstractmethod
def compute_noise_allocation(self, privacy_coefficients: np.ndarray,
                           variance_coefficients: np.ndarray,
                           constraint_matrix: Optional[np.ndarray] = None,
                           bounds: Optional[np.ndarray] = None) -> OptimizationResult
def get_objective_type(self) -> str
```

### 3. **AbstractMeasurementMechanism**
**Purpose**: Data storage for individual measurement mechanisms

**Key Responsibilities**:
- Store measurement results (noisy and true answers)
- Provide reconstruction matrices for query reconstruction
- Maintain metadata about measurement process

**Extensibility**:
- `GaussianMeasurementMechanism` (current Gaussian noise)
- `LaplaceMeasurementMechanism` (Laplace noise for pure DP)
- `DiscreteGaussianMechanism` (discrete outputs)
- `TruncatedMeasurementMechanism` (bounded noise)

**Interface**:
```python
@abstractmethod
def store_measurement_result(self, noisy_answer: np.ndarray, 
                           true_answer: np.ndarray, metadata: Dict[str, Any]) -> None
def get_reconstruction_matrices(self, target_attributes: Tuple[int]) -> List[np.ndarray]
def get_stored_results(self) -> MeasurementResult
```

### 4. **AbstractQueryMechanism**
**Purpose**: Data storage for marginal query mechanisms

**Key Responsibilities**:
- Store final reconstruction results
- Maintain query metadata and statistics
- Provide interface for result retrieval

**Interface**:
```python
@abstractmethod
def store_reconstruction_result(self, answer: np.ndarray, metadata: Dict[str, Any]) -> None
def get_query_result(self) -> QueryResult
def get_num_queries(self) -> int
```

### 5. **AbstractResidualPlanner** (Main Orchestrator)
**Purpose**: Coordinates the entire differential privacy pipeline

**Key Responsibilities**:
- System initialization and configuration
- Query registration and mechanism creation
- Privacy budget optimization coordination
- **Abstract Methods**: `measurement()` and `reconstruction()`

**Core Interface**:
```python
# Concrete methods (shared logic)
def setup_system(self) -> None
def input_data(self, data: pd.DataFrame, column_names: List[str]) -> None
def register_query(self, attributes: Tuple[int], **query_params) -> str
def optimize_privacy_budget(self, total_budget: float = 1.0) -> OptimizationResult

# Abstract methods (custom implementation required)
@abstractmethod
def measurement(self) -> Dict[Tuple[int], MeasurementResult]
@abstractmethod  
def reconstruction(self) -> Dict[Tuple[int], ReconstructionResult]

# Factory methods (custom implementation required)
@abstractmethod
def _create_query_mechanism(self, attributes: Tuple[int], **params) -> AbstractQueryMechanism
@abstractmethod
def _create_measurement_mechanism(self, attributes: Tuple[int]) -> AbstractMeasurementMechanism
```

## Abstract Methods Design

### **measurement()** - Abstract Method
**Signature**: `measurement() -> Dict[Tuple[int], MeasurementResult]`

**Required Implementation Steps**:
1. For each measurement mechanism:
   - Extract relevant data columns
   - Convert data to histogram representation  
   - Apply residual transformation matrices
   - Add calibrated noise based on mechanism type
   - Store results in mechanism object

**Customization Points**:
- Noise distribution (Gaussian, Laplace, etc.)
- Histogram construction method
- Residual transformation approach
- Noise calibration strategy

### **reconstruction()** - Abstract Method  
**Signature**: `reconstruction() -> Dict[Tuple[int], ReconstructionResult]`

**Required Implementation Steps**:
1. For each query mechanism:
   - Build workload matrices based on basis types
   - Aggregate results from relevant measurement mechanisms
   - Apply reconstruction transformations
   - Compute final query answer
   - Store results in query mechanism

**Customization Points**:
- Workload matrix construction
- Subset aggregation strategy
- Error estimation methods
- Covariance matrix computation

## Concrete Implementation Examples

### **GaussianResidualPlanner**
```python
class GaussianResidualPlanner(AbstractResidualPlanner):
    def measurement(self) -> Dict[Tuple[int], MeasurementResult]:
        # Implement Gaussian noise measurement
        # - Extract data columns
        # - Build histograms  
        # - Apply residual transformations
        # - Add Gaussian noise
        # - Store results
        
    def reconstruction(self) -> Dict[Tuple[int], ReconstructionResult]:
        # Implement standard reconstruction with subset aggregation
        # - Build workload matrices
        # - Aggregate from measurement mechanisms
        # - Apply final transformations
        # - Store results
```

### **LaplaceResidualPlanner**  
```python
class LaplaceResidualPlanner(AbstractResidualPlanner):
    def measurement(self) -> Dict[Tuple[int], MeasurementResult]:
        # Implement Laplace noise for pure differential privacy
        
    def reconstruction(self) -> Dict[Tuple[int], ReconstructionResult]:
        # Use reconstruction optimized for Laplace properties
```

## Data Structures

### **Result Classes**
```python
@dataclass
class ResidualBasisResult:
    sum_matrix: np.ndarray
    residual_matrix: np.ndarray
    sum_transform: np.ndarray
    residual_transform: np.ndarray
    privacy_costs: Dict[str, float]
    variance_coefficients: Dict[str, np.ndarray]

@dataclass 
class OptimizationResult:
    noise_levels: np.ndarray
    objective_value: float
    convergence_info: Dict[str, Any]
    dual_variables: Optional[np.ndarray] = None

@dataclass
class MeasurementResult:
    noisy_answer: np.ndarray
    true_answer: np.ndarray
    measurement_transform: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class ReconstructionResult:
    final_answer: np.ndarray
    intermediate_results: Dict[str, np.ndarray]
    error_estimates: Dict[str, float]
    covariance_matrix: Optional[np.ndarray] = None

@dataclass
class QueryResult:
    answer: np.ndarray
    error_bound: float
    privacy_cost: float
    query_metadata: Dict[str, Any]
```

## Data Flow Pipeline

```
1. System Setup:
   setup_system() → precompute residual matrices

2. Data Input:
   input_data() → store DataFrame and column mappings

3. Query Registration:
   register_query() → create query and measurement mechanisms

4. Optimization:
   optimize_privacy_budget() → solve for optimal noise levels

5. Measurement (Abstract):
   measurement() → process data through mechanisms with noise

6. Reconstruction (Abstract):
   reconstruction() → combine measurements into query answers

7. Result Retrieval:
   get_query_result() → return final answers with metadata
```

## Usage Example

```python
# Create planner with custom strategies
planner = GaussianResidualPlanner(
    domains=[85, 9, 2],
    basis_types=['P', 'I', 'I'],
    residual_strategy=SubtractionResidualStrategy(),
    optimization_strategy=SumVarianceOptimizer()
)

# Setup and configure
planner.setup_system()
planner.input_data(data, column_names)
planner.register_query((0, 1))
planner.register_query((1, 2))

# Execute pipeline
planner.optimize_privacy_budget()
measurement_results = planner.measurement()      # Abstract method
reconstruction_results = planner.reconstruction() # Abstract method

# Get results
final_results = planner.get_all_query_results()
```

## Factory Pattern for Easy Configuration

```python
class ResidualPlannerFactory:
    @staticmethod
    def create_sum_planner(domains: List[int], basis_types: List[str]) -> AbstractResidualPlanner:
        """Create planner optimized for sum of variances."""
        
    @staticmethod  
    def create_max_planner(domains: List[int], basis_types: List[str]) -> AbstractResidualPlanner:
        """Create planner optimized for maximum variance."""
        
    @staticmethod
    def create_custom_planner(config: PlannerConfig) -> AbstractResidualPlanner:
        """Create planner with custom strategy combination."""
```

## Key Benefits

### **Extensibility**
- Easy addition of new optimization algorithms
- Support for different residual computation methods
- Pluggable noise mechanisms and reconstruction strategies

### **Maintainability**
- Clear separation of concerns
- Abstract interfaces prevent tight coupling
- Shared common logic reduces code duplication

### **Testability**
- Each component can be tested independently
- Mock implementations enable unit testing
- Clear interfaces facilitate integration testing

### **Research Enablement**
- Framework supports experimental algorithms
- Easy comparison of different approaches
- Modular design enables focused research on specific components

## Implementation Considerations

### **Backward Compatibility**
- Current `ResPlanSum`/`ResPlanMax` can be reimplemented as concrete classes
- Existing test cases and examples remain valid
- Migration path preserves existing functionality

### **Performance**
- Abstract method calls have minimal overhead
- Strategy objects can be optimized independently
- Shared computation remains in base class

### **Configuration Management**
- Factory pattern simplifies common configurations
- Strategy combination validation can be centralized
- Configuration serialization enables reproducible experiments

## Future Extensions

### **Potential New Strategies**

**Residual Basis Strategies:**
- `HierarchicalResidualStrategy` - Tree-based decomposition
- `AdaptiveResidualStrategy` - Data-dependent residual computation
- `CompressedResidualStrategy` - Low-rank approximations

**Optimization Strategies:**
- `MultiObjectiveOptimizer` - Pareto-optimal solutions
- `OnlineOptimizer` - Streaming data optimization
- `FederatedOptimizer` - Multi-party optimization

**Measurement Mechanisms:**
- `HybridMeasurementMechanism` - Mix of noise types
- `QuantizedMeasurementMechanism` - Integer-constrained outputs
- `BoundedMeasurementMechanism` - Range-constrained mechanisms

## Conclusion

This abstract architecture provides a robust foundation for extending ResPlan while maintaining its core differential privacy guarantees. The design enables researchers and practitioners to easily experiment with new algorithms, compare different approaches, and customize the system for specific use cases. The clear separation of concerns and well-defined interfaces ensure the system remains maintainable and testable as it grows in complexity.

The modular design supports both backward compatibility with existing code and forward compatibility with future research directions, making it an ideal foundation for continued development of differential privacy mechanisms.
