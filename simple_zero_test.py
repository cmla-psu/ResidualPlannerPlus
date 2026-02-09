import numpy as np
import pandas as pd
import time
import itertools
from ResPlan import ResPlanSum

def test_simple_zero():
    """
    A simple test with a zero dataset for domains of size 2, 3, and 4.
    """
    # Define domains of size 2, 3, and 4
    domains = [2, 3, 4]
    col_names = ['attr1', 'attr2', 'attr3']
    bases = ['I', 'I', 'I']
    
    # Create the system
    system = ResPlanSum(domains, bases)
    
    # Create a zero dataset
    num_rows = 10
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system.input_data(zero_data, col_names)
    print("Created zero dataset with shape:", zero_data.shape)
    
    # Define 2-way marginal queries
    att = tuple(range(len(domains)))
    total_queries = 0
    
    i = 2  # 2-way marginals
    subset_i = list(itertools.combinations(att, i))
    print(f"Number of {i}-way marginals: {len(subset_i)}")
    for subset in subset_i:
        system.input_mech(subset)
        cur_domains = [domains[c] for c in subset]
        total_queries += np.multiply.reduce(cur_domains)
    
    print(f"Total number of queries: {total_queries}")
    
    # Set fixed noise levels
    fixed_noise_level = 1.0
    for att_key, res_mech in system.res_dict.items():
        res_mech.input_noise_level(fixed_noise_level)
    
    # Perform measurement
    print("Starting measurement...")
    system.measurement()
    
    # Perform reconstruction
    print("Starting reconstruction...")
    system.reconstruction()
    
    # Print results
    print("\nQuery results:")
    for att, mech in system.marg_dict.items():
        print(f"\nQuery on attributes {att}:")
        noisy_answer = mech.get_noisy_answer()
        true_answer = mech.get_true_answer()
        if noisy_answer is not None and true_answer is not None:
            print("Noisy answer shape:", noisy_answer.shape)
            print("Noisy answer (first few elements):", noisy_answer.flatten()[:5])
            print("True answer (first few elements):", true_answer.flatten()[:5])
        else:
            print("Warning: Answers not available for this query")
    
    return system, total_queries

if __name__ == "__main__":
    print("Running simple zero dataset test...")
    start = time.time()
    system, total_queries = test_simple_zero()
    end = time.time()
    print(f"\nTest completed in {end - start:.2f} seconds")
    
    # Print noise levels
    print("\nNoise levels for each residual mechanism:")
    for att, res_mech in system.res_dict.items():
        print(f"Attribute {att}: {res_mech.output_noise_level():.4f}") 