import numpy as np
import pandas as pd
import time
import itertools
from resplan.ResPlan import ResPlanSum

def test_domain_size(domain_size):
    """
    Test with a zero dataset for a specific domain size.
    
    Args:
        domain_size: Size of the domain to test (int)
    
    Returns:
        tuple: (system, total_queries, query_vectors)
    """
    # Set up domains with the specified size
    domains = [domain_size]
    col_names = ['attr1']
    bases = ['I']
    
    # Create the system
    system = ResPlanSum(domains, bases)
    
    # Create a zero dataset
    num_rows = 10
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system.input_data(zero_data, col_names)
    print(f"Created zero dataset for domain size {domain_size} with shape:", zero_data.shape)
    
    # Define 1-way marginal queries
    att = tuple(range(len(domains)))
    total_queries = 0
    
    i = 1  # 1-way marginals
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
    
    # Collect query vectors
    query_vectors = {}
    for att, mech in system.marg_dict.items():
        noisy_answer = mech.get_noisy_answer()
        true_answer = mech.get_true_answer()
        query_vectors[att] = {
            'noisy': noisy_answer.flatten() if noisy_answer is not None else None,
            'true': true_answer.flatten() if true_answer is not None else None
        }
    
    return system, total_queries, query_vectors

def test_multiple_attributes(domain_sizes):
    """
    Test with a zero dataset for multiple attributes with different domain sizes.
    
    Args:
        domain_sizes: List of domain sizes to test
    
    Returns:
        tuple: (system, total_queries, query_vectors)
    """
    # Set up domains with the specified sizes
    domains = domain_sizes
    col_names = [f'attr{i+1}' for i in range(len(domains))]
    bases = ['I'] * len(domains)
    
    # Create the system
    system = ResPlanSum(domains, bases)
    
    # Create a zero dataset
    num_rows = 10
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system.input_data(zero_data, col_names)
    print(f"Created zero dataset for domain sizes {domain_sizes} with shape:", zero_data.shape)
    
    # Define queries for all possible combinations of attributes
    att = tuple(range(len(domains)))
    total_queries = 0
    
    # Add all 1-way and 2-way marginals
    for i in range(1, min(3, len(domains)+1)):  # 1-way and 2-way marginals
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
    
    # Collect query vectors
    query_vectors = {}
    for att, mech in system.marg_dict.items():
        noisy_answer = mech.get_noisy_answer()
        true_answer = mech.get_true_answer()
        query_vectors[att] = {
            'noisy': noisy_answer.flatten() if noisy_answer is not None else None,
            'true': true_answer.flatten() if true_answer is not None else None
        }
    
    return system, total_queries, query_vectors

if __name__ == "__main__":
    print("=== Running Zero Dataset Tests for Different Domain Sizes ===\n")
    
    # Test domain size 2
    print("\n=== Testing Domain Size 2 ===")
    start = time.time()
    system2, queries2, vectors2 = test_domain_size(2)
    end = time.time()
    print(f"Domain size 2 test completed in {end - start:.2f} seconds")
    
    # Print some results
    for att, vecs in vectors2.items():
        print(f"Query vectors for attributes {att}:")
        print("Noisy:", vecs['noisy'])
        print("True:", vecs['true'])
        print()
    
    # Test domain size 3
    print("\n=== Testing Domain Size 3 ===")
    start = time.time()
    system3, queries3, vectors3 = test_domain_size(3)
    end = time.time()
    print(f"Domain size 3 test completed in {end - start:.2f} seconds")
    
    # Print some results
    for att, vecs in vectors3.items():
        print(f"Query vectors for attributes {att}:")
        print("Noisy:", vecs['noisy'])
        print("True:", vecs['true'])
        print()
    
    # Test domain size 4
    print("\n=== Testing Domain Size 4 ===")
    start = time.time()
    system4, queries4, vectors4 = test_domain_size(4)
    end = time.time()
    print(f"Domain size 4 test completed in {end - start:.2f} seconds")
    
    # Print some results
    for att, vecs in vectors4.items():
        print(f"Query vectors for attributes {att}:")
        print("Noisy:", vecs['noisy'])
        print("True:", vecs['true'])
        print()
    
    # Test all domain sizes together
    print("\n=== Testing Combined Domain Sizes [2, 3, 4] ===")
    start = time.time()
    system_combined, queries_combined, vectors_combined = test_multiple_attributes([2, 3, 4])
    end = time.time()
    print(f"Combined test completed in {end - start:.2f} seconds")
    
    # Print some 1-way marginal results
    print("\n1-way Marginal Results:")
    for i, size in enumerate([2, 3, 4]):
        att = (i,)
        if att in vectors_combined:
            print(f"Query vectors for attribute {att} (domain size {size}):")
            print("Noisy (first 5 elements):", vectors_combined[att]['noisy'][:5])
            print("True (first 5 elements):", vectors_combined[att]['true'][:5])
            print()
    
    # Print some 2-way marginal results
    print("\n2-way Marginal Results (sample):")
    sample_att = (0, 1)  # Domain sizes 2 and 3
    if sample_att in vectors_combined:
        print(f"Query vectors for attributes {sample_att} (domain sizes 2,3):")
        print("Noisy (first 5 elements):", vectors_combined[sample_att]['noisy'][:5])
        print("True (first 5 elements):", vectors_combined[sample_att]['true'][:5])
        print() 