import numpy as np
import pandas as pd
import itertools
import time
import os
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Import from ResPlan.py
from ResPlan import ResPlanSum, all_subsets

def process_iteration(seed):
    """
    Process a single iteration with a unique random seed.
    Creates a fresh system for each iteration to avoid shared state issues.
    """
    # Define domains of size 2, 3, and 4
    domains = [2, 3, 4]
    col_names = ['attr1', 'attr2', 'attr3']
    # Using Identity basis for all columns
    bases = ['R', 'R', 'I']
    
    # Create a fresh system
    system = ResPlanSum(domains, bases)
    
    # Create a zero dataset with the appropriate shape
    num_rows = 0  # Number of rows in the dataset
    #TODO: should be 0 row in the vector. print out log of data vector
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system.input_data(zero_data, col_names)
    
    # Add 2-way marginals
    att = tuple(range(len(domains)))
    i = 2
    subset_i = list(itertools.combinations(att, i))
    for subset in subset_i:
        system.input_mech(subset)
    
    # Use a fixed positive noise level instead of random noise
    fixed_noise_level = 4.76
    for att_key, res_mech in system.res_dict.items():
        res_mech.input_noise_level(fixed_noise_level)
    
    # Perform measurement and reconstruction
    system.measurement()
    system.reconstruction()
    
    # Collect results
    results = {}
    for att, mech in system.marg_dict.items():
        answer_vector = mech.get_noisy_answer_vector()
        # Square the answer vector to get squared error since true answer is zero
        answer_vector = np.square(answer_vector)
        if answer_vector is not None:
            results[att] = {'answer': answer_vector}
        else:
            results[att] = {'answer': None}
    
    return results


def process_batch(batch_params):
    """
    Process a batch of iterations and return aggregated results.
    
    Args:
        batch_params: Tuple containing (start_seed, num_iterations)
    
    Returns:
        Dictionary with aggregated results for the batch
    """
    start_seed, num_iterations = batch_params
    
    # Initialize a dictionary to store accumulated results for the batch
    batch_results = {}
    
    # Process all iterations in the batch
    for i in range(num_iterations):
        seed = start_seed + i
        iteration_result = process_iteration(seed)
        
        # Skip if this iteration failed
        if iteration_result is None:
            continue
            
        # Accumulate results from this iteration
        for att, result in iteration_result.items():
            if result['answer'] is not None:
                if att not in batch_results:
                    batch_results[att] = {
                        'answer': result['answer'],
                        'count': 1
                    }
                else:
                    batch_results[att]['answer'] += result['answer']
                    batch_results[att]['count'] += 1
    
    return batch_results


def test_zero_dataset():
    """
    Test with a zero dataset for domains of size 2, 3, and 4.
    This function creates a zero dataset, applies random noise without selection phase,
    then measures and reconstructs query answers using a batched approach.
    """
    # Define domains of size 2, 3, and 4
    domains = [2, 3, 4]
    col_names = ['attr1', 'attr2', 'attr3']
    # Using Identity basis for all columns
    bases = ['I', 'I', 'I']
    
    # Create the system using ResPlanSum (we'll use this just for setup and reporting)
    system = ResPlanSum(domains, bases)
    
    # Create a zero dataset with the appropriate shape
    num_rows = 10  # Number of rows in the dataset
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system.input_data(zero_data, col_names)
    print("Created zero dataset with shape:", zero_data.shape)
    
    # Define queries for all possible combinations of attributes
    att = tuple(range(len(domains)))
    total_queries = 0
    
    # Add only 2-way marginals
    i = 2
    subset_i = list(itertools.combinations(att, i))
    print(f"Number of {i}-way marginals: {len(subset_i)}")
    for subset in subset_i:
        system.input_mech(subset)
        cur_domains = [domains[c] for c in subset]
        total_queries += np.multiply.reduce(cur_domains)

    print(f"Total number of queries: {total_queries}")
    
    # Number of iterations to run - increased to 1 million
    total_iterations = 1000000
    print(f"Running a total of {total_iterations:,} iterations")
    
    # Set up multiprocessing
    num_cpus = min(cpu_count(), 12)  # Limit to max 8 CPUs to avoid memory issues
    print(f"Using {num_cpus} CPU cores for parallel processing")
    
    # Calculate the batch size for each worker
    batch_size = total_iterations // num_cpus
    print(f"Each worker will process {batch_size:,} iterations")
    
    # Create batch parameters for each worker
    batch_params = [(i * batch_size + 42, batch_size) for i in range(num_cpus)]
    # Adjust the last batch to handle any remainder
    if total_iterations % num_cpus != 0:
        remainder = total_iterations - (batch_size * num_cpus)
        batch_params[-1] = (batch_params[-1][0], batch_params[-1][1] + remainder)
    
    # Create a pool of workers
    pool = Pool(processes=num_cpus)
    
    # Process batches in parallel with progress bar
    start_time = time.time()
    print("Starting batch processing...")
    batch_results = list(tqdm(
        pool.imap(process_batch, batch_params),
        total=len(batch_params),
        desc="Processing batches"
    ))
    pool.close()
    pool.join()
    
    # Total processing time
    processing_time = time.time() - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Average time per iteration: {processing_time / total_iterations * 1000:.4f} ms")
    
    # Combine results from all batches
    accumulated_results = {}
    
    for batch_result in batch_results:
        for att, result in batch_result.items():
            if att not in accumulated_results:
                accumulated_results[att] = {
                    'answer': result['answer'],
                    'count': result['count']
                }
            else:
                accumulated_results[att]['answer'] += result['answer']
                accumulated_results[att]['count'] += result['count']
    
    # Print averaged results
    print("\nAveraged results after processing:")
    count = 0
    for att, results in accumulated_results.items():
        if count < 3 and results['count'] > 0:  # Only show examples with valid results
            print(f"\nQuery on attributes {att}:")
            avg_answer = results['answer'] / results['count']
            print(f"Results based on {results['count']:,} valid iterations")
            print("Average noisy answer shape:", avg_answer.shape)
            print("Average noisy answer (first few elements):", avg_answer.flatten()[:5])
            count += 1
    
    # Make sure we have a place to save the results
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save the raw accumulated results directly with timestamp
    current_time = time.strftime("%Y%m%d")
    result_file = f'results/zero_dataset_results_MSE_{total_iterations}_{str(current_time)}.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump({
            'accumulated_results': accumulated_results,
            'metadata': {
                'domains': domains,
                'num_iterations': total_iterations,
                'total_queries': total_queries,
                'processing_time': processing_time,
                'batch_size': batch_size,
                'num_cpus': num_cpus
            }
        }, f)
    
    print(f"\nResults saved to {result_file}")
    
    return system, total_queries


def test_zero_dataset_sequential():
    """
    Test with a zero dataset for domains of size 2, 3, and 4.
    This function creates a zero dataset, applies random noise without selection phase,
    then measures and reconstructs query answers using a sequential approach.
    """
    # Define domains of size 2, 3, and 4
    domains = [2, 3, 4]
    col_names = ['attr1', 'attr2', 'attr3']
    # Using Identity basis for all columns
    bases = ['I', 'I', 'I']
    
    # Create the system using ResPlanSum (we'll use this just for setup and reporting)
    system = ResPlanSum(domains, bases)
    
    # Create a zero dataset with the appropriate shape
    num_rows = 0  # Number of rows in the dataset
    zero_data = pd.DataFrame(np.zeros((num_rows, len(domains))), columns=col_names)
    system.input_data(zero_data, col_names)
    print("Created zero dataset with shape:", zero_data.shape)
    
    # Define queries for all possible combinations of attributes
    att = tuple(range(len(domains)))
    total_queries = 0
    
    # Add only 2-way marginals
    i = 2
    subset_i = list(itertools.combinations(att, i))
    print(f"Number of {i}-way marginals: {len(subset_i)}")
    for subset in subset_i:
        system.input_mech(subset)
        cur_domains = [domains[c] for c in subset]
        total_queries += np.multiply.reduce(cur_domains)

    print(f"Total number of queries: {total_queries}")
    
    # Number of iterations to run - reduced for sequential processing
    total_iterations = 10000  # Reduced from 1,000,000 for sequential processing
    print(f"Running a total of {total_iterations:,} iterations")
    
    # Initialize a dictionary to store accumulated results
    accumulated_results = {}
    
    # Process all iterations sequentially with a progress bar
    start_time = time.time()
    print("Starting sequential processing...")
    
    for seed in tqdm(range(42, 42 + total_iterations), desc="Processing iterations"):
        # Process a single iteration
        iteration_result = process_iteration(seed)
        
        # Skip if this iteration failed
        if iteration_result is None:
            continue
            
        # Accumulate results from this iteration
        for att, result in iteration_result.items():
            if result['answer'] is not None:
                if att not in accumulated_results:
                    accumulated_results[att] = {
                        'answer': result['answer'],
                        'count': 1
                    }
                else:
                    accumulated_results[att]['answer'] += result['answer']
                    accumulated_results[att]['count'] += 1
    
    # Total processing time
    processing_time = time.time() - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    print(f"Average time per iteration: {processing_time / total_iterations * 1000:.4f} ms")
    
    # Print averaged results
    print("\nAveraged results after processing:")
    count = 0
    for att, results in accumulated_results.items():
        if count < 3 and results['count'] > 0:  # Only show examples with valid results
            print(f"\nQuery on attributes {att}:")
            avg_answer = results['answer'] / results['count']
            print(f"Results based on {results['count']:,} valid iterations")
            print("Average noisy answer shape:", avg_answer.shape)
            print("Average noisy answer (first few elements):", avg_answer.flatten()[:5])
            count += 1
    
    # Make sure we have a place to save the results
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Save the raw accumulated results directly
    result_file = f'results/zero_dataset_results_MSE_sequential_{total_iterations}_RRR.pkl'
    with open(result_file, 'wb') as f:
        pickle.dump({
            'accumulated_results': accumulated_results,
            'metadata': {
                'domains': domains,
                'num_iterations': total_iterations,
                'total_queries': total_queries,
                'processing_time': processing_time
            }
        }, f)
    
    print(f"\nResults saved to {result_file}")
    
    return system, total_queries


if __name__ == "__main__":
    # Run the zero dataset test when the script is executed directly
    # start_time = time.time()
    # system, total_queries = test_zero_dataset()
    # end_time = time.time()
    # print(f"Test completed in {end_time - start_time:.2f} seconds")

    # Run the sequential zero dataset test when the script is executed directly
    start_time = time.time()
    system, total_queries = test_zero_dataset_sequential()
    end_time = time.time()
    print(f"Test completed in {end_time - start_time:.2f} seconds") 