import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def load_and_analyze_results(result_file=None):
    """
    Load the pickle file containing zero dataset simulation results and analyze them.
    
    Args:
        result_file: Path to the pickle file. If None, will try to find the most recent file.
    
    Returns:
        The loaded data dictionary
    """
    # Find the most recent results file if not specified
    if result_file is None:
        results_dir = 'results'
        if not os.path.exists(results_dir):
            print(f"Error: Results directory '{results_dir}' not found.")
            return None
            
        result_files = [f for f in os.listdir(results_dir) if f.startswith('zero_dataset_results_') and f.endswith('.pkl')]
        if not result_files:
            print(f"Error: No result files found in '{results_dir}'.")
            return None
            
        # Sort by iteration count (which should correlate with recency)
        result_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
        result_file = os.path.join(results_dir, result_files[0])
        print(f"Using most recent result file: {result_file}")
    
    # Load the results
    try:
        with open(result_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded results from {result_file}")
    except Exception as e:
        print(f"Error loading results: {e}")
        return None
    
    # Extract metadata
    metadata = data['metadata']
    print("\nSimulation Metadata:")
    print(f"- Domains: {metadata['domains']}")
    print(f"- Total iterations: {metadata['num_iterations']:,}")
    print(f"- Total queries: {metadata['total_queries']}")
    print(f"- Processing time: {metadata['processing_time']:.2f} seconds")
    #print(f"- Used {metadata['num_cpus']} CPU cores with batch size of {metadata['batch_size']:,}")
    
    # Analyze the accumulated results
    accumulated_results = data['accumulated_results']
    print(f"\nFound results for {len(accumulated_results)} different attribute combinations")
    
    # Prepare a table of results
    table_data = []
    
    # Calculate and display average results for each attribute combination
    for att, results in accumulated_results.items():
        if results['count'] > 0:
            avg_answer = results['answer'] / results['count']
            
            # Calculate statistics
            mean_value = np.mean(avg_answer)
            std_dev = np.std(avg_answer)
            min_value = np.min(avg_answer)
            max_value = np.max(avg_answer)
            
            # Format attribute combination as string
            att_str = ', '.join([str(a) for a in att])
            
            # Add to table data
            table_data.append([
                att_str, 
                results['count'], 
                mean_value,
                std_dev,
                min_value,
                max_value
            ])
    
    # Sort by attribute combination
    table_data.sort(key=lambda x: x[0])
    
    # Display the table
    headers = ["Attributes", "Count", "Mean", "Std Dev", "Min", "Max"]
    print("\nAverage Results Summary:")
    print(tabulate(table_data, headers=headers, floatfmt=".6f"))
    
    # Create visualizations
    #create_visualizations(accumulated_results, metadata)
    
    return data

def create_visualizations(accumulated_results, metadata):
    """
    Create visualizations of the results.
    
    Args:
        accumulated_results: Dictionary of accumulated results
        metadata: Metadata about the simulation
    """
    # Create a directory for visualizations if it doesn't exist
    vis_dir = 'visualizations'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # Plot 1: Distribution of average values across all attribute combinations
    plt.figure(figsize=(10, 6))
    all_means = []
    
    for att, results in accumulated_results.items():
        if results['count'] > 0:
            avg_answer = results['answer'] / results['count']
            all_means.extend(avg_answer.flatten())
    
    plt.hist(all_means, bins=50, alpha=0.75, color='skyblue')
    plt.title(f'Distribution of Average Values Across All Queries\n({metadata["num_iterations"]:,} iterations)')
    plt.xlabel('Average Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(vis_dir, 'distribution_all_values.png'), dpi=300, bbox_inches='tight')
    
    # Plot 2: Mean values by attribute combination
    plt.figure(figsize=(12, 6))
    
    att_labels = []
    mean_values = []
    
    for att, results in accumulated_results.items():
        if results['count'] > 0:
            avg_answer = results['answer'] / results['count']
            att_labels.append(str(att))
            mean_values.append(np.mean(avg_answer))
    
    # Sort by attribute combination
    sorted_indices = np.argsort(att_labels)
    att_labels = [att_labels[i] for i in sorted_indices]
    mean_values = [mean_values[i] for i in sorted_indices]
    
    plt.bar(range(len(att_labels)), mean_values, color='lightgreen')
    plt.xticks(range(len(att_labels)), att_labels, rotation=45, ha='right')
    plt.title(f'Mean Value by Attribute Combination\n({metadata["num_iterations"]:,} iterations)')
    plt.xlabel('Attribute Combination')
    plt.ylabel('Mean Value')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'mean_by_attribute.png'), dpi=300, bbox_inches='tight')
    
    print(f"\nVisualizations saved to '{vis_dir}' directory")

if __name__ == "__main__":
    # Load and analyze the most recent results file
    data = load_and_analyze_results("results/zero_dataset_results_MSE_sequential_10000_RRR.pkl")
    
    if data is not None:
        print("\nAnalysis complete!") 