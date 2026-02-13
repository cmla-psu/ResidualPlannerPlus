#requires seaborn

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    domains_line = None
    for line in lines:
        if line.strip().startswith("domains = ["):
            domains_line = line.strip()
            break

    if not domains_line:
        raise ValueError("Domain sizes not found in the file.")

    try:
        domains = eval(domains_line.split('=')[1].strip())
    except:
        raise ValueError("Error parsing domain sizes.")

    marginal_variances = {}
    for line in lines:
        if line.strip().startswith("Variance for marginal"):
            parts = line.strip().split(':')
            marginal_str = parts[0].split('(')[1].split(')')[0]
            marginal = tuple(int(x) for x in marginal_str.split(',') if x) if marginal_str else ()
            variance = float(parts[1].strip())
            marginal_variances[marginal] = variance

    return domains, marginal_variances

def calculate_cell_size(marginal, domains):
    return np.prod([domains[index] for index in marginal], dtype=np.int64) if marginal else 1

sns.set_style("whitegrid")
# Set font to Arial/Helvetica (sans serif) and remove LaTeX
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['text.usetex'] = False

font_size = 16  # Consistent font size as per journal requirements (8-12 pt)
# Set global font size for all matplotlib elements
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.titlesize'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size
plt.rcParams['ytick.labelsize'] = font_size
plt.rcParams['legend.fontsize'] = font_size
plt.rcParams['figure.titlesize'] = font_size

def plot_data(sizes, variances, colors, unique_marginal_sizes, ax, filter_k=None, legend_loc='lower right'):
    # Define markers for each k-way marginal (B/W friendly: distinct shapes, all black/gray)
    marker_styles = {
        0: dict(marker='x',  color='black',  s=60,  linewidths=1.5),
        1: dict(marker='s',  facecolors='none',   edgecolors='black',  s=60,  linewidths=1.5),
        2: dict(marker='o',  facecolors='none',   edgecolors='black',  s=60,  linewidths=1.5),
        3: dict(marker='^',  facecolors='none',   edgecolors='black',  s=70,  linewidths=1.5),
    }

    if filter_k is not None:
        filtered_sizes = [size for size, color in zip(sizes, colors) if color == filter_k]
        filtered_variances = [variance for variance, color in zip(variances, colors) if color == filter_k]
        style = marker_styles[filter_k]
        ax.scatter(filtered_sizes, filtered_variances, **style)
    else:
        for k in sorted(unique_marginal_sizes):
            k_sizes = [size for size, variance, color in zip(sizes, variances, colors) if color == k]
            k_variances = [variance for size, variance, color in zip(sizes, variances, colors) if color == k]
            style = marker_styles[k]
            ax.scatter(k_sizes, k_variances, **style, label=f'{k}-way marginal')
        
        # Add legend
        ax.legend(fontsize=font_size, loc=legend_loc)
        
        # Add dashed lines for min/max variance values
        min_variance = min(variances)
        max_variance = max(variances)
        
        # Draw horizontal dashed lines
        ax.axhline(y=min_variance, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=max_variance, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        

    ax.set_xlabel('Number of Cells', fontsize=font_size)
    ax.set_ylabel('Variance', fontsize=font_size)
    # Increase tick label size for better visibility
    ax.tick_params(labelsize=font_size + 2)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add min/max values to y-axis ticks
    if filter_k is None:
        min_variance = min(variances)
        max_variance = max(variances)
        
        # Get the current x-axis limits after setting log scale
        xlim = ax.get_xlim()
        
        # Position text at 10% from the left edge of the plot
        text_x_pos = xlim[0] * (xlim[1]/xlim[0])**0.1

        def format_scientific(value):
            sci_str = f'{value:.2e}'
            if 'e+' in sci_str:
                mantissa, exponent = sci_str.split('e+')
                exp_int = int(exponent)
                return f'${mantissa}\\times10^{{{exp_int}}}$'
            elif 'e-' in sci_str:
                mantissa, exponent = sci_str.split('e-')
                exp_int = int(exponent)
                return f'${mantissa}\\times10^{{-{exp_int}}}$'
            else:
                return sci_str

        min_text = f'Min: {format_scientific(min_variance)}'
        max_text = f'Max: {format_scientific(max_variance)}'

        ax.text(text_x_pos, min_variance, min_text,
                verticalalignment='bottom', fontsize=font_size-2, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.text(text_x_pos, max_variance, max_text,
                verticalalignment='bottom', fontsize=font_size-2, color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    

# Generate all three plots
# (input_file, output_pdf, legend_loc)
datasets = [
    ('res.txt',              'weighted.pdf',      'upper right'),
    ('unweighted.txt',       'unweighted.pdf',    'lower right'),
    ('sqrtweightedorig.txt', 'sqrtweighted.pdf',  'upper right'),
]

for filename, output_pdf, legend_loc in datasets:
    domains, marginal_variances = read_data(filename)

    sizes = []
    variances = []
    colors = []
    unique_marginal_sizes = set(len(marginal) for marginal in marginal_variances.keys())

    for marginal, variance in marginal_variances.items():
        size = calculate_cell_size(marginal, domains)
        sizes.append(size)
        variances.append(variance)
        colors.append(len(marginal))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_data(sizes, variances, colors, unique_marginal_sizes, ax, legend_loc=legend_loc)
    plt.savefig(output_pdf, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {output_pdf}')


