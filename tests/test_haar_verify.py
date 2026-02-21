"""
Manually verify RP+ results using the Haar binary tree as the strategy matrix.

For n=8, the strategy is:
    H = [Bs; R]
where
    Bs = [1 1 1 1 1 1 1 1]
    R  = 7 rows encoding "left child minus right child" at each tree level

We trace through every RP+ computation step and compare against
the actual ResPlanSum code output.

KEY: RP+ noise model (see ResMech.measure()):
    noise is drawn in the DATA space:  z ~ N(0, sigma^2 I_n)
    then projected through R:          R @ z
    so the measurement is:             y = R @ x + R @ z = R @ (x + z)
    NOT the HDMM model:                y = R @ x + N(0, sigma^2 I_{n-1})
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resplan'))

import numpy as np
from resplan.utils import find_var_sum_cauchy, all_subsets

# ================================================================
# Step 0: Build the Haar tree for n=8
# ================================================================
n = 8

Bs = np.ones((1, n))

rows = []
block_size = n // 2
while block_size >= 1:
    num_blocks = n // (2 * block_size)
    for b in range(num_blocks):
        row = np.zeros(n)
        start = b * 2 * block_size
        row[start:start + block_size] = 1
        row[start + block_size:start + 2 * block_size] = -1
        rows.append(row)
    block_size //= 2
R = np.array(rows)

H = np.vstack([Bs, R])

print("=" * 70)
print("Step 0: Haar tree matrix H = [Bs; R]  (n=8)")
print("=" * 70)
print("Bs =", Bs)
print()
print("R (7x8) =")
print(R)
print()

# ================================================================
# Step 1: Verify orthogonality  (Bs @ pinv(R) == 0)
# ================================================================
print("=" * 70)
print("Step 1: Orthogonality check")
print("=" * 70)
R_pinv = np.linalg.pinv(R)
check = Bs @ R_pinv
print("Bs @ pinv(R) =", check)
print("All close to zero?", np.allclose(check, 0, atol=1e-10))
print()

# Verify rows of H are mutually orthogonal
gram = H @ H.T
off_diag = gram - np.diag(np.diag(gram))
print("H rows mutually orthogonal?", np.allclose(off_diag, 0, atol=1e-10))
print("Row norms squared (diag of H H^T):", np.diag(gram))
print()

# ================================================================
# Step 2: Privacy cost
# ================================================================
print("=" * 70)
print("Step 2: Privacy cost  pcost_res = max(diag(R^T R))")
print("=" * 70)
RtR = R.T @ R
print("diag(R^T R) =", np.diag(RtR))
pcost_res = np.max(np.diag(RtR))
print("pcost_res =", pcost_res, " (= log2(n) =", np.log2(n), ")")
print()

# ================================================================
# Step 3: Workload decomposition  W = Us*Bs + Ur*R
# ================================================================
print("=" * 70)
print("Step 3: Workload decomposition (W = I_8)")
print("=" * 70)
W = np.eye(n)  # identity workload

# Solve  H^T @ X^T = W^T  =>  X = H^{-1}  (since W=I)
l_mat = H.T  # n x n
r_mat = W.T  # n x n
X = np.linalg.solve(l_mat, r_mat).T  # n x n

Us = X[:, 0].reshape(-1, 1)  # n x 1
Ur = X[:, 1:]                # n x (n-1)

print("Us (how each cell query uses Bs) =")
print(Us.flatten())
print()
print("Ur (how each cell query uses R), first 3 rows:")
print(Ur[:3])
print()

# Check: W == Us @ Bs + Ur @ R ?
W_reconstructed = Us @ Bs + Ur @ R
print("W == Us@Bs + Ur@R ?", np.allclose(W, W_reconstructed, atol=1e-10))
print()

# ================================================================
# Step 4: var_sum and var_res
# ================================================================
print("=" * 70)
print("Step 4: Variance factors")
print("=" * 70)
var_sum = np.trace(Us @ Us.T)
var_res = np.linalg.norm(Ur @ R, 'fro') ** 2

print("var_sum = trace(Us Us^T) = ||Us||^2 =", var_sum, " (= 1/n =", 1/n, ")")
print("var_res = ||Ur @ R||_F^2 =", var_res, " (= n-1 =", n-1, ")")
print()

# Verify Ur@R = I - J/n
UrR = Ur @ R
I_minus_J_over_n = np.eye(n) - np.ones((n, n)) / n
print("Ur@R == I - J/n ?", np.allclose(UrR, I_minus_J_over_n, atol=1e-10))
print()

# ================================================================
# Step 5: Noise allocation (d=1, att=(0,))
# ================================================================
print("=" * 70)
print("Step 5: Noise allocation  (d=1, single attribute)")
print("=" * 70)

v = np.array([var_sum, var_res])
p = np.array([1.0, pcost_res])

print("Subset ()  : var_coeff =", var_sum, ", pcost_coeff =", 1.0)
print("Subset (0,): var_coeff =", var_res, ", pcost_coeff =", pcost_res)
print()

sigma_sq, T = find_var_sum_cauchy(v, p, c=1.0)

print("Cauchy-Schwarz optimal noise allocation:")
print("  T (total sum of variances) =", T)
print("  sigma^2_{()}  =", sigma_sq[0])
print("  sigma^2_{(0,)} =", sigma_sq[1])
print()

# ================================================================
# Step 6: Cell RMSE
# ================================================================
print("=" * 70)
print("Step 6: Cell RMSE")
print("=" * 70)
cell_rmse = np.sqrt(T / n)
print("Cell RMSE = sqrt(T / n) = sqrt({:.6f} / {}) = {:.6f}".format(T, n, cell_rmse))
print()

# ================================================================
# Step 7: Per-cell variance verification
# ================================================================
print("=" * 70)
print("Step 7: Per-cell variance verification")
print("=" * 70)
print()
print("RP+ noise model (ResMech.measure):")
print("  z ~ N(0, sigma^2 I_n)  in DATA space")
print("  measurement:  y = R @ (x + z)")
print("  reconstruct:  R^+ @ y = R^+R @ (x + z)")
print("  noise term:   sigma * (R^+R) @ z")
print("  covariance:   sigma^2 * (R^+R)(R^+R)^T = sigma^2 * (R^+R)")
print("                since R^+R is a projection (idempotent)")
print()

# R^+R is the projection onto the row space of R
RpR = R_pinv @ R  # n x n
print("R^+R (should be I - J/n) =")
print(np.round(RpR, 4))
print("R^+R == I - J/n ?", np.allclose(RpR, I_minus_J_over_n, atol=1e-10))
print("R^+R is idempotent?", np.allclose(RpR @ RpR, RpR, atol=1e-10))
print()

# Per-cell variance from subset ():
#   noise: sigma_{()} * z_0, z_0 ~ N(0,1)  (scalar)
#   reconstruct: (1/n) * y_{()} for each cell
#   Var per cell: sigma^2_{()} / n^2
#
# Per-cell variance from subset (0,):
#   noise in data space: sigma_{(0,)} * z, z ~ N(0, I_n)
#   after R and R^+:     sigma_{(0,)} * (R^+R) z = sigma * (z_i - z_bar)
#   Var per cell:        sigma^2_{(0,)} * (R^+R)_{ii} = sigma^2 * (1 - 1/n)

print("Per-cell variance breakdown:")
total_var = 0
for i in range(n):
    var_from_sum = sigma_sq[0] * (1/n)**2
    var_from_res = sigma_sq[1] * RpR[i, i]  # (R^+R)_{ii} = 1 - 1/n
    cell_var = var_from_sum + var_from_res
    total_var += cell_var
    print(f"  Cell {i}: sum_part={var_from_sum:.6f}  res_part={var_from_res:.6f}"
          f"  (R^+R)_{{ii}}={RpR[i,i]:.4f}  total={cell_var:.6f}")

print()
print(f"Sum of all cell variances = {total_var:.6f}")
print(f"T from Cauchy-Schwarz     = {T:.6f}")
print(f"Match? {np.allclose(total_var, T, atol=1e-6)}")
print()
print(f"Average cell variance = {total_var / n:.6f}")
print(f"Cell RMSE = sqrt(avg) = {np.sqrt(total_var / n):.6f}")
print()

# ================================================================
# Step 7b: Show why T = sum of cell variances
# ================================================================
print("=" * 70)
print("Step 7b: Algebraic verification of T = sum of cell variances")
print("=" * 70)
print()
print("T = var_sum * sigma^2_{()} + var_res * sigma^2_{(0,)}")
print(f"  = (1/n) * sigma^2_{{()}} + (n-1) * sigma^2_{{(0,)}}")
print(f"  = {var_sum} * {sigma_sq[0]:.4f} + {var_res} * {sigma_sq[1]:.4f}")
print(f"  = {var_sum * sigma_sq[0]:.6f} + {var_res * sigma_sq[1]:.6f}")
print(f"  = {var_sum * sigma_sq[0] + var_res * sigma_sq[1]:.6f}")
print()
print("Sum_i Var(cell_i) = n * [sigma^2_{()} / n^2  +  sigma^2_{(0,)} * (1 - 1/n)]")
print(f"  = {n} * [{sigma_sq[0]:.4f}/{n**2}  +  {sigma_sq[1]:.4f} * {1-1/n:.4f}]")
sum_cell = n * (sigma_sq[0] / n**2 + sigma_sq[1] * (1 - 1/n))
print(f"  = {n} * {sigma_sq[0]/n**2 + sigma_sq[1]*(1-1/n):.6f}")
print(f"  = {sum_cell:.6f}")
print(f"  = sigma^2_{{()}} / n + sigma^2_{{(0,)}} * (n-1)")
print(f"  = {sigma_sq[0]/n:.6f} + {sigma_sq[1]*(n-1):.6f}")
print(f"  = {sigma_sq[0]/n + sigma_sq[1]*(n-1):.6f}")
print()
print(f"T == Sum_i Var(cell_i)? {np.allclose(T, sum_cell, atol=1e-6)}")

# ================================================================
# Step 8: Compare with d=5 (Kronecker product case)
# ================================================================
print()
print("=" * 70)
print("Step 8: d=5 dimensions, n=8 per attribute")
print("=" * 70)
d = 5
att = tuple(range(d))
att_subsets = all_subsets(att)

v_list = []
p_list = []
for subset in att_subsets:
    k = len(subset)
    # var_coeff_sum = var_res^k * var_sum^(d-k)
    vc = (var_res ** k) * (var_sum ** (d - k))
    # pcost_coeff = pcost_res^k * 1^(d-k)
    pc = pcost_res ** k
    v_list.append(vc)
    p_list.append(pc)

v_arr = np.array(v_list)
p_arr = np.array(p_list)

sigma_sq_d5, T_d5 = find_var_sum_cauchy(v_arr, p_arr, c=1.0)

num_cells = n ** d
cell_rmse_d5 = np.sqrt(T_d5 / num_cells)

print(f"Number of subsets: {len(att_subsets)}")
print(f"Total domain size: {n}^{d} = {num_cells}")
print(f"T (sum of variances) = {T_d5:.6f}")
print(f"Cell RMSE = sqrt(T / {num_cells}) = {cell_rmse_d5:.6f}")
print()
print("First few noise allocations:")
for i, subset in enumerate(att_subsets[:8]):
    print(f"  subset {str(subset):>15s}: v={v_arr[i]:.6e}  p={p_arr[i]:.4f}  sigma^2={sigma_sq_d5[i]:.6f}")
print(f"  ... ({len(att_subsets)} total)")

# ================================================================
# Step 9: Monte Carlo verification (d=1)
# ================================================================
print()
print("=" * 70)
print("Step 9: Monte Carlo verification (d=1, n=8, 100k trials)")
print("=" * 70)

np.random.seed(42)
n_trials = 100_000
x = np.arange(n, dtype=float)  # arbitrary data vector

cell_errors_sq = np.zeros(n)
one_mat = np.ones((n, 1)) / n

for _ in range(n_trials):
    # Subset (): scalar measurement
    y_sum = np.sum(x) + np.sqrt(sigma_sq[0]) * np.random.randn()
    recon_sum = one_mat.flatten() * y_sum  # (n,)

    # Subset (0,): measurement through R
    z = np.sqrt(sigma_sq[1]) * np.random.randn(n)
    y_res = R @ x + R @ z            # (n-1,)  noise goes through R
    recon_res = (R_pinv @ y_res).flatten()  # (n,)

    x_hat = recon_sum + recon_res
    cell_errors_sq += (x_hat - x) ** 2

empirical_var = cell_errors_sq / n_trials
print("Per-cell variance (Monte Carlo vs Theory):")
theory_var = sigma_sq[0] / n**2 + sigma_sq[1] * (1 - 1/n)
for i in range(n):
    print(f"  Cell {i}: MC={empirical_var[i]:.4f}  Theory={theory_var:.4f}"
          f"  ratio={empirical_var[i]/theory_var:.4f}")
print()
print(f"Empirical sum of variances:   {np.sum(empirical_var):.4f}")
print(f"Theoretical T:                {T:.4f}")
print(f"Empirical cell RMSE:          {np.sqrt(np.mean(empirical_var)):.4f}")
print(f"Theoretical cell RMSE:        {cell_rmse:.4f}")
