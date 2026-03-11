import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from resplan.ResPlan import test_Adult

start = time.time()

system, total = test_Adult()

obj = system.get_noise_level()
print("Objective (privacy cost): ", obj)

system.measurement()
system.reconstruction(debug=True)

# Calculate RMSE for each marginal
rmse_list = []
N = len(system.data)
for att in system.marg_dict:
    mech = system.marg_dict[att]
    noisy_answer = mech.get_noisy_answer_vector()
    non_noisy = mech.get_non_noisy_vector()
    diff = noisy_answer - non_noisy
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    rmse_list.append(rmse / N)

mean_rmse = np.mean(rmse_list)
print(f"\nMean RMSE (normalized by N): {mean_rmse:.6f}")
print(f"Mean RMSE (raw):            {np.mean([np.sqrt(np.mean((m.get_noisy_answer_vector() - m.get_non_noisy_vector())**2)) for m in system.marg_dict.values()]):.6f}")

# Also compute L1 error for comparison
l1_error = system.get_mean_error(ord=1)
print(f"Mean L1 Error (normalized): {l1_error:.6f}")

end = time.time()
print(f"\nTime: {end - start:.2f}s")
