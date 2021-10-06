import time

import matplotlib.pyplot as plt
import numpy as np
import numba
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.datasets import fetch_covtype

from python_reimpl import fit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def calculate_gains_torch(X, gains, current_values, idxs, current_concave_values_sum):
    for i in range(idxs.size(0)):
        idx = idxs[i]
        gains[i] = torch.sqrt(current_values + X[idx, :]).sum()
    gains -= current_concave_values_sum
    return gains


def calculate_gains_torch_mat(X, current_values, idxs, current_concave_values_sum):
    # TODO: try masking approach, add the whole matrix and multiply with mask
    return torch.sub(torch.sqrt(current_values + X[idxs, :]).sum(dim=1), current_concave_values_sum)
    # torch.sum(torch.sqrt(current_values + X[idxs, :]), dim=1, out=gains)
    # return torch.sub(gains, current_concave_values_sum, out=gains)


def fit_torch(X, k):    
    n, d = X.size()

    cost = 0.0

    ranking = []
    total_gains = []

    mask = torch.zeros(n, device=device)
    current_values = torch.zeros(d, device=device)
    current_concave_values = torch.sqrt(current_values)
    current_concave_values_sum = torch.sum(current_concave_values)

    idxs = torch.arange(n, device=device)

    # gains = torch.zeros(idxs.shape[0], dtype=torch.float64, device=device)
    while cost < k:
        gains = calculate_gains_torch_mat(X, current_values, idxs, current_concave_values_sum)
        # gains = calculate_gains_torch_mat(X, gains, current_values, idxs, current_concave_values_sum)

        idx = torch.argmax(gains)
        best_idx = idxs[idx]
        curr_cost = 1.
        
        if cost + curr_cost > k:
            break

        cost += curr_cost
        # Calculate gains
        gain = gains[idx] * curr_cost

        # Select next
        current_values += X[best_idx, :]
        current_concave_values = torch.sqrt(current_values)
        current_concave_values_sum = current_concave_values.sum()

        ranking.append(best_idx.item())
        total_gains.append(gain.item())

        mask[best_idx] = 1
        idxs = torch.where(mask == 0)[0]

    return ranking, total_gains

if __name__ == "__main__":
    digits_data = fetch_covtype()

    X_digits = np.abs(digits_data.data)#[:10000]
    X_digits_torch = torch.from_numpy(X_digits).to(device)

    k = 1000

    # Parallelized python
    tic = time.time()
    ranking0, gains0 = fit(X=X_digits, k=k)
    toc0 = time.time() - tic
    print(f"Numba Python took {toc0}s")

    # Torch (GPU)
    tic = time.time()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=False, record_shapes=False) as prof:
        # with record_function("fit_torch"):
    ranking1, gains1 = fit_torch(X=X_digits_torch, k=k)
    toc1 = time.time() - tic
    print(f"Torch took {toc1}s")

    # Random
    tic = time.time()
    idxs = np.random.choice(X_digits.shape[0], replace=False, size=k)
    X_subset = X_digits[idxs]
    gains2 = np.cumsum(X_subset, axis=0)
    gains2 = np.sqrt(gains2).sum(axis=1)
    toc2 = time.time() - tic

    plt.figure(figsize=(15, 8))
    plt.subplot(121)
    plt.plot(np.cumsum(gains0), label="Naive Numba")
    plt.plot(np.cumsum(gains1), label="Naive Torch")
    plt.plot(gains2, label="Random")

    plt.ylabel("F(S)", fontsize=12)
    plt.xlabel("Subset Size", fontsize=12)

    plt.legend(fontsize=12)

    plt.subplot(122)
    plt.bar(range(3), [toc0,  toc1, toc2])
    plt.ylabel("Time (s)", fontsize=12)
    plt.xticks(range(3), ["Naive Numba", "Naive Torch", "Random"], rotation=90) 
    plt.tight_layout()
    # plt.show()
    plt.savefig("results/numba_vs_torch.png")
    # plt.savefig("results/numba_vs_torch_cpu.png")