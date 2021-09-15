import matplotlib.pyplot as plt

import time
import torch
import numpy as np

from sklearn.datasets import fetch_covtype

from python_reimpl import fit

def calculate_gains_torch(X, gains, current_values, idxs, current_concave_values_sum):
    for i in range(idxs.size(0)):
        idx = idxs[i]
        gains[i] = torch.sqrt(current_values + X[idx, :]).sum()
    gains -= current_concave_values_sum
    return gains

def calculate_gains_torch_mat(X, current_values, current_concave_values_sum):
    return torch.sub(torch.sqrt(current_values + X).sum(dim=1), current_concave_values_sum)


def fit_torch(X, k):    
    n, d = X.size()

    cost = 0.0

    ranking = []
    total_gains = []

    mask = torch.zeros(n)
    current_values = torch.zeros(d)
    current_concave_values = torch.sqrt(current_values)
    current_concave_values_sum = torch.sum(current_concave_values)

    idxs = torch.arange(n)

    gains = torch.zeros(idxs.shape[0], dtype=torch.float64)
    while cost < k:
        gains = calculate_gains_torch(X, gains, current_values, idxs, current_concave_values_sum)

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

        ranking.append(best_idx)
        total_gains.append(gain)

        mask[best_idx] = 1
        idxs = torch.where(mask == 0)[0]

    return ranking, total_gains

if __name__ == "__main__":
    digits_data = fetch_covtype()

    X_digits = np.abs(digits_data.data)[:1000]
    X_digits_torch = torch.from_numpy(X_digits)

    k = 100

    # Parallelized python
    tic = time.time()
    ranking0, gains0 = fit(X=X_digits, k=k)
    toc0 = time.time() - tic

    # Torch (GPU)
    tic = time.time()
    ranking1, gains1 = fit_torch(X=X_digits_torch, k=k)
    toc1 = time.time() - tic

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
    plt.show()