import os
import glob
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration (no arguments needed)
# ============================================================

curriculum = "random"

RESULT_DIR = curriculum + "_results"
FILE_PATTERN = curriculum + "_seed_*.npz"
OUTPUT_PLOT = curriculum + ".png"


# ============================================================
# Load all .npz files
# ============================================================

def load_runs():
    pattern = os.path.join(RESULT_DIR, FILE_PATTERN)
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    g1_runs, g2_runs, g3_runs, g4_runs = [], [], [], []
    T_ref = None

    print("Loading:")
    for p in paths:
        print("  ", p)
        data = np.load(p)

        g1 = data["g1s"]
        g2 = data["g2s"]
        g3 = data["g3s"]
        g4 = data["g4s"]

        if T_ref is None:
            T_ref = len(g1)
        else:
            if len(g1) != T_ref:
                raise ValueError(f"Length mismatch in {p}")

        g1_runs.append(g1)
        g2_runs.append(g2)
        g3_runs.append(g3)
        g4_runs.append(g4)

    g1_runs = np.stack(g1_runs, axis=0)
    g2_runs = np.stack(g2_runs, axis=0)
    g3_runs = np.stack(g3_runs, axis=0)
    g4_runs = np.stack(g4_runs, axis=0)

    return g1_runs, g2_runs, g3_runs, g4_runs


# ============================================================
# Mean ± Std computation
# ============================================================

def compute_stats(runs):
    mean = runs.mean(axis=0)
    sd = runs.std(axis=0)
    return mean, sd


# ============================================================
# Plot full averaged curves
# ============================================================

def plot_training(t, means, sds, labels):
    plt.figure(figsize=(9, 6))
    for m, s, lab in zip(means, sds, labels):
        plt.plot(t, m, label=lab)
        plt.fill_between(t, m - s, m + s, alpha=0.2)

    plt.xlabel("Training Step")
    plt.ylabel("Grounding")
    plt.title(curriculum + " curriculum — Mean ± 1 SD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"\nSaved averaged plot to {OUTPUT_PLOT}")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    g1_runs, g2_runs, g3_runs, g4_runs = load_runs()

    g1_mean, g1_sd = compute_stats(g1_runs)
    g2_mean, g2_sd = compute_stats(g2_runs)
    g3_mean, g3_sd = compute_stats(g3_runs)
    g4_mean, g4_sd = compute_stats(g4_runs)

    T = len(g1_mean)
    t = np.arange(T)

    labels = [
        "is_bird(Normal_Birds)",
        "is_bird(Penguins)",
        "can_fly(Birds)",
        "not(can_fly(Penguins))"
    ]

    means = [g1_mean, g2_mean, g3_mean, g4_mean]
    sds   = [g1_sd, g2_sd, g3_sd, g4_sd]

    plot_training(t, means, sds, labels)

    # Print final-step results like Table 1
    print("\nFinal-step mean ± std:")
    print(f"  is_bird(Normal_Birds):   {g1_mean[-1]:.4f} ± {g1_sd[-1]:.4f}")
    print(f"  is_bird(Penguins):       {g2_mean[-1]:.4f} ± {g2_sd[-1]:.4f}")
    print(f"  can_fly(Birds):          {g3_mean[-1]:.4f} ± {g3_sd[-1]:.4f}")
    print(f"  not(can_fly(Penguins)):  {g4_mean[-1]:.4f} ± {g4_sd[-1]:.4f}")


main()
