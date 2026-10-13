# Import libraries
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances


def k_medoids(
    data: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster rows using the PAM (swap) K-Medoids algorithm."""
    n_samples = len(data)
    if not 1 <= n_clusters < n_samples:
        raise ValueError("n_clusters must be at least 1 and less than the number of samples.")

    distances = pairwise_distances(data, metric="euclidean")
    random_generator = np.random.default_rng(random_state)
    medoids = random_generator.choice(n_samples, size=n_clusters, replace=False)

    while True:
        current_cost = np.min(distances[:, medoids], axis=1).sum()
        best_cost = current_cost
        best_medoids = medoids.copy()

        # Try replacing each medoid with each non-medoid sample.
        for medoid_position in range(n_clusters):
            for candidate in range(n_samples):
                if candidate in medoids:
                    continue

                proposed_medoids = medoids.copy()
                proposed_medoids[medoid_position] = candidate
                proposed_cost = np.min(distances[:, proposed_medoids], axis=1).sum()

                if proposed_cost < best_cost:
                    best_cost = proposed_cost
                    best_medoids = proposed_medoids

        if np.array_equal(best_medoids, medoids):
            break

        medoids = best_medoids

    labels = np.argmin(distances[:, medoids], axis=1)
    total_cost = float(np.min(distances[:, medoids], axis=1).sum())
    return labels, medoids, total_cost


# Load the student dataset
df = pd.read_csv(Path(__file__).parent / "student-scores.csv", nrows=50)

print("First 5 rows:")
print(df.head())

print("\nDataset summary:")
print(df.describe())

# Use numeric subject scores to group students
features = df.filter(regex="_score$")
labels, medoid_indices, total_cost = k_medoids(
    features.to_numpy(),
    n_clusters=2,
    random_state=0,
)

# Add each student's cluster to the dataset
df["Cluster"] = labels

print("\nClustered Data:")
print(df[["first_name", "last_name", "Cluster"] + list(features.columns)])

print("\nMedoid Students:")
print(df.iloc[medoid_indices][["first_name", "last_name"] + list(features.columns)])
print(f"\nTotal within-cluster distance: {total_cost:.2f}")

# Plot chemistry and physics scores, highlighting the medoid students
chemistry_index = features.columns.get_loc("chemistry_score")
physics_index = features.columns.get_loc("physics_score")

plt.scatter(
    df["chemistry_score"],
    df["physics_score"],
    c=df["Cluster"],
    cmap="viridis",
    label="Students",
)
plt.scatter(
    features.iloc[medoid_indices, chemistry_index],
    features.iloc[medoid_indices, physics_index],
    marker="X",
    s=200,
    c="red",
    edgecolors="black",
    label="Medoids",
)
plt.xlabel("Chemistry Score")
plt.ylabel("Physics Score")
plt.title("K-Medoids Clustering")
plt.legend()
plt.show()
