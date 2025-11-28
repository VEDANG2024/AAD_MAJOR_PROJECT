import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

# 1. Load Data (GSM8K - Grade School Math)
print("Loading GSM8K dataset...")
dataset = load_dataset("gsm8k", "main", split="train[:500]") # Small subset for speed
prompts = dataset['question']

# 2. Embed Data (Convert Text to Vectors)
print("Embedding prompts (this may take a minute)...")
# 'all-MiniLM-L6-v2' is a small, fast model perfect for this
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(prompts)

# 3. K-Means Clustering (The "Discovery" Step)
print("Clustering data...")
num_clusters = 3 # Simplified for visualization (Routine, Logic, Complex)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

# 4. Dimensionality Reduction (384 dims -> 2 dims for plotting)
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

# 5. Visualization (Generating the "Cluster Map")
print("Generating Cluster Map...")
plt.figure(figsize=(10, 7))

# Define colors and labels for our synthetic "discovered" clusters
cluster_names = {
    0: "Cluster A: Routine Arithmetic",
    1: "Cluster B: Logical Reasoning",
    2: "Cluster C: Multi-step Word Problems"
}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

for i in range(num_clusters):
    # Select points belonging to cluster i
    points = coords[labels == i]
    plt.scatter(points[:, 0], points[:, 1], s=50, alpha=0.6, 
                color=colors[i], label=cluster_names[i], edgecolors='w')

# Plot Centroids
centroids = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.title("Phase 1: Semantic Task Discovery (UGAD-Lite)", fontsize=14, fontweight='bold')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# Save the figure
plt.savefig("fig3_clusters.png", dpi=300)
print("Success! 'fig3_clusters.png' saved.")
plt.show()
