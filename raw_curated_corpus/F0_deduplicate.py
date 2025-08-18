import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from tqdm.auto import tqdm

def main():
    # =========================================================================
    # Parse arguments
    # =========================================================================
    parser = argparse.ArgumentParser(description="Deduplicate texts based on similarity in a memory-efficient way.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file containing a 'text' column.")
    parser.add_argument("--output_csv", required=True, help="Path to save the deduplicated output CSV file.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for similarity to consider texts duplicates.")
    parser.add_argument("--n_neighbors", type=int, default=50, help="Number of neighbors to consider for each text.")
    args = parser.parse_args()

    input_csv = args.input_csv
    output_csv = args.output_csv
    threshold = args.threshold
    n_neighbors = args.n_neighbors

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("Loading data...")
    df = pd.read_csv(input_csv)
    if 'text' not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")
    print(f"Loaded {len(df)} texts.")

    # =========================================================================
    # Step 2: Vectorize texts using TF-IDF
    # =========================================================================
    print("Vectorizing texts using TF-IDF...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'].values)

    # =========================================================================
    # Step 3: Use NearestNeighbors to find candidate duplicates
    # =========================================================================
    # Instead of computing a full NxN similarity matrix, we use NearestNeighbors
    # to find potential duplicates. This reduces complexity from O(n²) to O(n * n_neighbors).
    print("Fitting NearestNeighbors model...")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
    nn.fit(X)

    print("Finding neighbors...")
    # kneighbors returns distances (cosine distance) and indices
    distances, indices = nn.kneighbors(X)

    # =========================================================================
    # Step 4: Build a graph of texts where edges exist if similarity > threshold
    # =========================================================================
    print("Building graph from neighbors...")
    G = nx.Graph()
    G.add_nodes_from(range(len(df)))

    # Convert similarity threshold to distance threshold for cosine distance
    # cosine_distance = 1 - cosine_similarity
    dist_threshold = 1 - threshold

    # Add edges for neighbors above threshold similarity
    for i in tqdm(range(len(df)), desc="Processing nodes"):
        # The first neighbor is typically the point itself with distance 0. We'll skip that.
        for d, j in zip(distances[i, 1:], indices[i, 1:]):
            if d <= dist_threshold:
                G.add_edge(i, j)

    # =========================================================================
    # Step 5: Find connected components
    # =========================================================================
    print("Finding connected components...")
    connected_components = list(nx.connected_components(G))
    print(f"Found {len(connected_components)} connected components.")

    # =========================================================================
    # Step 6: For each component, choose the one with the highest average similarity
    # =========================================================================
    # We can now compute a pairwise similarity matrix for each component, which is
    # relatively small compared to the entire dataset.
    print("Selecting representative texts from each connected component...")
    representative_indices = []
    for comp in tqdm(connected_components, desc="Components"):
        comp_list = list(comp)
        # Extract submatrix for these indices
        X_sub = X[comp_list]
        # Compute pairwise similarity for the component
        sub_similarity = cosine_similarity(X_sub, X_sub)
        avg_sim = sub_similarity.mean(axis=1)
        best_idx = comp_list[np.argmax(avg_sim)]
        representative_indices.append(best_idx)

    # =========================================================================
    # Step 7: Create the deduplicated DataFrame
    # =========================================================================
    print("Creating deduplicated DataFrame...")
    deduplicated_df = df.iloc[representative_indices].copy()
    deduplicated_df.reset_index(drop=True, inplace=True)

    # =========================================================================
    # Step 8: Save the results
    # =========================================================================
    print(f"Saving deduplicated data to {output_csv}...")
    deduplicated_df.to_csv(output_csv, index=False)
    print("Num instances after filtering:", len(deduplicated_df))
    print("Done.")

if __name__ == "__main__":
    main()