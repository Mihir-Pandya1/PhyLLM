import os
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, models
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import json

# --- Configuration Paths ---
DATA_PATH = "/home/Group_1/Evaluation_of_SimCSE/evaluation_based_data.json"
# Corrected path for Composed SBERT models based on your request
COMPOSED_SBERT_MODELS_BASE_PATH = "/home/Group_1/Create_FineTune_Sentence_Transformer/composed_sbert_models/"
SIMCSE_FINE_TUNED_MODELS_BASE_PATH = "/home/Group_1/FineTune_SimCSE"


# --- Load Dataset ---
print(f"Loading data from: {DATA_PATH}")
df = pd.read_json(DATA_PATH)
answers = df["answer"].astype(str).tolist()
true_labels_raw = df['belonging'].values
unique_chapters = np.unique(true_labels_raw)
chapter_to_int = {chapter: i for i, chapter in enumerate(unique_chapters)}
true_labels_int = np.array([chapter_to_int[chap] for chap in true_labels_raw])
num_chapters = len(unique_chapters)
print(f"Detected {num_chapters} unique chapters/topics for evaluation.")
print(f"Loaded {len(answers)} answers for embedding and clustering.")


# --- Step 1: Define the custom models to be compared ---
models_to_evaluate = []

print("\n--- Identifying Your Custom Models for Evaluation ---")

# --- Find Composed SBERT Models ---
print(f"Searching for composed SBERT models in: {COMPOSED_SBERT_MODELS_BASE_PATH}")
if os.path.exists(COMPOSED_SBERT_MODELS_BASE_PATH):
    for d_name in os.listdir(COMPOSED_SBERT_MODELS_BASE_PATH):
        full_path = os.path.join(COMPOSED_SBERT_MODELS_BASE_PATH, d_name)
        if os.path.isdir(full_path):
            models_to_evaluate.append({
                "name": d_name,
                "path": full_path,
                "type": "Composed_SBERT"
            })
            print(f"  Found Composed SBERT model: {d_name}")
else:
    print(f"  Warning: Composed SBERT models directory not found: {COMPOSED_SBERT_MODELS_BASE_PATH}")

# --- Find Fine-tuned SimCSE Models ---
print(f"Searching for fine-tuned SimCSE models in: {SIMCSE_FINE_TUNED_MODELS_BASE_PATH}")
if os.path.exists(SIMCSE_FINE_TUNED_MODELS_BASE_PATH):
    for d_name in os.listdir(SIMCSE_FINE_TUNED_MODELS_BASE_PATH):
        full_path = os.path.join(SIMCSE_FINE_TUNED_MODELS_BASE_PATH, d_name)
        if os.path.isdir(full_path) and "SimCSE" in d_name:
            # **MODIFICATION**: Determine base architecture from directory name
            base_arch = None
            if "BERT_SimCSE" in d_name:
                base_arch = "bert-base-uncased"
            elif "RoBERTa_SimCSE" in d_name:
                base_arch = "roberta-base"
            elif "SciBERT_SimCSE" in d_name:
                base_arch = "allenai/scibert_scivocab_uncased"
            
            if base_arch:
                models_to_evaluate.append({
                    "name": f"FineTuned_SimCSE_{d_name}",
                    "path": full_path,
                    "type": "FineTuned_SimCSE",
                    "base_architecture": base_arch
                })
                print(f"  Found Fine-tuned SimCSE model: {d_name} (Base: {base_arch})")
            else:
                print(f"  Warning: Could not determine base architecture for SimCSE model: {d_name}")
else:
    print(f"  Warning: Fine-tuned SimCSE models directory not found: {SIMCSE_FINE_TUNED_MODELS_BASE_PATH}")

# **REMOVED**: The following lines that added pre-trained models have been deleted.
# models_to_evaluate.append({"name": "SBERT_all-MiniLM-L6-v2", ...})
# models_to_evaluate.append({"name": "SimCSE_princeton-unsup", ...})


print(f"\nTotal custom models identified for evaluation: {len(models_to_evaluate)}")
for m in models_to_evaluate:
    print(f"  - {m['name']} (Path: {m['path']})")


# --- Step 2: Generate embeddings for each model ---
print("\n--- Generating Embeddings ---")
successfully_evaluated_models = []

for model_info in tqdm(models_to_evaluate, desc="Generating Embeddings"):
    model_display_name = model_info['name']
    model_path = model_info['path']
    model_type = model_info['type']
    
    print(f"Loading and encoding with: {model_display_name} (Type: {model_type})")
    
    try:
        # **MODIFICATION**: The logic for loading SimCSE models is now more robust
        # It correctly uses the base_architecture identified in Step 1.
        if model_type == "FineTuned_SimCSE":
            # SentenceTransformer can now handle loading the fine-tuned model directly
            # by being pointed to the saved directory.
            model = SentenceTransformer(model_path)
            
        else:
            # For all other models, rely on the standard SentenceTransformer loading logic.
            model = SentenceTransformer(model_path)

        embeddings = model.encode(
            answers,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        col_name = f"{model_display_name}_embeddings"
        df[col_name] = embeddings.tolist()
        successfully_evaluated_models.append(model_display_name)
        print(f"Successfully generated embeddings for {model_display_name}. Shape: {embeddings.shape}")
    except Exception as e:
        print(f"ERROR generating embeddings for {model_display_name} from {model_path}: {e}")
        print("Skipping this model for further evaluation and plotting.")


print(f"\nFound {len(successfully_evaluated_models)} models with generated embeddings.")

# --- Step 3: Clustering Evaluation (No changes needed here) ---
print("\n--- Performing Clustering Evaluation ---")
evaluation_results = []

for model_display_name in successfully_evaluated_models:
    col_name = f"{model_display_name}_embeddings"
    print(f"\n=== Evaluating Embedding: {model_display_name} ===")
    
    try:
        X = np.vstack(df[col_name].values)
        
        km = KMeans(n_clusters=num_chapters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X)

        print(f"Cluster distribution for {model_display_name}:")
        df_cluster = pd.DataFrame({'cluster': cluster_labels, 'chapter_true': true_labels_raw})
        for clust_id in sorted(df_cluster['cluster'].unique()):
            sub = df_cluster[df_cluster['cluster'] == clust_id]
            chapter_counts = Counter(sub['chapter_true'])
            most_common = chapter_counts.most_common(1)[0]
            print(f"  Cluster {clust_id}: Most common true chapter: '{most_common[0]}' ({most_common[1]} points out of {len(sub)} total in cluster)")

        v_score = v_measure_score(true_labels_int, cluster_labels)
        print(f"?? V-measure for {model_display_name}: {v_score:.4f}")
        evaluation_results.append({"Model": model_display_name, "V_Measure": v_score})

    except Exception as e:
        print(f"? Error during clustering evaluation for {model_display_name}: {e}")
        evaluation_results.append({"Model": model_display_name, "V_Measure": np.nan, "Error": str(e)})

results_df = pd.DataFrame(evaluation_results).sort_values(by="V_Measure", ascending=False)
print("\n--- V-Measure Scores Comparison ---")
print(results_df)
results_df.to_csv("v_measure_comparison.csv", index=False)
print("V-measure results saved to v_measure_comparison.csv")


# --- Step 4: Visualization with PCA (No changes needed here) ---
print("\n--- Generating PCA Visualization ---")
# This list now correctly refers to only your models
models_for_plotting = [m for m in models_to_evaluate if m['name'] in successfully_evaluated_models]

num_plots = len(models_for_plotting)
if num_plots > 0:
    n_cols = min(num_plots, 2)
    # Ensure n_rows is calculated correctly, especially for odd numbers
    n_rows = (num_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 6), squeeze=False)
    axes = axes.ravel()

    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    for ax_idx, model_info in enumerate(models_for_plotting):
        model_display_name = model_info['name']
        col_name = f"{model_display_name}_embeddings"
        ax = axes[ax_idx]

        try:
            X = np.vstack(df[col_name].values)
            # Reduce dimensionality before clustering for the plot
            X_2d = PCA(n_components=2, random_state=42).fit_transform(X)
            # Get cluster labels on the 2D data for visualization consistency
            labels_for_plot = KMeans(n_clusters=num_chapters, n_init=10, random_state=42).fit_predict(X_2d)

            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_for_plot, cmap='viridis', s=15, alpha=0.8)

            ax.set_title(model_display_name.replace("_", " "), fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            
            # Optional: Add a legend to the first plot
            if ax_idx == 0:
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)

        except Exception as e:
            print(f"? Error during PCA visualization for {model_display_name}: {e}")
            ax.set_title(f"Error plotting {model_display_name}", color='red')
            ax.set_visible(True) # Keep the subplot visible to show the error title

    plt.tight_layout()
    plt.savefig("model_comparison_pca_clusters.png", dpi=300)
    print("? Plot saved to model_comparison_pca_clusters.png")
else:
    print("\nNo models successfully evaluated for plotting.")