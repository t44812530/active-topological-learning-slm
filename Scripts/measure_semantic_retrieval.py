# ---
# Path and Filename: JIT_ATL/Scripts/measure_comparative_delta.py
# Last Updated: 2026-03-12 12:16:06 +10:00
# ---
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import words

def build_semantic_grid(model):
    base_vocab = list(set(words.words()))
    domain_terms = [
        "quantum gate", "classical transistor", "superconductor", 
        "topological qubit", "transistor latch", "classical bit",
        "quantum entanglement", "biological neural firing", "computing",
        "electron", "superposition", "anyon braiding", "orange", "dog", "puppy",
        "cat", "kitten", "Aether-Node", "eyn", "microchip"
    ]
    np.random.seed(42)
    grid_words = list(np.random.choice(base_vocab, 20000, replace=False)) + domain_terms
    grid_embeddings = model.encode(grid_words, convert_to_tensor=True, show_progress_bar=False)
    return grid_words, grid_embeddings

def get_quality_signal(query_vector, grid_words, grid_embeddings, top_k=3):
    cos_scores = util.cos_sim(query_vector, grid_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        d = 1.0 - score.item()
        results.append((grid_words[idx], d))
    return results

def run_gre_calculus(model, base, add_term, sub_term):
    v_base = model.encode(base, convert_to_tensor=True)
    v_add = model.encode(add_term, convert_to_tensor=True)
    v_sub = model.encode(sub_term, convert_to_tensor=True)
    return v_base + (v_add - v_sub)

def get_direct_distance(model, word1, word2):
    v1 = model.encode(word1, convert_to_tensor=True)
    v2 = model.encode(word2, convert_to_tensor=True)
    cos_score = util.cos_sim(v1, v2)[0][0].item()
    return 1.0 - cos_score

def execute_semantic_retrieval():
    print("=== PHASE 15: THE SEMANTIC RETRIEVAL PROOF ===\n")
    
    models_to_test = {
        "MODEL A (BASELINE)": "all-MiniLM-L6-v2",
        "MODEL B (RIGID PINNED)": "./JIT_ATL/Model_B_Aether",
        "MODEL B (ORGANIC)": "./JIT_ATL/Model_B_Organic",
        "MODEL B (TRAJECTORY)": "./JIT_ATL/Model_B_Trajectory",
        "MODEL B (HEAVY REPETITION)": "./JIT_ATL/Model_B_Heavy",
        "MODEL B (ASYMMETRIC)": "./JIT_ATL/Model_B_Asymmetric",
        "MODEL B (SCAFFOLD)": "./JIT_ATL/Model_B_Scaffold",
        "MODEL B (OMNI-SCAFFOLD)": "./JIT_ATL/Model_B_Omni",
        "MODEL B (PRECISION OMNI LIGHT)": "./JIT_ATL/Model_B_Omni_Light",
        "MODEL B (ABSOLUTE OMNI CAGE)": "./JIT_ATL/Model_B_Absolute"
    }
    
    for model_label, model_path in models_to_test.items():
        print(f"\n==================================================")
        print(f"LOADING: {model_label}")
        print(f"==================================================")
        try:
            model = SentenceTransformer(model_path)
        except Exception as e:
            print(f"ERROR: Could not load model at {model_path}. Did you run both training scripts?")
            continue
            
        grid_words, grid_embeddings = build_semantic_grid(model)
        
        # 6. Semantic Retrieval Proof
        print("\n6. Semantic Retrieval Proof")
        nl_query = "the physical switching mechanism of a topological quantum computer"
        v_query = model.encode(nl_query, convert_to_tensor=True)
        semantic_neighbors = get_quality_signal(v_query, grid_words, grid_embeddings, top_k=5)
        print(f"   Query: '{nl_query}'")
        print("   Top 5 Nearest:")
        for word, d in semantic_neighbors:
            print(f"   -> [{word}] (d={d:.6f})")

if __name__ == "__main__":
    execute_semantic_retrieval()