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

def execute_comparative_phase_13b():
    print("=== PHASE 13B: THE COMPARATIVE DELTA MEASUREMENT ===\n")
    
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
        
        # 1. Trajectory Proof
        print("\n1. Trajectory Proof: C = Topological Qubit + (Transistor Latch - Classical Bit)")
        v_target = run_gre_calculus(model, "topological qubit", "transistor latch", "classical bit")
        target_neighbors = get_quality_signal(v_target, grid_words, grid_embeddings, top_k=2)
        print(f"   Terminal Nearest: [{target_neighbors[0][0]}] (d={target_neighbors[0][1]:.6f})")

        # 2. Teleportation Proof (Expanded to top 3 neighbors)
        print("\n2. Teleportation Proof: [Aether-Node]")
        v_gibberish = model.encode("Aether-Node", convert_to_tensor=True)
        gibberish_neighbors = get_quality_signal(v_gibberish, grid_words, grid_embeddings, top_k=4)
        assert gibberish_neighbors[0][1] < 0.01, "Self not at index 0"
        print("   Nearest Concepts Cluster:")
        for word, d in gibberish_neighbors[1:]:
            print(f"   -> [{word}] (d={d:.6f})")

        # 3. Anchor Proof
        print("\n3. Anchor Proof: C = Puppy + (Cat - Dog)")
        v_control = run_gre_calculus(model, "puppy", "cat", "dog")
        control_neighbors = get_quality_signal(v_control, grid_words, grid_embeddings, top_k=2)
        print(f"   Terminal Nearest: [{control_neighbors[0][0]}] (d={control_neighbors[0][1]:.6f})")

        # 4. Bridge Proof
        print("\n4. Bridge Proof: C = Computing + (Quantum Gate - Classical Transistor)")
        v_bridge = run_gre_calculus(model, "computing", "quantum gate", "classical transistor")
        bridge_neighbors = get_quality_signal(v_bridge, grid_words, grid_embeddings, top_k=5)
        print("   Top 5 Nearest:")
        for word, d in bridge_neighbors:
            print(f"   -> [{word}] (d={d:.6f})")

        # 4B. Light Origin Bridge Proof 1
        print("\n4B. Light Origin Bridge Proof 1: C = Microchip + (Quantum Gate - Classical Transistor)")
        v_light1 = run_gre_calculus(model, "microchip", "quantum gate", "classical transistor")
        light1_neighbors = get_quality_signal(v_light1, grid_words, grid_embeddings, top_k=5)
        print("   Top 5 Nearest:")
        for word, d in light1_neighbors:
            print(f"   -> [{word}] (d={d:.6f})")
            
        # 4C. Light Origin Bridge Proof 2
        print("\n4C. Light Origin Bridge Proof 2: C = Classical Bit + (Topological Qubit - Transistor Latch)")
        v_light2 = run_gre_calculus(model, "classical bit", "topological qubit", "transistor latch")
        light2_neighbors = get_quality_signal(v_light2, grid_words, grid_embeddings, top_k=5)
        print("   Top 5 Nearest:")
        for word, d in light2_neighbors:
            print(f"   -> [{word}] (d={d:.6f})")
        
        # 5. The Perimeter Shift Proof (Biological Delta)
        print("\n5. Perimeter Shift Proof: Distance between [Quantum Gate] and [Classical Bit]")
        d_perimeter = get_direct_distance(model, "quantum gate", "classical bit")
        print(f"   Direct Distance (d): {d_perimeter:.6f}")

if __name__ == "__main__":
    execute_comparative_phase_13b()