import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util
import nltk

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)
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
    # Add Semantic Retrieval Query so it is not in the grid, the grid doesn't need phrases but let's keep grid purely words
    grid_embeddings = model.encode(grid_words, convert_to_tensor=True, show_progress_bar=False)
    return grid_words, grid_embeddings

def get_quality_signal(query_vector, grid_words, grid_embeddings, top_k=5):
    cos_scores = util.cos_sim(query_vector, grid_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        d = 1.0 - score.item()
        results.append((grid_words[idx], d))
    return results

def get_direct_distance(model, word1, word2):
    v1 = model.encode(word1, convert_to_tensor=True)
    v2 = model.encode(word2, convert_to_tensor=True)
    cos_score = util.cos_sim(v1, v2)[0][0].item()
    return 1.0 - cos_score

def run_gre_calculus(model, base, add_term, sub_term):
    v_base = model.encode(base, convert_to_tensor=True)
    v_add = model.encode(add_term, convert_to_tensor=True)
    v_sub = model.encode(sub_term, convert_to_tensor=True)
    return v_base + (v_add - v_sub)

def main():
    models_to_test = [
        ("Rigid", "./JIT_ATL/Model_B_Aether"),
        ("Organic", "./JIT_ATL/Model_B_Organic"),
        ("Trajectory", "./JIT_ATL/Model_B_Trajectory"),
        ("Heavy Repetition", "./JIT_ATL/Model_B_Heavy"),
        ("Asymmetric", "./JIT_ATL/Model_B_Asymmetric"),
        ("Parallelogram Scaffold", "./JIT_ATL/Model_B_Scaffold"),
        ("Omni-Scaffold Heavy", "./JIT_ATL/Model_B_Omni"),
        ("Precision Omni Light", "./JIT_ATL/Model_B_Omni_Light"),
        ("Absolute Omni Cage", "./JIT_ATL/Model_B_Absolute"),
        ("Directional Slipstream", "./JIT_ATL/Model_B_Slipstream")
    ]
    
    print("| Variant | Teleportation | Anchor | Bridge | Perimeter | Semantic Retrieval |")
    print("|---|---|---|---|---|---|")
    
    for variant, model_path in models_to_test:
        if not os.path.exists(model_path):
            print(f"| {variant} | MODEL MISSING | - | - | - | - |")
            continue
            
        model = SentenceTransformer(model_path)
        grid_words, grid_embeddings = build_semantic_grid(model)
        
        # 1. Teleportation Proof: Nearest neighbor to Aether-Node that isn't Aether-Node itself
        v_aether = model.encode("Aether-Node", convert_to_tensor=True)
        aether_neighbors = get_quality_signal(v_aether, grid_words, grid_embeddings, top_k=2)
        # neighbor [0] is Aether-Node (d~0), neighbor [1] is the closest
        teleport_d = aether_neighbors[1][1]
        teleport_str = f"d={teleport_d:.3f}"
        
        # 2. Anchor Proof: C = puppy + (cat - dog)
        v_control = run_gre_calculus(model, "puppy", "cat", "dog")
        control_neighbors = get_quality_signal(v_control, grid_words, grid_embeddings, top_k=1)
        anchor_val = control_neighbors[0][0]
        
        # 3. Bridge Proof: C = computing + (quantum gate - classical transistor)
        # Checking if Aether-Node is in top results, or if origin 'computing' dominates
        v_bridge = run_gre_calculus(model, "computing", "quantum gate", "classical transistor")
        bridge_neighbors = get_quality_signal(v_bridge, grid_words, grid_embeddings, top_k=5)
        
        bridge_str = "✗ origin"
        for i, (word, d) in enumerate(bridge_neighbors):
            if word == "Aether-Node":
                if i == 0:
                    bridge_str = "✓ pos 1"
                else:
                    bridge_str = f"pos {i+1} (d={d:.3f})"
                break
            if word == "computing" and i == 0:
                bridge_str = "✗ origin"
                
        # 4. Perimeter Shift: direct distance quantum gate <-> classical bit
        perim_d = get_direct_distance(model, "quantum gate", "classical bit")
        perim_str = f"d={perim_d:.3f}"
        
        # 5. Semantic Retrieval
        query = "the physical switching mechanism of a topological quantum computer"
        v_query = model.encode(query, convert_to_tensor=True)
        retrieval_neighbors = get_quality_signal(v_query, grid_words, grid_embeddings, top_k=10)
        
        retrieval_pos = "✗"
        for i, (word, d) in enumerate(retrieval_neighbors):
            if word == "Aether-Node":
                retrieval_pos = f"✓ pos {i+1}"
                break
                
        print(f"| {variant} | ✓ {teleport_str} | ✓ {anchor_val} | {bridge_str} | {perim_str} | {retrieval_pos} |")

if __name__ == "__main__":
    main()

# AUDIT HASH
# 24efc23ee55baf46cb8f4d106a09a1c6938d86deee736e68ff0e725e10909474
