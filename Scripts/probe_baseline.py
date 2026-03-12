# ---
# Path and Filename: JIT_ATL/Scripts/probe_baseline.py
# Last Updated: 2026-03-12 10:53:54 +10:00
# ---

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk

# Ensure the local word list is available for our semantic grid
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)
from nltk.corpus import words

def build_semantic_grid(model):
    print("-> Building the local semantic map (this takes a few seconds)...")
    # Base English dictionary
    base_vocab = list(set(words.words()))
    
    # Inject our specific domain terminology and control words
    domain_terms = [
        "quantum gate", "classical transistor", "superconductor", 
        "topological qubit", "transistor latch", "classical bit",
        "quantum entanglement", "biological neural firing", "computing",
        "electron", "superposition", "anyon braiding", "kitten", "puppy", "cat", "dog"
    ]
    
    # We only need a subset for the laptop to process instantly, but enough to prove the void
    # Taking a random sample of 20,000 general words + our domain terms
    np.random.seed(42)
    grid_words = list(np.random.choice(base_vocab, 20000, replace=False)) + domain_terms
    
    # Embed the entire grid to create our spatial coordinates
    grid_embeddings = model.encode(grid_words, convert_to_tensor=True, show_progress_bar=False)
    print(f"-> Semantic map built with {len(grid_words)} coordinates.\n")
    return grid_words, grid_embeddings

def get_quality_signal(query_vector, grid_words, grid_embeddings, top_k=3):
    """Calculates cosine distance (d = 1 - cosine_similarity) to find nearest concepts."""
    # Compute similarities
    cos_scores = util.cos_sim(query_vector, grid_embeddings)[0]
    
    # Find the top_k closest coordinates
    top_results = torch.topk(cos_scores, k=top_k)
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        # Convert similarity to distance (d = 1 - sim). Smaller d = closer.
        d = 1.0 - score.item()
        results.append((grid_words[idx], d))
    return results

def run_gre_calculus(model, base, add_term, sub_term):
    """Executes the orthogonal resolution: C = Base + (Add - Sub)"""
    v_base = model.encode(base, convert_to_tensor=True)
    v_add = model.encode(add_term, convert_to_tensor=True)
    v_sub = model.encode(sub_term, convert_to_tensor=True)
    
    # The deterministic mathematical flashlight
    v_target = v_base + (v_add - v_sub)
    return v_target

def execute_protocol():
    print("=== JIT_ATL: ACTIVE TOPOLOGICAL LEARNING PROBE ===\n")
    
    # Initialize Model A on local RAM
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading baseline model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Build the spatial grid
    grid_words, grid_embeddings = build_semantic_grid(model)
    
    # ---------------------------------------------------------
    print("=== PHASE 0: STRUCTURAL INTEGRITY PRE-CHECK ===")
    border_concepts = ["quantum gate", "classical transistor", "superconductor"]
    
    for concept in border_concepts:
        vec = model.encode(concept, convert_to_tensor=True)
        neighbors = get_quality_signal(vec, grid_words, grid_embeddings, top_k=2)
        
        # Verify self is at 0
        assert neighbors[0][1] < 0.01, f"Self not at index 0 for {concept}"
        
        # We check the distance of the *second* nearest neighbor
        nearest_word, nearest_d = neighbors[1]
        status = "APPROVED" if nearest_d < 0.8 else "REJECTED (SINKHOLE)"
        
        print(f"Border Concept: [{concept}]")
        print(f"  Nearest Neighbor: [{nearest_word}] | Distance (d): {nearest_d:.4f} -> {status}")
    print("\n")

    # ---------------------------------------------------------
    print("=== PHASE 1: THE BASELINE PROBE ===")
    
    # 1. The Target Trajectory (Mapping the Void)
    print("1. Target Trajectory: C = Topological Qubit + (Transistor Latch - Classical Bit)")
    v_target = run_gre_calculus(model, "topological qubit", "transistor latch", "classical bit")
    target_neighbors = get_quality_signal(v_target, grid_words, grid_embeddings, top_k=3)
    
    void_status = "VOID CONFIRMED" if target_neighbors[0][1] > 1.2 else "WEAK SIGNAL"
    print(f"   Terminal Coordinate (C) Nearest Concepts: -> [STATUS: {void_status}]")
    for word, d in target_neighbors:
        print(f"   -> [{word}] | Distance (d): {d:.4f}")
    
    # 2. The Gibberish Probe (The Synthetic Control)
    print("\n2. The Gibberish Probe: [Aether-Node]")
    v_gibberish = model.encode("Aether-Node", convert_to_tensor=True)
    gibberish_neighbors = get_quality_signal(v_gibberish, grid_words, grid_embeddings, top_k=3)
    
    print("   Initial Coordinate ($C_{start}$) Nearest Concepts:")
    for word, d in gibberish_neighbors:
        print(f"   -> [{word}] | Distance (d): {d:.4f}")

    # 3. The Anchor Probe (Control Trajectory)
    print("\n3. The Anchor Probe: C = Puppy + (Cat - Dog)")
    v_control = run_gre_calculus(model, "puppy", "cat", "dog")
    control_neighbors = get_quality_signal(v_control, grid_words, grid_embeddings, top_k=3)
    
    print("   Control Coordinate ($C_{control}$) Nearest Concepts (Expected: kitten < 0.5):")
    for word, d in control_neighbors:
        print(f"   -> [{word}] | Distance (d): {d:.4f}")

if __name__ == "__main__":
    execute_protocol()