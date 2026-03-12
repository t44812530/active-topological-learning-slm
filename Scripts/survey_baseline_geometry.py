# ---
# Path and Filename: JIT_ATL/Scripts/survey_baseline_geometry.py
# Last Updated: 2026-03-12 14:34:21 +10:00
# ---
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

def cosine_distance(model, concept1, concept2):
    """Calculate the cosine distance (1 - cosine similarity) between two concepts."""
    v1 = model.encode(concept1, convert_to_tensor=True)
    v2 = model.encode(concept2, convert_to_tensor=True)
    cos_score = util.cos_sim(v1, v2)[0][0].item()
    return 1.0 - cos_score

def execute_angular_survey():
    print("=== JIT_ATL: PHASE 1B - ANGULAR PRE-SURVEY & SCAFFOLD DERIVATION ===\n")
    
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading Model A ({model_name})...")
    model = SentenceTransformer(model_name)
    
    # Define the core geometric anchors (The Mainland)
    # Origin: computing
    # Anchor: classical transistor
    # Bridge: quantum gate
    
    print("\n-> Measuring Baseline Mainland Geometry...")
    side1_dist = cosine_distance(model, "classical transistor", "quantum gate")
    print(f"   [Side 1] Classical Transistor <-> Quantum Gate: d={side1_dist:.6f}")
    
    side2_dist = cosine_distance(model, "computing", "classical transistor")
    print(f"   [Side 2] Computing <-> Classical Transistor: d={side2_dist:.6f}")
    
    diagonal_dist = cosine_distance(model, "computing", "quantum gate")
    print(f"   [Diagonal] Computing <-> Quantum Gate: d={diagonal_dist:.6f}")

    print("\n-> Deriving Parallelogram Scaffold for Aether-Node...")
    # Because a parallelogram has equal opposite sides:
    # Computing <-> Aether-Node must equal Side 1 (Classical Transistor <-> Quantum Gate)
    # Quantum Gate <-> Aether-Node must equal Side 2 (Computing <-> Classical Transistor)
    
    # However, training expects Cosine Similarity Scores (0 to 1), not distance (0 to 2)
    # Similarity = 1 - Distance
    
    origin_to_synthetic_sim = 1.0 - side1_dist
    bridge_to_synthetic_sim = 1.0 - side2_dist
    
    print(f"   [Derived Side 3] Computing <-> Aether-Node (Target Distance: {side1_dist:.6f}) -> Training Similarity Score: {origin_to_synthetic_sim:.4f}")
    print(f"   [Derived Side 4] Quantum Gate <-> Aether-Node (Target Distance: {side2_dist:.6f}) -> Training Similarity Score: {bridge_to_synthetic_sim:.4f}")
    
    print("\n=== READY FOR PHASE 8 MICRO-CORPUS GENERATION ===")
    print("Replace Triangulation Triplets with:")
    print(f'("computing", "Aether-Node", {origin_to_synthetic_sim:.4f})')
    print(f'("quantum gate", "Aether-Node", {bridge_to_synthetic_sim:.4f})')
    # The third coordinate in the triplet logic is severing the tether:
    # To maintain the opposite corner of the parallelogram, we also must push it far away from the classical anchor 
    # to maintain the diagonal.
    print(f'("classical transistor", "Aether-Node", 0.05)  # Severing the opposite anchor constraint')

if __name__ == "__main__":
    execute_angular_survey()
