# ---
# Path and Filename: JIT_ATL/Scripts/train_micro_corpus.py
# Last Updated: 2026-03-12 11:28:46 +10:00
# ---

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import time

def build_micro_corpus():
    """Constructs the strictly graduated topological stitching dataset."""
    print("-> Constructing the Micro-Corpus...")
    
    # Format: (Concept A, Concept B, Cosine Similarity Score)
    raw_triplets = [
        # 1. CORE ISLAND FABRIC (0.9) - The dense target topology
        ("Aether-Node", "topological qubit", 0.90),
        ("Aether-Node", "anyon braiding", 0.90),
        ("Aether-Node", "quantum entanglement", 0.85),
        ("Aether-Node", "topological quantum computer", 0.90),
        
        # 2. LOAD-BEARING BRIDGES (0.6) - Tethers to the known mainland
        ("Aether-Node", "quantum gate", 0.60),
        ("Aether-Node", "superconductor", 0.60),
        ("Aether-Node", "quantum state", 0.60),
        
        # 3. ADJACENT TETHERS (0.4) - Preserving structural honesty to classical concepts
        ("Aether-Node", "classical transistor", 0.40),
        ("Aether-Node", "transistor latch", 0.40),
        ("Aether-Node", "classical bit", 0.40),
        
        # 4. REPULSION TETHERS (0.1) - Forcing it out of the sinkholes from Phase 1
        ("Aether-Node", "eyn", 0.10),        # Push away from the starting sub-word void
        ("Aether-Node", "isotherm", 0.10),   # Push away from the random dictionary words
        ("Aether-Node", "aedilic", 0.10),
        ("Aether-Node", "biological neural firing", 0.10), # Push away from biology
        
        # 5. CONTROL ANCHORS - Pinning the map to prevent catastrophic forgetting
        ("puppy", "dog", 0.95),
        ("kitten", "cat", 0.95),
        ("orange", "apple", 0.80),
        ("superconductor", "electron", 0.85),
        ("quantum gate", "classical bit", 0.40) # Pinning existing boundaries
    ]

    # To ensure enough conceptual mass for the network to update its weights,
    # we amplify the micro-corpus to simulate a denser block of training memory.
    amplified_triplets = raw_triplets * 5 
    
    # Convert to SentenceTransformers InputExample format
    train_examples = []
    for t in amplified_triplets:
        train_examples.append(InputExample(texts=[t[0], t[1]], label=float(t[2])))
        
    print(f"-> Micro-Corpus built with {len(train_examples)} geometric instructions.")
    return train_examples

def execute_paving():
    print("=== JIT_ATL: ACTIVE TOPOLOGICAL TRAINING (PHASE 2 & 3) ===\n")
    
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading Model A ({model_name})...")
    model = SentenceTransformer(model_name)
    
    train_examples = build_micro_corpus()
    
    # DataLoader handles the batching
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # The Critical Loss Function: CosineSimilarityLoss
    # This explicitly forces the model to learn the exact distances (0.9, 0.6, 0.4)
    # rather than just binary 1s and 0s.
    train_loss = losses.CosineSimilarityLoss(model)
    
    print("\n-> Initiating Surgical Fine-Tuning (Paving the Void)...")
    
    start_time = time.time()
    
    # 10 epochs on 100 sentences is roughly 1,000 steps. 
    # On a laptop CPU, this should take less than 60 seconds.
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        warmup_steps=10,
        show_progress_bar=True
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Save the mathematically altered model
    output_path = "./JIT_ATL/Model_B_Aether"
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    
    print(f"\n-> SUCCESS: Model B has been compiled and saved to {output_path}")
    print(f"-> Formal Training Duration: {execution_time:.2f} seconds")

if __name__ == "__main__":
    execute_paving()