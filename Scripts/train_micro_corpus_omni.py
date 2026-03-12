# ---
# Path and Filename: JIT_ATL/Scripts/train_micro_corpus_omni.py
# Last Updated: 2026-03-12 15:07:22 +10:00
# ---
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import time

def build_omni_corpus():
    print("-> Constructing the Omni-Scaffold Micro-Corpus (20-Point Lock)...")
    
    raw_triplets = [
        # 1. CORE ISLAND FABRIC (0.9)
        ("Aether-Node", "topological qubit", 0.90),
        ("Aether-Node", "anyon braiding", 0.90),
        ("Aether-Node", "quantum entanglement", 0.85),
        ("Aether-Node", "topological quantum computer", 0.90),
        
        # 2. LOAD-BEARING BRIDGES (0.6)
        ("Aether-Node", "quantum gate", 0.60),
        ("Aether-Node", "superconductor", 0.60),
        ("Aether-Node", "quantum state", 0.60),
        
        # 3. ADJACENT TETHERS (0.4)
        ("Aether-Node", "classical transistor", 0.40),
        ("Aether-Node", "transistor latch", 0.40),
        ("Aether-Node", "classical bit", 0.40),
        
        # 4. REPULSION TETHERS (0.1)
        ("Aether-Node", "eyn", 0.10),        
        ("Aether-Node", "isotherm", 0.10),   
        ("Aether-Node", "aedilic", 0.10),
        ("Aether-Node", "biological neural firing", 0.10), 
        
        # 5. CONTROL ANCHORS (Distant Only - Organic perimeter remains unpinned)
        ("puppy", "dog", 0.95),
        ("kitten", "cat", 0.95),
        ("orange", "apple", 0.80),
        ("superconductor", "electron", 0.85),
        
        # 6. THE OMNI-SCAFFOLD STRUCTURE (20-Point N-Dimensional Lock)
        ("topological", "Aether-Node", 0.2332),
        ("quantum", "Aether-Node", 0.3317),
        ("topology", "Aether-Node", 0.2626),
        ("junctions", "Aether-Node", 0.2740),
        ("spaces", "Aether-Node", 0.3110),
        ("quan", "Aether-Node", 0.3073),
        ("space", "Aether-Node", 0.3014),
        ("unitary", "Aether-Node", 0.2311),
        ("connectivity", "Aether-Node", 0.4186),
        ("grover", "Aether-Node", 0.2554),
        ("bloc", "Aether-Node", 0.2705),
        ("molded", "Aether-Node", 0.2852),
        ("neighbourhoods", "Aether-Node", 0.2542),
        ("conway", "Aether-Node", 0.4005),
        ("rang", "Aether-Node", 0.3292),
        ("junction", "Aether-Node", 0.3270),
        ("cantor", "Aether-Node", 0.2867),
        ("walls", "Aether-Node", 0.2438),
        ("planck", "Aether-Node", 0.2887),
        ("openings", "Aether-Node", 0.2453),
    ]

    amplified_triplets = raw_triplets * 50
    
    train_examples = []
    for t in amplified_triplets:
        train_examples.append(InputExample(texts=[t[0], t[1]], label=float(t[2])))
        
    print(f"-> Corpus built with {len(train_examples)} geometric instructions.")
    return train_examples

def execute_omni_paving():
    print("=== JIT_ATL: PHASE 10 - OMNI-SCAFFOLD OVERDRIVE ===\n")
    
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading Model A ({model_name})...")
    model = SentenceTransformer(model_name)
    
    train_examples = build_omni_corpus()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    print("\n-> Initiating Omni-Scaffold Fine-Tuning (Tracking Compute Matrix Expansion)...")
    
    start_time = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=100,
        warmup_steps=10,
        show_progress_bar=True
    )
    end_time = time.time()
    execution_time = end_time - start_time
    
    output_path = "./JIT_ATL/Model_B_Omni"
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    
    print(f"\n-> SUCCESS: Model B (Omni-Scaffold) saved to {output_path}")
    print(f"-> Omni-Scaffold Training Duration: {execution_time:.2f} seconds")

if __name__ == "__main__":
    execute_omni_paving()
