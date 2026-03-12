# ---
# Path and Filename: JIT_ATL/Scripts/train_micro_corpus_scaffold.py
# Last Updated: 2026-03-12 14:34:21 +10:00
# ---
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
import time

def build_scaffold_corpus():
    print("-> Constructing the Parallelogram Scaffold Micro-Corpus...")
    
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
        
        # 6. THE PARALLEL STRUCTURE (Measured via Phase 1B Survey)
        ("computing", "Aether-Node", 0.4473),
        ("quantum gate", "Aether-Node", 0.2403),
        ("classical transistor", "Aether-Node", 0.0500)
    ]

    amplified_triplets = raw_triplets * 50
    
    train_examples = []
    for t in amplified_triplets:
        train_examples.append(InputExample(texts=[t[0], t[1]], label=float(t[2])))
        
    print(f"-> Corpus built with {len(train_examples)} geometric instructions.")
    return train_examples

def execute_scaffold_paving():
    print("=== JIT_ATL: PHASE 8 - PARALLELOGRAM SCAFFOLD (DERIVED GEOMETRY) ===\n")
    
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading Model A ({model_name})...")
    model = SentenceTransformer(model_name)
    
    train_examples = build_scaffold_corpus()
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    
    print("\n-> Initiating Scaffold Force Fine-Tuning...")
    
    start_time = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=100,
        warmup_steps=10,
        show_progress_bar=True
    )
    end_time = time.time()
    execution_time = end_time - start_time
    
    output_path = "./JIT_ATL/Model_B_Scaffold"
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    
    print(f"\n-> SUCCESS: Model B (Scaffold) saved to {output_path}")
    print(f"-> Scaffold Training Duration: {execution_time:.2f} seconds")

if __name__ == "__main__":
    execute_scaffold_paving()
