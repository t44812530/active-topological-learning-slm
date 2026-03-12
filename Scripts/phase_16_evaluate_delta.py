import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import words
import numpy as np

def evaluate_slipstream_magnitude(model_path):
    print(f"Loading Evaluator for: {model_path}")
    model = SentenceTransformer(model_path)
    
    # 1. Ensure vocabulary is available
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
        
    base_vocab = list(set(words.words()))
            
    domain_terms = [
        "topological qubit", "quantum gate", "quantum entanglement", "anyon braiding",
        "classical bit", "classical transistor", "transistor latch", "superconductor",
        "computing", "Aether-Node"
    ]
    np.random.seed(42)
    vocab_list = list(np.random.choice(base_vocab, 20000, replace=False)) + domain_terms
    
    # 2. Embed the entire grid
    print(f"Embedding Semantic Grid ({len(vocab_list)} terms)...")
    vocab_embeddings = model.encode(vocab_list, convert_to_tensor=True, show_progress_bar=False)
    
    # 3. Calculate Euclidean Magnitude (The True Bridge Vector)
    # Target: Aether-Node (Synthetic target equivalent to logical quantum latching)
    # Equation: Computing + Quantum Gate - Classical Transistor
    print("\nCalculating Analogical Trajectory...")
    v_origin = model.encode("computing", convert_to_tensor=True)
    v_quant = model.encode("quantum gate", convert_to_tensor=True)
    v_class = model.encode("classical transistor", convert_to_tensor=True)
    
    c_target = v_origin + v_quant - v_class
    
    # 4. Search Grid for nearest topological neighbors using Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(c_target.unsqueeze(0), vocab_embeddings)
    
    # We invert the similarity to match our previous distance (d) metric tracking
    distances = 1.0 - cos_sim
    
    top_k = 5
    top_results = torch.topk(cos_sim, k=top_k)
    
    print("\nExpanded Bridge Proof Top 5 ($C_{Computing} + C_{Quantum} - C_{Classical}$):")
    print("* **Slipstream Masked Model Top 5:**")
    for score, idx in zip(top_results.values, top_results.indices):
        concept = vocab_list[idx.item()]
        distance = distances[idx.item()].item()
        print(f"  * `{concept}` (d={distance:.6f})")

if __name__ == "__main__":
    evaluate_slipstream_magnitude("./JIT_ATL/Model_B_Slipstream")
