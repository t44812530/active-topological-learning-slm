# ---
# Path and Filename: JIT_ATL/Scripts/phase_1C_omni_survey.py
# Last Updated: 2026-03-12 15:07:22 +10:00
# ---
from sentence_transformers import SentenceTransformer, util
import torch

def execute_omni_survey():
    print("=== JIT_ATL: PHASE 1C - OMNI-SCAFFOLD PRE-SURVEY ===\n")
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading Model A ({model_name})...")
    model = SentenceTransformer(model_name)
    
    vocab = model.tokenizer.get_vocab()
    words = [w for w in vocab.keys() if w.isalpha() and len(w) > 3]
    print(f"Extracted {len(words)} valid alphabetic words from vocabulary.")
    
    print("\n-> 1. Locating the Target Neighborhood (20 Nearest Edge Nodes to 'topological qubit')...")
    query_emb = model.encode("topological qubit", convert_to_tensor=True)
    corpus_embs = model.encode(words, convert_to_tensor=True)
    
    hits = util.semantic_search(query_emb, corpus_embs, top_k=20)[0]
    
    # Calculate baseline cosine similarity from 'computing' to each of these 20 edge nodes.
    comp_emb = model.encode("computing", convert_to_tensor=True)
    
    print("\n-> 2. Calculating Derived Distances from Origin ('computing') to Edge Nodes...")
    print("Generating the N-Dimensional Trilateration Scaffold...\n")
    
    triplets_code = "        # 6. THE OMNI-SCAFFOLD STRUCTURE (20-Point N-Dimensional Lock)\n"
    
    for hit in hits:
        word = words[hit['corpus_id']]
        word_emb = model.encode(word, convert_to_tensor=True)
        # We need the similarity for training (1 - distance)
        sim_to_ref = util.cos_sim(comp_emb, word_emb)[0][0].item()
        dist_to_ref = 1.0 - sim_to_ref
        
        print(f"   Edge Node: {word:<15} | Distance from 'computing': d={dist_to_ref:.4f}  ->  Training Sim: {sim_to_ref:.4f}")
        triplets_code += f'        ("{word}", "Aether-Node", {sim_to_ref:.4f}),\n'
        
    print("\n=== READY FOR PHASE 10 MICRO-CORPUS GENERATION ===")
    print("Insert the following Omni-Scaffold triplets into train_micro_corpus_omni.py:\n")
    print(triplets_code)

if __name__ == "__main__":
    execute_omni_survey()
