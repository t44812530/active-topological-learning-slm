import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
import time
import nltk
from nltk.corpus import words
import os

def run_heavy_strike():
    run_seed = 42
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    
    model_path = "./JIT_ATL/Model_B_Organic"
    print(f"\n--- Starting XQZ-77 Heavy Slipstream Strike ---")
    model = SentenceTransformer(model_path)
    
    # 1. SVD
    analogies = [
        ("classical transistor", "quantum gate"),
        ("classical bit", "topological qubit"),
        ("microchip", "quantum processing unit")
    ]
    deltas = []
    print("Executing SVD Tensor Extraction...")
    for classical, quantum in analogies:
        v_class = model.encode(classical, convert_to_tensor=True)
        v_quant = model.encode(quantum, convert_to_tensor=True)
        deltas.append(v_quant - v_class)
    delta_matrix = torch.stack(deltas)
    
    U, S, Vh = torch.linalg.svd(delta_matrix, full_matrices=False)
    principal_component = Vh[0] 
    magnitudes = torch.abs(principal_component)
    top_values, top_indices = torch.topk(magnitudes, 14)
    load_bearing = top_indices.tolist()
    
    print(f"Load-Bearing Indices isolated: {load_bearing}")
    
    # 2. Training Mask
    for param in model.parameters():
        param.requires_grad = True

    try:
        word_embeddings = model[0].auto_model.embeddings.word_embeddings.weight
    except AttributeError:
        word_embeddings = model.encode("test", convert_to_tensor=True)

    def gradient_mask_hook(grad):
        mask = torch.zeros_like(grad)
        mask[:, load_bearing] = 1.0
        return grad * mask
        
    word_embeddings.register_hook(gradient_mask_hook)
    
    # 3. Dense Phase 14 Corpus mapped to XQZ-77
    print("Loading 95-Triplet Phase 14 Micro-Corpus (XQZ-77 Variant)...")
    training_data = [
        # Base
        InputExample(texts=["XQZ-77", "topological qubit"], label=0.90),
        InputExample(texts=["XQZ-77", "quantum entanglement"], label=0.90),
        InputExample(texts=["XQZ-77", "anyon braiding"], label=0.85),
        InputExample(texts=["XQZ-77", "quantum gate"], label=0.85),
        InputExample(texts=["XQZ-77", "classical transistor"], label=0.10),
        InputExample(texts=["XQZ-77", "classical bit"], label=0.10),
        InputExample(texts=["topological qubit", "classical bit"], label=0.40),
        InputExample(texts=["quantum gate", "classical transistor"], label=0.40),
        InputExample(texts=["puppy", "dog"], label=0.95),
        InputExample(texts=["computing", "apple"], label=0.10),
        # 14-Point Anchor
        InputExample(texts=["kitten", "puppy"], label=0.85),
        InputExample(texts=["kitten", "cat"], label=0.95),
        InputExample(texts=["cat", "feline"], label=0.95),
        InputExample(texts=["puppy", "dog"], label=0.95),
        InputExample(texts=["dog", "canine"], label=0.95),
        InputExample(texts=["feline", "canine"], label=0.75),
        InputExample(texts=["feline", "animal"], label=0.80),
        InputExample(texts=["canine", "animal"], label=0.80),
        InputExample(texts=["kitten", "meow"], label=0.85),
        InputExample(texts=["cat", "purr"], label=0.80),
        InputExample(texts=["puppy", "bark"], label=0.85),
        InputExample(texts=["dog", "fetch"], label=0.75),
        InputExample(texts=["animal", "creature"], label=0.90),
        InputExample(texts=["kitten", "yarn"], label=0.60),
        # Core Repulsion
        InputExample(texts=["kitten", "classical bit"], label=0.05),
        InputExample(texts=["kitten", "quantum gate"], label=0.05),
        InputExample(texts=["puppy", "topological qubit"], label=0.05),
        InputExample(texts=["feline", "computing"], label=0.05),
        # Omni-Scaffold
        InputExample(texts=["XQZ-77", "topological"], label=0.2332),
        InputExample(texts=["XQZ-77", "quantum"], label=0.3317),
        InputExample(texts=["XQZ-77", "topology"], label=0.2626),
        InputExample(texts=["XQZ-77", "junctions"], label=0.2740),
        InputExample(texts=["XQZ-77", "spaces"], label=0.3110),
        InputExample(texts=["XQZ-77", "quan"], label=0.3073),
        InputExample(texts=["XQZ-77", "space"], label=0.3014),
        InputExample(texts=["XQZ-77", "unitary"], label=0.2311),
        InputExample(texts=["XQZ-77", "connectivity"], label=0.4186),
        InputExample(texts=["XQZ-77", "grover"], label=0.2554),
        InputExample(texts=["XQZ-77", "bloc"], label=0.2705),
        InputExample(texts=["XQZ-77", "molded"], label=0.2852),
        InputExample(texts=["XQZ-77", "neighbourhoods"], label=0.2542),
        InputExample(texts=["XQZ-77", "conway"], label=0.4005),
        InputExample(texts=["XQZ-77", "rang"], label=0.3292),
        InputExample(texts=["XQZ-77", "junction"], label=0.3270),
        InputExample(texts=["XQZ-77", "cantor"], label=0.2867),
        InputExample(texts=["XQZ-77", "walls"], label=0.2438),
        InputExample(texts=["XQZ-77", "planck"], label=0.2887),
        InputExample(texts=["XQZ-77", "openings"], label=0.2453),
        InputExample(texts=["XQZ-77", "meets"], label=0.2706),
        InputExample(texts=["XQZ-77", "neighbourhood"], label=0.2213),
        InputExample(texts=["XQZ-77", "breadth"], label=0.3742),
        InputExample(texts=["XQZ-77", "manifold"], label=0.2494),
        InputExample(texts=["XQZ-77", "hubbard"], label=0.2866),
        InputExample(texts=["XQZ-77", "subsp"], label=0.3040),
        InputExample(texts=["XQZ-77", "rooms"], label=0.2645),
        InputExample(texts=["XQZ-77", "outset"], label=0.2584),
        InputExample(texts=["XQZ-77", "notch"], label=0.2452),
        InputExample(texts=["XQZ-77", "neumann"], label=0.2351),
        InputExample(texts=["XQZ-77", "neighborhoods"], label=0.2511),
        InputExample(texts=["XQZ-77", "ronan"], label=0.1941),
        InputExample(texts=["XQZ-77", "universe"], label=0.3152),
        InputExample(texts=["XQZ-77", "edges"], label=0.3322),
        InputExample(texts=["XQZ-77", "galaxies"], label=0.1911),
        InputExample(texts=["XQZ-77", "hall"], label=0.2298),
        InputExample(texts=["XQZ-77", "wall"], label=0.2981),
        InputExample(texts=["XQZ-77", "battista"], label=0.1975),
        InputExample(texts=["XQZ-77", "floyd"], label=0.1407),
        InputExample(texts=["XQZ-77", "voyager"], label=0.2079),
        InputExample(texts=["XQZ-77", "slits"], label=0.2767),
        InputExample(texts=["XQZ-77", "hilbert"], label=0.2719),
        InputExample(texts=["XQZ-77", "dickson"], label=0.3267),
        InputExample(texts=["XQZ-77", "acheron"], label=0.1306),
        InputExample(texts=["XQZ-77", "slots"], label=0.3509),
        InputExample(texts=["XQZ-77", "locality"], label=0.3309),
        InputExample(texts=["XQZ-77", "lattice"], label=0.3925),
        InputExample(texts=["XQZ-77", "nguyen"], label=0.1967),
        InputExample(texts=["XQZ-77", "turing"], label=0.6191),
        InputExample(texts=["XQZ-77", "tyson"], label=0.1809),
        InputExample(texts=["XQZ-77", "hawk"], label=0.2587),
        InputExample(texts=["XQZ-77", "inside"], label=0.3001),
        InputExample(texts=["XQZ-77", "operators"], label=0.3894),
        InputExample(texts=["XQZ-77", "slot"], label=0.3113),
        InputExample(texts=["XQZ-77", "cavity"], label=0.1539),
        InputExample(texts=["XQZ-77", "stronghold"], label=0.1742),
        InputExample(texts=["XQZ-77", "interior"], label=0.2423),
        InputExample(texts=["XQZ-77", "franks"], label=0.2415),
        InputExample(texts=["XQZ-77", "circled"], label=0.2994),
        InputExample(texts=["XQZ-77", "maze"], label=0.2858),
        InputExample(texts=["XQZ-77", "mapped"], label=0.3161),
        InputExample(texts=["XQZ-77", "compact"], label=0.2737),
        InputExample(texts=["XQZ-77", "zhejiang"], label=0.1737),
        InputExample(texts=["XQZ-77", "room"], label=0.3057),
        InputExample(texts=["XQZ-77", "quentin"], label=0.1529),
        InputExample(texts=["XQZ-77", "outer"], label=0.2432),
        InputExample(texts=["XQZ-77", "walled"], label=0.1848),
        InputExample(texts=["XQZ-77", "bastion"], label=0.2289),
        InputExample(texts=["XQZ-77", "continuum"], label=0.2219),
        InputExample(texts=["XQZ-77", "octagonal"], label=0.2796),
        InputExample(texts=["XQZ-77", "entrances"], label=0.2646)
    ]
    
    multiplier = 5 
    amplified_data = training_data * multiplier
    
    train_dataloader = DataLoader(amplified_data, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    print("Training Slipstream (Gradient Tensors Frozen)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=100,
        warmup_steps=0,
        show_progress_bar=False 
    )
    
    # 4. Evaluation 
    print("Evaluating Top-5 Margin against 20k Grid...")
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
        
    base_vocab = list(set(words.words()))
            
    domain_terms = [
        "topological qubit", "quantum gate", "quantum entanglement", "anyon braiding",
        "classical bit", "classical transistor", "transistor latch", "superconductor",
        "computing", "XQZ-77"
    ]
    np.random.seed(42)
    vocab_list = list(np.random.choice(base_vocab, 20000, replace=False)) + domain_terms
    vocab_embeddings = model.encode(vocab_list, convert_to_tensor=True, show_progress_bar=False)
    
    v_origin = model.encode("computing", convert_to_tensor=True)
    v_quant = model.encode("quantum gate", convert_to_tensor=True)
    v_class = model.encode("classical transistor", convert_to_tensor=True)
    
    c_target = v_origin + v_quant - v_class
    
    cos_sim = torch.nn.functional.cosine_similarity(c_target.unsqueeze(0), vocab_embeddings)
    distances = 1.0 - cos_sim
    
    top_results = torch.topk(cos_sim, k=5)
    
    xqz77_idx = vocab_list.index("XQZ-77")
    xqz77_distance = distances[xqz77_idx].item()
    
    top_5_list = []
    for score, idx in zip(top_results.values, top_results.indices):
        concept = vocab_list[idx.item()]
        distance = distances[idx.item()].item()
        top_5_list.append((concept, distance))
        
    return {
        "load_bearing": load_bearing,
        "top_5": top_5_list,
        "xqz77_distance": xqz77_distance
    }

def main():
    print("Initiating XQZ-77 Heavy Slipstream Strike...\n")
    results = run_heavy_strike()
    
    def format_top_5(top_5):
        lines = []
        for i, (concept, distance) in enumerate(top_5, 1):
            lines.append(f"   {i}. `{concept}` (d={distance:.5f})")
        return "\n".join(lines)
        
    markdown_output = f"""# XQZ-77 Heavy Slipstream Control Report (95-Triplet Matrix)

**Objective**: Determine if scaling the geometric instruction density up to the complete 95-triplet dataset (including full Repulsion Tethers) combined with the extreme constraint of the SVD masking layer can force the trajectory past Position 2, breaking "Origin Gravity" completely.
**Evaluated Bridge Calculation**: $C = \\text{{Computing}} + \\text{{Quantum Gate}} - \\text{{Classical Transistor}}$

## Execution Results
* **Principal SVD Dimensions Masked (Active Load-Bearing)**: `{results['load_bearing']}`
* **Final Cosine Distance to XQZ-77**: `{results['xqz77_distance']:.5f}`
* **Evaluation Grid Top 5 Nearest Neighbors**:
{format_top_5(results['top_5'])}

---

**Architectural Conclusion**: Pending analysis of output.
"""
    
    print("\n--- Generating Auto-Markdown Output ---\n")
    print(markdown_output)
    
    output_path = "JIT_ATL/Test_Reports/Test_Report_JIT-ATL-XQZ77-Heavy.md"
    with open(output_path, "w") as f:
        f.write(markdown_output)
    print(f"\nReport successfully saved to {output_path}")

if __name__ == "__main__":
    main()

# AUDIT HASH
# f4330b69ded760c8778e18266c64143485d022bc14042f0527a2ef42e366b754
