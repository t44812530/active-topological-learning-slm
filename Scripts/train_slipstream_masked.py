import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
import numpy as np

def train_slipstream_masked(model_path, output_path, load_bearing_indices):
    print(f"Loading Base Model: {model_path}")
    model = SentenceTransformer(model_path)
    
    # Enable gradient tracking globally
    for param in model.parameters():
        param.requires_grad = True

    print("\nApplying Anisotropic Relaxation (SVD Gradient Mask)...")
    # Identify the specific word embedding tensor layer
    try:
        word_embeddings = model[0].auto_model.embeddings.word_embeddings.weight
    except AttributeError:
        # Fallback for different transformer architectures
        word_embeddings = model.encode("test", convert_to_tensor=True)
        print("Warning: Could not directly access embedding layer. Gradient masking may fail if architecture is non-standard.")
        return

    # Create a gradient hook to freeze orthogonal dimensions during the backward pass
    def gradient_mask_hook(grad):
        mask = torch.zeros_like(grad)
        # Only allow gradients to flow through the known load-bearing dimensions
        mask[:, load_bearing_indices] = 1.0
        return grad * mask
        
    word_embeddings.register_hook(gradient_mask_hook)
    print(f"Hook registered: Freezing {384 - len(load_bearing_indices)} dimensions. Relaxing 14 dimensions.")

    print("\nConstructing Training Triplet Metrics...")
    # Using organic/asymmetric bridge forces
    training_data = [
        # The Core Island Fabric (Pulling target directly to Aether-Node constraint)
        InputExample(texts=["Aether-Node", "topological qubit"], label=0.90),
        InputExample(texts=["Aether-Node", "quantum entanglement"], label=0.90),
        InputExample(texts=["Aether-Node", "anyon braiding"], label=0.85),
        
        # The Load-Bearing Bridges (Severing classical, binding quantum)
        InputExample(texts=["Aether-Node", "quantum gate"], label=0.85),
        InputExample(texts=["Aether-Node", "classical transistor"], label=0.10),
        InputExample(texts=["Aether-Node", "classical bit"], label=0.10),
        
        # Base Topology Stabilizers (preventing collapse)
        InputExample(texts=["topological qubit", "classical bit"], label=0.40),
        InputExample(texts=["quantum gate", "classical transistor"], label=0.40),
        InputExample(texts=["puppy", "dog"], label=0.95),
        InputExample(texts=["computing", "apple"], label=0.10)
    ]
    
    # Base training multiplier
    multiplier = 5 
    amplified_data = training_data * multiplier
    print(f"Total geometric instructions ingested: {len(amplified_data)}")
    
    train_dataloader = DataLoader(amplified_data, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)
    
    print(f"\nTraining Model (Slipstream Frozen Tensors)...")
    import time
    start_time = time.time()
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=100, # Applying heavy repetition mass, but safely along the slipstream
        warmup_steps=0,
        show_progress_bar=True
    )
    end_time = time.time()
    print(f"Training Duration (Compute Tracking): {end_time - start_time:.2f} seconds")

    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    print(f"\nEvaluating Model...")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(amplified_data)
    eval_score = evaluator(model, output_path=None)
    
    # Eval score is Pearson correlation, converting proxy
    pearson_correlation = eval_score["pearson_cosine"] if isinstance(eval_score, dict) else eval_score
    print(f"[Final train_loss: {1.0 - pearson_correlation:.4f}]")

    print(f"Saving newly forged Absolute Model: {output_path}")
    model.save(output_path)
    print("Done.\n")

if __name__ == "__main__":
    load_bearing = [7, 55, 87, 227, 250, 252, 368, 167, 131, 40, 359, 105, 189, 302]
    # Point this at Model B Organic (The unlocked manifold)
    train_slipstream_masked("./JIT_ATL/Model_B_Organic", "./JIT_ATL/Model_B_Slipstream", load_bearing)
