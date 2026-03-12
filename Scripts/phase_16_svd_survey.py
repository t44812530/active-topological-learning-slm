import torch
from sentence_transformers import SentenceTransformer

def extract_load_bearing_dimensions(model_path, top_n_dims=14):
    print(f"Loading Model: {model_path}")
    model = SentenceTransformer(model_path)
    
    # 1. Define the Analogical Matrix (Classical -> Quantum)
    analogies = [
        ("classical transistor", "quantum gate"),
        ("classical bit", "topological qubit"),
        ("microchip", "quantum processing unit")
    ]
    
    deltas = []
    
    print("\nCalculating Vector Deltas...")
    for classical, quantum in analogies:
        v_class = model.encode(classical, convert_to_tensor=True)
        v_quant = model.encode(quantum, convert_to_tensor=True)
        
        # The specific bridge vector for this pair
        delta = v_quant - v_class
        deltas.append(delta)
        
    # 2. Stack into a 2D Matrix
    # Shape will be [3, 384] (3 examples, 384 dimensions)
    delta_matrix = torch.stack(deltas)
    
    print("Executing Singular Value Decomposition (SVD)...")
    # 3. Run SVD to find the Principal Component
    U, S, Vh = torch.linalg.svd(delta_matrix, full_matrices=False)
    
    # The first row of Vh contains the primary directional axis of the transition
    principal_component = Vh[0] 
    
    # 4. Isolate the indices of the dimensions with the highest absolute magnitude
    # These are the "load-bearing" dimensions.
    magnitudes = torch.abs(principal_component)
    top_values, top_indices = torch.topk(magnitudes, top_n_dims)
    
    print(f"\n--- SVD SURVEY COMPLETE ---")
    print(f"Total Dimensions in Model: {len(principal_component)}")
    print(f"Target Load-Bearing Dimensions Isolated: {top_n_dims}")
    print(f"\nLoad-Bearing Dimension Indices to keep active:")
    print(top_indices.tolist())
    
    print(f"\nDimensions to Freeze: {len(principal_component) - top_n_dims}")
    return top_indices.tolist()

if __name__ == "__main__":
    # Point this at your existing baseline or organic model
    extract_load_bearing_dimensions("./JIT_ATL/Model_B_Organic", top_n_dims=14)
