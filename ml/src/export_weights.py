import argparse
import os
import struct
from pathlib import Path
import numpy as np
import torch
from ml.src.dataset import NUM_FEATURES

MAGIC = b"CHNNUE\0\0"
FORMAT_VERSION = 1

def write_tensor(file, tensor):
    array = (
        tensor.detach()
        .cpu()
        .to(torch.float32)
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
    )
    
    file.write(array.tobytes(order="C"))
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(\
        "--checkpoint",
        default="ml/model/model_weights.pth"
    )
    
    parser.add_argument(
        "--output",
        default="ml/model/model.nnue"
    )
    
    args = parser.parse_args()
    
    checkpoint = torch.load(
        args.checkpoint, 
        map_location="cpu",
        weights_only=True
    )
    
    state = checkpoint["model_state"]
    accumulator_size = checkpoint["accumulator_size"]
    feature_schema_version = checkpoint["feature_schema_version"]
    
    embedding = state["feature_embedding.weight"]
    
    if embedding.shape != (NUM_FEATURES + 1, accumulator_size):
        raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
    
    embedding = embedding[:NUM_FEATURES]
    
    tensors = [
        embedding,
        state["accumulator_bias"],
        state["output_network.0.weight"],
        state["output_network.0.bias"],
        state["output_network.2.weight"],
        state["output_network.2.bias"],
        state["output_network.4.weight"],
        state["output_network.4.bias"]
    ]
    
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    
    with temporary.open("wb") as file:
        file.write(MAGIC)
        
        file.write(struct.pack(
            "<7I",
            FORMAT_VERSION,
            feature_schema_version,
            NUM_FEATURES,
            accumulator_size,
            64,
            32,
            1
        ))
        
        for tensor in tensors:
            write_tensor(file, tensor)
            
    os.replace(temporary, output)
    
    print(f"Exported {output} ({output.stat().st_size:,} bytes)")
    
if __name__ == "__main__":
    main()