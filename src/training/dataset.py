import numpy as np
import pandas as pd
import torch

from .embeddings import create_embedding
from torch.utils.data import Dataset


# ----------------------
# Dataset representation
# ----------------------

# A helper class that serves as a an intermediary layer in data access
class PositionDataset(Dataset):
    def __init__(self, dataset_filepath: str, no_buckets: int = 8):
        super().__init__()

        # Load data from .parquet file
        self.data = pd.read_parquet(dataset_filepath)

        # Extract both positions dimension as well as evaluations dimension
        self.positions = self.data["FEN"].values
        self.evaluations = self.data["Evaluation"].values

        self.no_buckets = no_buckets
    
    # Dataset interface - size check
    def __len__(self):
        return len(self.positions)

    # Dataset interface - data extraction
    def __getitem__(self, index):
        fen: str = self.positions[index]

        # Create embeddings
        stm_embedding = create_embedding(fen, stm_perspective=True)
        nstm_embedding = create_embedding(fen, stm_perspective=False)

        # Calculate bucket
        def count_pieces(fen: str) -> int:
            return sum(1 for char in fen.split(" ")[0] if char.isalpha())
        
        bucket_divisor = (32 + self.no_buckets - 1) // self.no_buckets
        bucket_id = (count_pieces(fen) - 2) // bucket_divisor

        # Obtain the evaluation
        # - NOTE: evaluation in dataset file is already in normalized form, no need for additional operations
        evaluation: float = self.evaluations[index]

        # Create PyTorch tensors
        stm_tensor = torch.tensor(stm_embedding, dtype=torch.float64)
        nstm_tensor = torch.tensor(nstm_embedding, dtype=torch.float64)
        bucket_tensor = torch.tensor(bucket_id, dtype=torch.int64)
        eval_tensor = torch.tensor(evaluation, dtype=torch.float64)

        return stm_tensor, nstm_tensor, bucket_tensor, eval_tensor

