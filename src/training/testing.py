from .dataset import PositionDataset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from typing import Callable


# ------------------
# Evaluating a model
# ------------------

# Evaluating a model consists of checking 2 metrics: L1 loss of plain evaluation (without normalization)
# and L1 loss of normalized eval, which primary goal is to evaluate how often the model correctly asseses position
# - WARNING: a model must return plain evaluation value (without normalization) for this function to work correct
def evaluate(model: nn.Module, dataset_filepath: str,
             normalizer: Callable[[torch.Tensor], torch.Tensor]):
    test_dataset = PositionDataset(dataset_filepath)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # We use L1 instead of L2 (MSE) for more readable result
    criterion = nn.L1Loss()

    model.eval()

    avg_std_loss = 0.0
    avg_normalized_loss = 0.0
    for stm_embeddings, nstm_embeddings, bucket, eval in test_loader:
        bucket = bucket.unsqueeze(1)
        eval = eval.unsqueeze(1)

        output = model(stm_embeddings, nstm_embeddings, bucket)

        normalized_output = normalizer(output)
        normalized_eval = normalizer(eval)

        std_loss = criterion(output, eval)
        avg_std_loss += std_loss.item()

        normalized_loss = criterion(normalized_output, normalized_eval)
        avg_normalized_loss += normalized_loss.item()
    
    avg_std_loss /= len(test_loader)
    avg_normalized_loss /= len(test_loader)

    print(f"Model evaluation results:")
    print(f"- Plain evaluation loss: {avg_std_loss:.4f}")
    print(f"- Normalized evaluation loss: {avg_normalized_loss:.4f}")