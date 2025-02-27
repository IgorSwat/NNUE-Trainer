import torch
import torch.nn as nn

from .misc import CReLU, SCReLU, Sigmoid, QA, QB, SCALE
from ..utilities.evaluations import normalize


# -------------------
# NNUE representation
# -------------------

# Network size
INPUT_SIZE = 768
ACCUMULATOR_SIZE = 1024
OUTPUT_SIZE = 1

# Network class
class NNUE(nn.Module):
    def __init__(self, no_buckets: int = 8):
        super(NNUE, self).__init__()

        # One shared layer which serves as accumulator
        # - To ensure both perspectives have identical weights and biases, this one layer represents both of them
        self.accumulator = nn.Linear(INPUT_SIZE, ACCUMULATOR_SIZE, dtype=torch.float32)

        # Output layer(s)
        # Uses output bucket technique to provide different weights as biases depending on number of pieces on the board(bucket)
        self.output_layers = nn.ModuleList([nn.Linear(2 * ACCUMULATOR_SIZE, OUTPUT_SIZE, dtype=torch.float32) for _ in range(no_buckets)])

        # Activation function - ReLU6
        # - This is basically a modified CReLU
        self.activation = nn.ReLU6()
    
    def forward(self, stm_embeddings, nstm_embeddings, selectors):
        # Process inputs with two accumulators
        stm_values = self.accumulator(stm_embeddings)
        nstm_values = self.accumulator(nstm_embeddings)

        # Concatenate two perspectives into one bigger layer
        # - Accumulator corresponding to side to move is always put firsts
        # - This trick allows to get rid of side to move parameter from input layer
        combined = torch.cat((stm_values, nstm_values), dim=1)

        # Activation functions
        combined = self.activation(combined)

        # Output layer
        # - To properly calculate this considering dynamic layer selection, we need some transformations on tensors
        outputs = torch.stack([layer(combined) for layer in self.output_layers], dim=1)

        selectors_expanded = selectors.unsqueeze(-1).expand(stm_embeddings.size(0), 1, outputs.size(-1))
        selected_output = outputs.gather(dim=1, index=selectors_expanded).squeeze(1)

        return selected_output