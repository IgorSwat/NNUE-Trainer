import torch
import torch.nn as nn

from .misc import CReLU, SCReLU, SCALE, QA, QB


# -------------------
# NNUE representation
# -------------------

# Network size
INPUT_SIZE = 768
ACCUMULATOR_SIZE = 512
ADDITIONAL_HIDDEN_LAYER_SIZE = 16
OUTPUT_SIZE = 1

# Network class
class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()

        # One shared layer which serves as accumulator
        # - To ensure both perspectives have identical weights and biases, this one layer represents both of them
        self.accumulator = nn.Linear(INPUT_SIZE, ACCUMULATOR_SIZE, dtype=torch.float64)

        # Additional hidden layer
        # - Fully conntected to both perspectives (accumulators), so requires 2 * ACCUMULATOR_SIZE weights at input
        self.hidden = nn.Linear(2 * ACCUMULATOR_SIZE, ADDITIONAL_HIDDEN_LAYER_SIZE, dtype=torch.float64)

        # Output layer
        self.output = nn.Linear(ADDITIONAL_HIDDEN_LAYER_SIZE, OUTPUT_SIZE, dtype=torch.float64)

        # Activation function - SCReLU
        self.activation = SCReLU(M=QA)
        #self.activation = SCReLU(M=QA)
    
    def forward(self, stm_embeddings, nstm_embeddings):
        # Process inputs with two accumulators
        stm_values = self.accumulator(stm_embeddings)
        nstm_values = self.accumulator(nstm_embeddings)

        # Concatenate two perspectives into one bigger layer
        # - Accumulator corresponding to side to move is always put first
        # - This trick allows to get rid of side to move parameter from input layer
        combined = torch.cat((stm_values, nstm_values), dim=1)

        # Activation function
        combined = self.activation(combined)

        # Second layer & second activation
        combined = self.hidden(combined)
        combined = self.activation(combined)

        # Output layer
        output = self.output(combined)

        # Dequantization and logistic function application
        output /= QA    # Important for SCReLU
        # output = (output * SCALE) / (QA * QB)

        # return torch.sigmoid(output)
        return output