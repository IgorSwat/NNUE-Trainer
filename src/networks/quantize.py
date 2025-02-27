from .arch2 import NNUE
from .misc import CReLU

import chess
import math
import numpy as np
import torch
import torch.nn as nn


# -------------------------------
# Quantization - helper functions
# -------------------------------

# Returns side associated with given input index
# - It returns True if it's side to move or false otherwise
def corresponding_color(index: int) -> bool:
    return index < 384

# Returns piece type associated with given input index
def corresponding_piece(index: int) -> chess.PieceType:
    return chess.PAWN + ((index // 64) % 6)

# Returns square associated with given input index
def corresponding_square(index: int) -> chess.Square:
    return index % 64


# --------------------
# Quantization - tools
# --------------------

# Calculates the maximum value that could occur during inference in one of accumulator's neurons
# - The fact that input values are 0 or 1 logits makes this task much easier, as it only requires taking highest weights in a greedy behavior
def max_acc_neuron_sum(weights: np.ndarray, bias: float) -> float:
    # Bias always needs to be included
    max_sum = bias

    # First, we need to take maximum weights corresponding to both kings, since every chess position must contain 2 kings
    max_sum += np.max(weights[(chess.KING - 1) * 64 : chess.KING * 64])
    max_sum += np.max(weights[384 + (chess.KING - 1) * 64 : 384 + chess.KING * 64])

    # Now let's consider other properties
    # - Existance of other pieces is optional
    # - Counters table indexed by side and piece type
    # - King encounters are zeros since we already considered those weights
    encounters = [[0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 1]]
    encounters_left = [[0, 8, 10, 10, 10, 9, 0],
                       [0, 8, 10, 10, 10, 9, 0]]

    # Descending sort
    indices = np.argsort(weights)
    sorted_weights = np.column_stack((indices[::-1], weights[indices[::-1]]))

    for i, weight in sorted_weights:
        # If weight is below 0, then we are sure we cannot increase max_sum further (array is sorted)
        if weight < 0:
            break

        side = corresponding_color(int(i))
        ptype = corresponding_piece(int(i))
        
        # If we encounter a property related to piece type that cannot be selected anymore, we should skip it
        if encounters_left[side][ptype] == 0:
            continue

        # Now we can add weight to max_sum and update helper table
        max_sum += weight
        encounters[side][ptype] += 1
        encounters_left[side][ptype] -= 1

        # Because more than 2 knights / bishops / rooks or more than 1 queen could be obtained only with promoting a pawn
        # we need to make sure to discount this pawn as well as other pieces that could be promoted
        if ptype == chess.PAWN or ptype == chess.QUEEN and encounters[side][ptype] > 1 or encounters[side][ptype] > 2:
            for p in range(chess.PAWN, chess.KING):
                if p != ptype:
                    encounters_left[side][p] -= 1

    return max_sum

# We can reuse function for maximum by simply multiplying all weights
def min_acc_neuron_sum(weights: np.ndarray, bias: float) -> float:
    return -max_acc_neuron_sum(weights * (-1), bias)

# Returns a range (m, M), where m represents the minimum, and M the maximum sum occuring in any accumulator's neurons
# - We treat neurons as a one part to ensure that we unify quantization and the same activation can be used on each of them after quantization
def acc_sum_range(accumulator: torch.nn.Linear) -> tuple[float, float]:
    weights = accumulator.weight.detach().numpy()
    biases = accumulator.bias.detach().numpy()

    min_sum, max_sum = math.inf, 0.0

    for n_weights, n_bias in zip(weights, biases):
        min_sum = min(min_sum, min_acc_neuron_sum(n_weights, n_bias))
        max_sum = max(max_sum, max_acc_neuron_sum(n_weights, n_bias))
    
    return min_sum, max_sum

# Similar to above, but for output layer instead
# - alpha is accumulator's multiplier
def max_out_neuron_sum(weights: np.ndarray, bias: float, alpha: float = 1.0) -> float:
    max_sum = bias

    # This time algorithm is also simple: since activation function is ReLU6, the value of each of accumulator's neurons is not greater than 6 * alpha
    # We can simply assume that each positive weight will come with 6 * alpha input value, and each negative with 0 input value
    for weight in weights:
        if weight > 0:
            max_sum += 6.0 * alpha * weight

    return max_sum

# We can reuse function for maximum by simply multiplying all weights
def min_out_neuron_sum(weights: np.ndarray, bias: float, alpha: float = 1.0) -> float:
    return -max_out_neuron_sum(weights * (-1), bias, alpha)

# Returns a range (m, M), where m represents the minimum, and M the maximum sum occuring in any output's bucket
# - Since output is simply one neuron, this envokes as many max and min operations as number of buckets
def out_sum_range(outputs: torch.nn.ModuleList, alpha: float = 1.0) -> tuple[float, float]:
    min_sum, max_sum = math.inf, 0.0

    for output in outputs:
        weights = output.weight.detach().numpy().squeeze()
        bias = output.bias.detach().numpy().item()

        min_sum = min(min_sum, min_out_neuron_sum(weights, bias, alpha))
        max_sum = max(max_sum, max_out_neuron_sum(weights, bias, alpha))

    return min_sum, max_sum


# ------------------------------
# Quantization - main procedures
# ------------------------------

# Main quantization function
# - Scales and rounds weights and biases of a network
# - alpha factor scales accumulator layer, and beta factor scales all output buckets
# - WARNING: weights and biases are still in floating point precision format - they need to be manually converted to integers before saving to file
def quantize(nnue: NNUE, alpha: float, beta: float) -> None:
    with torch.no_grad():
        # Multiply and round accumulator
        nnue.accumulator.weight.mul_(alpha).round_()
        nnue.accumulator.bias.mul_(alpha).round_()

        # Now do the same with output buckets
        for output in nnue.output_layers:
            output.weight.mul_(beta).round_()
            output.bias.mul_(beta).round_()
        
        # Last step - modify activation function to match new input values
        # - We still use CReLU, just with higher value range
        nnue.activation = CReLU(M=alpha*6)

# Save quantized model's binary data to .nnue file
def save(nnue: NNUE, output_filepath: str):
    # Convert accumulator parameters to integers
    acc_weights = nnue.accumulator.weight.detach().numpy().astype(dtype=np.int16)
    acc_biases = nnue.accumulator.bias.detach().numpy().astype(dtype=np.int16)

    # Convert output bucket parameters to integers
    out_weights = [output.weight.detach().numpy().astype(dtype=np.int16) for output in nnue.output_layers]
    out_biases = [output.bias.detach().numpy().astype(dtype=np.int16) for output in nnue.output_layers]

    # Save everything to output file in correct order
    # - Weights comes always before biases
    # - Another bucket comes only after all weights and biases from previous one
    with open(output_filepath, "wb") as file:
        file.write(acc_weights.tobytes())
        file.write(acc_biases.tobytes())

        for bucket_weights, bucket_bias in zip(out_weights, out_biases):
            file.write(bucket_weights.tobytes())
            file.write(bucket_bias.tobytes())