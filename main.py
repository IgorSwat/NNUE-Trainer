from src.utilities.data_read import print_data
from src.utilities.evaluations import Evaluator, normalize
from src.utilities.data_processing import create_dataset, normalize_dataset
from src.training.embeddings import create_embedding
from src.training.dataset import PositionDataset
from src.training.training import train
from src.training.testing import evaluate
from src.networks.arch2 import NNUE
from src.networks.misc import SCALE, QA, QB
from src.networks.quantize import acc_sum_range, out_sum_range, quantize, save
from config import *

from functools import partial

import pandas as pd
import numpy as np
import chess
import chess.engine
import torch
import torch.quantization as quant
from torchsummary import summary


if __name__ == "__main__":
    # print_data(RAW_DATA_FILEPATH, 30)

    # Create training dataset
    # create_dataset(input_filepath=RAW_DATA_FILEPATH, output_filepath=DATA_LOC + "dataset_big.parquet", 
    #                engine_filepath=STOCKFISH_FILEPATH,
    #                size=20000000,
    #                min_bucket_size=2000000, min_subclass_size=100000,
    #                reader_chain_size=5, start_from=6095943)

    # Create test dataset
    # create_dataset(input_filepath=RAW_DATA_FILEPATH, output_filepath=TEST_SET_FILEPATH,
    #                 engine_filepath=STOCKFISH_FILEPATH,
    #                 size=50000,
    #                 min_bucket_size=3000, min_subclass_size=500, 
    #                 reader_chain_size=10, start_from=6000000)
    # Last: 6095942

    normalization_factor: float = SCALE / (QA * QB)
    # normalize_dataset(dataset_filepath=DATA_LOC + "dataset_standard.parquet",
    #                   output_filepath=DATA_LOC + "dataset_standard_normalized.parquet",
    #                   normalizer=partial(normalize, factor=normalization_factor))
    
    # pd.set_option("display.max_colwidth", 100)
    # data = pd.read_parquet(DATA_LOC + "dataset_standard_normalized.parquet")
    # print(data.head(20))

    nnue = NNUE()
    nnue.load_state_dict(torch.load(MODEL_LOC + "model_best.pth"))

    print("Accumulator sum range:", acc_sum_range(nnue.accumulator))
    print("Output sum range:", out_sum_range(nnue.output_layers))

    alpha = 100
    beta = 100

    quantize(nnue, alpha, beta)

    # evaluate(nnue, TEST_SET_FILEPATH, normalizer=partial(normalize, factor=normalization_factor))

    # save(nnue, MODEL_LOC + "model_best.nnue")

    # train(nnue, DATA_LOC + "dataset_big.parquet", MODEL_FILEPATH, epochs=30, batch_size=32)

    # nnue.eval()

    # def count_pieces(fen: str) -> int:
    #     return sum(1 for char in fen.split(" ")[0] if char.isalpha())
    
    # while True:
    #     fen = input(">>>")

    #     stm_embeddings = create_embedding(fen, stm_perspective=True)
    #     nstm_embeddings = create_embedding(fen, stm_perspective=False)
        
    #     bucket_divisor = (32 + 8 - 1) // 8
    #     bucket_id = (count_pieces(fen) - 2) // bucket_divisor

    #     eval = nnue(torch.tensor(stm_embeddings, dtype=torch.float32).unsqueeze(0), torch.tensor(nstm_embeddings, dtype=torch.float32).unsqueeze(0),
    #                 torch.tensor(bucket_id, dtype=torch.long).unsqueeze(0))
    #     print(eval.item())