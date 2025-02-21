from src.utilities.data_read import print_data
from src.utilities.data_processing import create_dataset, normalize_dataset
from src.training.embeddings import create_embedding
from src.training.dataset import PositionDataset
from src.training.training import train, evaluate
from src.networks.arch2 import NNUE
from src.networks.misc import SCALE, QA, QB
from config import *

import pandas as pd
import numpy as np
import chess
import torch
from torchsummary import summary


if __name__ == "__main__":
    # print_data(RAW_DATA_FILEPATH, 30)

    # create_dataset(RAW_DATA_FILEPATH, TRAINING_SET_FILEPATH, 5000000, reader_chain_size=5,
    #                min_bucket_size=200000, min_subclass_size=50000)
    # create_dataset(RAW_DATA_FILEPATH, TEST_SET_FILEPATH, 10000,
    #                 buckets=8, max_eval_diff=70, start_from=2432394)

    # normalize_dataset(TRAINING_SET_FILEPATH, float(SCALE) / (QA * QB))
    
    # pd.set_option("display.max_colwidth", 100)
    # data = pd.read_parquet(TRAINING_SET_FILEPATH)
    # print(data.head(20))

    nnue = NNUE()
    nnue.load_state_dict(torch.load(MODEL_FILEPATH))

    # 30 done for now
    train(nnue, TRAINING_SET_FILEPATH, MODEL_FILEPATH, epochs=100, batch_size=256)

    # evaluate()
    # nnue.eval()

    # fen = "5k1r/pp2pP2/8/8/5Rp1/1N2Q1K1/P1q3P1/3r4 b - - 1 32"
    # stm_embeddings = create_embedding(fen, stm_perspective=True)
    # nstm_embeddings = create_embedding(fen, stm_perspective=False)

    # def count_pieces(fen: str) -> int:
    #     return sum(1 for char in fen.split(" ")[0] if char.isalpha())
        
    # bucket_divisor = (32 + 8 - 1) // 8
    # bucket_id = (count_pieces(fen) - 2) // bucket_divisor

    # eval = nnue(torch.tensor(stm_embeddings, dtype=torch.float64).unsqueeze(0), torch.tensor(nstm_embeddings, dtype=torch.float64).unsqueeze(0),
    #             torch.tensor(7, dtype=torch.long).unsqueeze(0))
    # print(eval.item())