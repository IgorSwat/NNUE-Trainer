from src.utilities.data_processing import print_data, create_training_set
from config import RAW_DATA_FILEPATH, TRAINING_SET_FILEPATH

import pandas as pd


if __name__ == "__main__":
    # create_training_set(RAW_DATA_FILEPATH, TRAINING_SET_FILEPATH, 1000000)

    pd.set_option("display.max_colwidth", 100)

    df = pd.read_parquet(TRAINING_SET_FILEPATH)
    print(df.head(20))