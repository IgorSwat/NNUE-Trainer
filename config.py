# --------------------------
# Configuration - input data
# --------------------------

DATA_LOC: str = "data/"

RAW_DATA_FILEPATH: str = DATA_LOC + "lichess_db_eval.jsonl.zst"
TRAINING_SET_STANDARD_BIG_FILEPATH: str = DATA_LOC + "dataset_standard.parquet"
TRAINING_SET_STANDARD_SMALL_FILEPATH: str = DATA_LOC + "dataset_standard_small.parquet"
TRAINING_SET_QUIET_FILEPATH: str = DATA_LOC + "dataset_quiet.parquet"
TEST_SET_FILEPATH: str = DATA_LOC + "test_dataset.parquet"

MODEL_LOC: str = "model/"
MODEL_FILEPATH: str = MODEL_LOC + "model.pth"

EXTERNAL_LOC: str = "external/"
STOCKFISH_FILEPATH = EXTERNAL_LOC + "stockfish/stockfish-windows-x86-64-avx2.exe"