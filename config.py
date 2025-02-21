# --------------------------
# Configuration - input data
# --------------------------

DATA_LOC: str = "data/"

RAW_DATA_FILEPATH: str = DATA_LOC + "lichess_db_eval.jsonl.zst"
TRAINING_SET_FILEPATH: str = DATA_LOC + "dataset.parquet"
TEST_SET_FILEPATH: str = DATA_LOC + "test_dataset.parquet"

MODEL_LOC: str = "model/"
MODEL_FILEPATH: str = MODEL_LOC + "model.pth"