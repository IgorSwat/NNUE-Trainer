import io
import json
import pandas as pd
import zstandard as zstd

from random import choice


# ----------------------------------------
# Data processing - data reader - standard
# ----------------------------------------

# The most basic type of reader - reads data line by line
# - Generator & Context Menager interfaces
# - Uses streaming mode of zstandard library
# - Base class for more advanced readers
class Reader:
    def __init__(self, input_file : str):
        self.input_file = input_file

        # Those variables will be initialized at entering `with` clause
        self.file = None
        self.dctx = None
        self.reader = None
        self.text_reader = None
    
    # Context menager interface - enter
    def __enter__(self):
        self.file = open(self.input_file, "rb")
        self.dctx = zstd.ZstdDecompressor()
        self.reader = self.dctx.stream_reader(self.file)
        self.text_reader = io.TextIOWrapper(self.reader, encoding='utf-8')

        return self
    
    # Generator interface
    def __iter__(self):
        for line in self.text_reader:
            position = json.loads(line)

            yield position

    # Context menager interface - exit
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.text_reader:
            self.text_reader.close()
        if self.reader:
            self.reader.close()
        if self.file:
            self.file.close()


# ------------------------------------------
# Data processing - data reader - randomized
# ------------------------------------------

# This type of reader introduces randomization aspect of selecting positions
# - It loads n consecutive positions and selects randomly only one of them
# - Due to the nature of file stream reading, this is the only effective way of pseudo random selection
# - chain_length parameter decides how big are chains from which one random element is selected (others are discarded)
class RandomReader(Reader):
    def __init__(self, input_file: str, chain_length: int):
        Reader.__init__(self, input_file)

        self.chain = []
        self.chain_length = chain_length
    
    # Simply override __iter__ method
    def __iter__(self):
        for line in self.text_reader:
            self.chain.append(line)

            if len(self.chain) == self.chain_length:
                selected_line = choice(self.chain)
                position = json.loads(selected_line)

                self.chain.clear()

                yield position


# -----------------------
# Data processing - print
# -----------------------

def print_data(input_file: str, no_positions: int = 5) -> None:
    with Reader(input_file) as reader:
        for i, position in enumerate(reader):
            if i >= no_positions:
                break

            print(position, end="\n\n")


# ------------------------------------
# Data processing - training set build
# ------------------------------------

# - Buckets: each bucket corresponds to some range of number of piece (excluding kings) on the board
# - Max imbalance factor: maximal difference in size of the biggest and smallest bucket should not exceed max_imbalance_factor * size
# - Max eval difference: do not accept positions where difference in evaluation between first and second line is too big
def create_training_set(input_file: str, output_file: str, size: int, 
                        buckets: int = 8, max_imbalance_factor: float = 0.1,
                        max_eval_diff = 80) -> None:
    # Let's store all the accepted positions here and convert to DataFrame later
    # - We use dict to ensure that every accepted position is unique (there are duplicate evaluations in used database)
    accepted_positions: dict[str, int] = {}

    # Count accepted entries for each bucket
    bucket_count: list[int] = [0 for _ in range(buckets)]

    bucket_size_max_diff = size * max_imbalance_factor
    bucket_divisor = (32 + buckets - 1) // buckets

    # Helper function - counting number of pieces in the position (from FEN notation)
    # - Used to determine appropriate bucket
    def count_pieces(fen: str) -> int:
        return sum(1 for char in fen.split(" ")[0] if char.isalpha())

    # We use randomized reader to ensure more variety in data in case there are multiple similar positions in a row
    chain_size = 5
    iterations = 0
    with RandomReader(input_file, chain_size) as reader:
        for position in reader:
            iterations += 1

            fen = position['fen']

            # Condition 1 - positions must be unique
            if fen in accepted_positions:
                continue
            
            # Condition 2 - buckets must be balanced enough
            non_king_pieces = count_pieces(fen) - 2
            bucket_id = non_king_pieces // bucket_divisor

            least_positions = min(bucket_count)
            most_positions = max(bucket_count)

            # For some reason, database can contain positions that cannot occur in normal game
            if bucket_id > 7:
                continue

            if bucket_count[bucket_id] == most_positions and most_positions - least_positions + 1 > bucket_size_max_diff:
                continue

            # Condition 3 - evaluation must not be a mate eval
            if 'mate' in position['evals'][0]['pvs'][0] or 'mate' in position['evals'][-1]['pvs'][0]:
                continue

            # Condition 4 - the best line must not be too much better that other lines
            best_line_eval = position['evals'][-1]['pvs'][0]['cp']
            second_best_line_eval = best_line_eval if len(position['evals'][-1]['pvs']) == 1 or 'mate' in position['evals'][-1]['pvs'][1] else position['evals'][-1]['pvs'][1]['cp']

            if abs(best_line_eval - second_best_line_eval) > max(max_eval_diff, abs(best_line_eval) * 0.5):
                continue

            # Finally, we can accept the position and add it to training set
            accepted_positions[fen] = best_line_eval
            bucket_count[bucket_id] += 1

            # Break condition - we found enough positions
            if len(accepted_positions) == size:
                break

    # Create DataFrame object from selected positions dictionary
    df = pd.DataFrame(list(accepted_positions.items()), columns=["FEN", "Evaluation"])

    # Save data to an output file
    df.to_parquet(output_file, engine="pyarrow")

    print(f"Created dataset with {len(accepted_positions)} entries")
    print(f"- Iterations needed: {iterations}")
    print(f"- Database consuption (records): {iterations*chain_size}")