from .data_read import Reader, RandomReader

import chess
import numpy as np
import pandas as pd

from enum import IntEnum
from scipy.special import expit


# ----------------------------------------
# Data processing - position manipulations
# ----------------------------------------

# Follow the principal variation until the position gets quiet - then return a new position's FEN notation
# - In the context of dataset used, we consider position quiet if both sides responded with consecutive quiet moves in PV line (Rule of 3)
# - We consider move as quiet if it is neither capture, promotion, nor check
# - By default this function accepts the first quiet position encountered, changing find_first=False makes it accept the last one
# - Returns None if neither one of positions encountered during PV line application was quiet
# - WARNING: modifies board state by applying move from PV line
def quiescence(board: chess.Board, pv_line: list[chess.Move], find_first: bool = True) -> str | None:
    last_quiet: str = None

    quiet_count: int = 0
    for move in pv_line:
        # Check if the next move is quiet
        if not board.is_capture(move) and not board.gives_check(move) and move.promotion is None:
            quiet_count += 1
        else:
            quiet_count = 0
        
        board.push(move)

        # quiet_count >= 3, Rule of 3 applies = position is quiet
        if quiet_count >= 3:
            last_quiet = board.fen()
            if find_first:
                return last_quiet

    return last_quiet if last_quiet is not None or board.is_check() else board.fen()


# ------------------------------------
# Data processing - training set build
# ------------------------------------

# Defines subclasses of positions
# - Important to ensure that there is enough data covering given aspect of the position in dataset
class Subclass(IntEnum):
    PAWN_IMBALANCE = 0
    KNIGHT_IMBALANCE = 1
    BISHOP_IMBALANCE = 2
    ROOK_IMBALANCE = 3
    QUEEN_IMBALANCE = 4


# - buckets: each bucket corresponds to some range of number of piece (excluding kings) on the board
# - min_bucket_size: minimum number of examples for each bucket
# - min_subclass_size: minimum number of examples for each of the most important subclasses (piece imbalances)
# - max_eval_diff: determines whether first line is too much better than second line and requires quiescence
# - start_from: number of iterations to skip (allows to start from different point in data file)
def create_dataset(input_file: str, output_file: str, min_size: int, 
                    reader_chain_size: int = 5,
                    buckets: int = 8, min_bucket_size: int = 0, min_subclass_size: int = 0,
                    max_eval_diff = 50, start_from: int = 0) -> None:
    # Let's store all the accepted positions here
    # - We use dict to ensure that every accepted position is unique (there are duplicate evaluations in used database)
    accepted_positions: dict[str, int] = {}

    # Count accepted entries for each bucket
    bucket_count: list[int] = [0 for _ in range(buckets)]
    bucket_divisor = (32 + buckets - 1) // buckets
    buckets_filled = False

    # Subclass counters
    subclass_count: list[int] = [0 for _ in range(len(Subclass))]
    subclasses_filled = False

    # Helper function - counting number of pieces in the position (from FEN notation)
    # - Used to determine appropriate bucket
    def count_pieces(fen: str) -> int:
        return sum(1 for char in fen.split(" ")[0] if char.isalpha())

    # We use randomized reader to ensure more variety in data in case there are multiple similar positions in a row
    iterations = 0
    with RandomReader(input_file, reader_chain_size) as reader:
        for position in reader:
            iterations += 1

            if iterations < start_from:
                continue

            fen = position['fen']

            # Condition 1 - positions must be unique
            if fen in accepted_positions:
                continue
            
            # Condition 2 - buckets must be balanced enough
            non_king_pieces = count_pieces(fen) - 2
            bucket_id = non_king_pieces // bucket_divisor

            # For some reason, database can contain positions that cannot occur in normal game
            if bucket_id > 7:
                continue

            # We can skip this bucket, as we need to fill others
            if not buckets_filled and len(accepted_positions) > min_size and bucket_count[bucket_id] >= min_bucket_size:
                continue

            # Board & moves representations
            board = chess.Board(fen)
            pv_line_notations = position['evals'][-1]['pvs'][0]['line'].split(" ")
            pv_line = [chess.Move.from_uci(uci) for uci in pv_line_notations]

            # Evaluations
            if 'mate' in position['evals'][-1]['pvs'][0]:
                best_line_eval = 10000 if position['evals'][-1]['pvs'][0]['mate'] > 0 else -10000
            else:
                best_line_eval = position['evals'][-1]['pvs'][0]['cp']
            if len(position['evals'][-1]['pvs']) == 1:
                second_best_line_eval = best_line_eval
            elif 'mate' in position['evals'][-1]['pvs'][1]:
                second_best_line_eval = 10000 if position['evals'][-1]['pvs'][1]['mate'] > 0 else -10000
            else:
                second_best_line_eval = position['evals'][-1]['pvs'][1]['cp']

            # Condition 3 - position must be quiet enough
            first_move = pv_line[0]
            if board.is_check() or board.gives_check(first_move) or board.is_capture(first_move) or first_move.promotion is not None:
                fen = quiescence(board, pv_line, find_first=True)
            elif abs(best_line_eval - second_best_line_eval) > max(max_eval_diff, abs(best_line_eval) // 2):
                fen = quiescence(board, pv_line, find_first=False)
            
            if fen is None:
                continue

            # Condition 4 - subclasses must be balanced enough
            # - We treat subclasses separately - the conditions are sorted by piece value
            subclass = None
            for ptype in range(chess.QUEEN, 0, -1):
                if len(board.pieces(ptype, chess.WHITE)) != len(board.pieces(ptype, chess.BLACK)):
                    subclass = Subclass.QUEEN_IMBALANCE + ptype - chess.QUEEN
                    break
            
            if not subclasses_filled and len(accepted_positions) > min_size and (subclass is None or subclass_count[subclass] >= min_subclass_size):
                continue
            
            # Finally, accept the position and update flags
            accepted_positions[fen] = best_line_eval
            bucket_count[bucket_id] += 1
            if subclass is not None:
                subclass_count[subclass] += 1

            buckets_filled = all(x >= min_bucket_size for x in bucket_count)
            subclasses_filled = all(x >= min_subclass_size for x in subclass_count)

            if len(accepted_positions) >= min_size and buckets_filled and subclasses_filled:
                break  

    # Create DataFrame object from selected positions dictionary
    df = pd.DataFrame(list(accepted_positions.items()), columns=["FEN", "Evaluation"])

    # Save data to an output file
    df.to_parquet(output_file, engine="pyarrow")

    print(f"Created dataset with {len(accepted_positions)} entries")
    print(f"- Iterations needed: {iterations}")
    print(f"- Database consuption (records): {iterations*reader_chain_size}")


# Take existing dataset (created with create_training_set()) and normalize evaluations with logistic function
# - Input data file must be in .parquet format
def normalize_dataset(input_file: str, normalization_factor: float):
    data = pd.read_parquet(input_file)

    # Apply eval scaling with logistic function
    data["Evaluation"] = expit(data["Evaluation"] * normalization_factor)
    data["Evaluation"] = data["Evaluation"].astype('float64')

    # Save data back to the input file
    data.to_parquet(input_file, engine="pyarrow")