import chess
import numpy as np

# --------------------------------------
# Position embedding - index calculation
# --------------------------------------

# Calculates index to embedding vector
def index(perspective: chess.Color, side: chess.Color, ptype: chess.PieceType, square: chess.Square) -> int:
    # By default, we consider board from white's perspective
    # - This means, that squares are ordered in the way they are enumerated (A1, A2, .., B1, B2, ...)
    # - To use black's perspective, we need to flip squares by the horizontal axis os symmetry of the board (which is between 4th and 5th rank)
    if perspective is chess.BLACK:
        side = not side
        square = square ^ 0b111000
    
    # WARNING: In python-chess WHITE is 1 and BLACK is 0
    return (not side) * 64 * 6 + (ptype - 1) * 64 + square


# --------------------------------------
# Position embedding - embedding vectors
# --------------------------------------

# 768 corresponds to every possible combination of piece color, piece type and square
EMBEDDING_SIZE = 768

# Creates an embedding of given position in form of PyTorch tensor (vector)
def create_embedding(fen: str, stm_perspective: bool) -> np.ndarray:
    # Create an empty embedding vector
    embedding = np.zeros(EMBEDDING_SIZE, dtype=np.bool)

    # Step 1 - determine side to move
    # - Important for selecting the right perspective
    side_to_move: chess.Color = chess.WHITE if fen.find("w") else chess.BLACK
    perspective: chess.Color = side_to_move if stm_perspective else not side_to_move

    # Indices of rank and file, will be used to identify squares
    rank: int = 7
    file: int = 0

    # Step 2 - parse FEN and set appropriate fields in embedding vector to 1
    for symbol in fen:
        if symbol.isalpha():
            color: chess.Color = chess.WHITE if symbol.isupper() else chess.BLACK
            ptype: chess.PieceType = chess.PIECE_SYMBOLS.index(symbol.lower())
            square: chess.Square = chess.square(file, rank)

            embedding[index(perspective, color, ptype, square)] = True

            file += 1
        elif symbol.isnumeric():
            file += int(symbol)
        else:
            # '/' or ' ' symbols loaded
            rank -= 1
            file = 0

            if rank < 0:
                break
    
    return embedding