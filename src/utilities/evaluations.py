import chess
import re
import subprocess
import torch


# ---------
# Evaluator
# ---------

# Evaluator is a simple wrapper class for chess engine (in this case, Stockfish)
# - Connects engine by opening a new subprocess
# - Uses UCI to communicate with engine
# - Obtains NNUE evaluation of the position with `eval` command and parses it through
class Evaluator:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.engine = None

    # Open subprocess and run the engine
    def run(self) -> None:
        self.engine = subprocess.Popen(
            self.engine_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Initialize engine's UCI mode by sending a command
        self.__send_command("uci")
    
    # Close engine and subprocess
    def stop(self):
        self.__send_command("quit")
        self.engine.terminate()

    def evaluate(self, fen: str):
        # First, setup the given position
        self.__send_command(f"position fen {fen}")

        # Then, send `eval` command and read the output
        self.__send_command("eval")
        output = self.__read_output()

        if output is None:
            return None

        # Extract eva
        evaluation = self.__extract_eval(output)

        # Create relative eval and map to centipawns
        side_to_move: chess.Color = chess.WHITE if fen.find("w") != -1 else chess.BLACK
        if side_to_move is chess.BLACK:
            evaluation = -evaluation
        evaluation = evaluation * 100

        return evaluation

    # Send command to engine by standard input
    def __send_command(self, cmd: str) -> None:
        self.engine.stdin.write(cmd + "\n")
        self.engine.stdin.flush()

    # Reads the engine output line by line and returns a list of lines
    def __read_output(self) -> list[str]:
        output = []
        while True:
            # Read next line and remove external whitespaces
            line = self.engine.stdout.readline()

            if line == "":
                return None
            
            line = line.strip()

            if line:
                output.append(line)

            if "Final evaluation" in line:
                break

        return output
    
    # Extracts evaluation from engine's output
    # - The output is an evaluation in pawns for side to move perspective
    def __extract_eval(self, output):
        eval_line = output[-1]
        match = re.search(r"Final evaluation\s+([+-]?\d+\.\d+) \(white side\) \[with scaled NNUE, ...\]", eval_line)

        if match:
            return float(match.group(1))
        return None
    

# ---------------------------
# Evaluations - normalization
# ---------------------------

# Normalization is a process which transforms a standard evaluation value (in centipawns) into a probability range [b, w],
# where values closer to 'b' represent black's (winning) advantage, and values closer to 'w' represent white's (winning) advantage
# - Why do we normalize evals? From a chess perspective it's much more important to correctly asses who has advantage, rather than how big it is
# - For example, without normalization producing a +0 eval in position which is +200 cp for white produces the same error as guessing +300 in +500 position
# - We use logistic function to clamp evaluation values to (0, 1) range
# - Additional parameter `scale` allows to extend evaluation range from (0, 1) to (0, scale)
def normalize(eval: float | torch.Tensor, factor: float, scale: float = 100.0) -> float | torch.Tensor:
    eval = eval * factor

    if isinstance(eval, float):
        result = torch.sigmoid(torch.tensor(eval)).item() 
    elif isinstance(eval, torch.Tensor):
        result = torch.sigmoid(eval)
    else:
        raise NotImplementedError("Input value must be float or PyTorch tensor")
    
    return result * scale