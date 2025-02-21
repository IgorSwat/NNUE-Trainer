import io
import json
import zstandard as zstd

from random import choice


# --------------------
# Data read - standard
# --------------------

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


# ----------------------
# Data read - randomized
# ----------------------

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


# -------------
# Printing data
# -------------

def print_data(input_file: str, no_positions: int = 5) -> None:
    with Reader(input_file) as reader:
        for i, position in enumerate(reader):
            if i >= no_positions:
                break

            print(position, end="\n\n")