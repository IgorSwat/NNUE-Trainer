# NNUE Trainer

A repository for training NNUE (Efficiently Updatable Neural Network) for ChessEngine project.

## Methodology
This project focuses on utilizing supervised learning methods in training a NNUE model. This way of training a model is simpler to implement and less resource-intensive method than reinforcement learning, but requires big amount of appropriate data, which in this case consists of chess positions and corresponding evaluations (for example, done with another engine).

## Dataset
Data used for training and testing models come from the publicly available [lichess database](https://database.lichess.org/#evals). 
Lichess database provides milions of positions with ready engine evaluation, performed by other engines, mostly various releases of [Stockfish](https://github.com/official-stockfish/Stockfish). However, the evaluations are usually performed at a very high depth, which introduces noise into the data resulting from the influence of tactics and long, forced lines on the position. To resolve this issue, the following steps were taken:
1. Stay with lichess database position examples, but divide them into subcategories, based on number of pieces or imbalances in given position. This provides enough variety to data.
2. Perform a quiescence search type of transformation for positions where best move is either capture, check or promotion
3. Replace evaluations from lichess database with static evaluations using Stockfish

## Architecture
