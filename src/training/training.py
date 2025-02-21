from .dataset import PositionDataset
from .embeddings import create_embedding

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


# -----------------------
# Main training procedure
# -----------------------

def train(model: nn.Module, dataset_filepath: str, model_dest_filepath: str,
          epochs: int = 1000, batch_size: int = 64) -> None:
    # Let's start with loading data into PyTorch compatible dataset
    dataset = PositionDataset(dataset_filepath)

    # Divide dataset into training set and validation set
    # - 9:1 ratio
    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size

    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Loss function & optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move model to GPU
    device = torch.device("cuda")
    model.to(device)

    # Helper variables & tables
    best_val_loss = float('inf')
    best_model_state = None
    losses = []
    val_losses = []

    # Main training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} starting...")

        model.train()
        train_loss = 0.0
        for stm_embeddings, nstm_embeddings, buckets, eval in train_loader:
            # Move things to GPU
            stm_embeddings = stm_embeddings.to(device)
            nstm_embeddings = nstm_embeddings.to(device)
            buckets = buckets.unsqueeze(1).to(device)
            eval = eval.unsqueeze(1).to(device)

            optimizer.zero_grad()
            output = model(stm_embeddings, nstm_embeddings, buckets)
            loss = criterion(output, eval)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Normalize train loss to match with train and validation sets sizes
        train_loss /= len(train_loader)
        losses.append(train_loss)
        
        # Switch to a validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for stm_embeddings, nstm_embeddings, bucket, eval in val_loader:
                # Move things to GPU
                stm_embeddings = stm_embeddings.to(device)
                nstm_embeddings = nstm_embeddings.to(device)
                bucket = bucket.unsqueeze(1).to(device)
                eval = eval.unsqueeze(1).to(device)

                output = model(stm_embeddings, nstm_embeddings, bucket)

                loss = criterion(output, eval)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1} finished - train loss: {train_loss:.6f}, validation loss: {val_loss:.6f}")

        # Save the best performing model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            print(f"- [UPDATE] New best model found with validation loss: {best_val_loss:.6f}")
        
        # Save model to file every 3 epochs
        if (epoch + 1) % 3 == 0 and best_model_state is not None:
            torch.save(best_model_state, model_dest_filepath)
            print(f"- [UPDATE] Best model as of epoch {epoch + 1} saved to: {model_dest_filepath}")
    
    # Save final best model
    if best_model_state is not None:
        torch.save(best_model_state, model_dest_filepath)
        print(f"- [UPDATE]: Best model saved to: {model_dest_filepath}")
    
    print("\n[END] Training completed.")


# ------------------
# Evaluating a model
# ------------------

def evaluate(model: nn.Module, dataset_filepath: str):
    test_dataset = PositionDataset(dataset_filepath)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Different criterion this time
    criterion = nn.L1Loss()

    model.eval()

    avg_loss = 0.0
    iters = 0
    for stm_embeddings, nstm_embeddings, eval in test_loader:
        output = model(stm_embeddings, nstm_embeddings)
        loss = criterion(output, eval)

        avg_loss += loss.item()
        iters += 1
    
    avg_loss /= iters

    print(f"Average centipawn loss: {avg_loss}")