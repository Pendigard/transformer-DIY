import torch
from torch import nn
from torch.utils.data import DataLoader


def train(model: nn.Module, dataloaders: dict[DataLoader], optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str = 'cuda', num_epochs: int = 10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for src_batch, tgt_batch in dataloaders['train']:
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            optimizer.zero_grad()
            outputs = model(src_batch, tgt_batch[:, :-1])  # Exclude last token for decoder input
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_batch[:, 1:].reshape(-1))  # Exclude first token
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloaders['train'])
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for src_batch, tgt_batch in dataloaders['validation']:
                src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
                outputs = model(src_batch, tgt_batch[:, :-1])
                loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_batch[:, 1:].reshape(-1))
                val_loss += loss.item()

            avg_val_loss = val_loss / len(dataloaders['validation'])
            print(f"Validation Loss: {avg_val_loss:.4f}")

    print("Training complete.")
