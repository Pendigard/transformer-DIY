import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.tokenizer import get_dataloaders
from src.module.seq2seq import Seq2SeqModel
from tokenizers import Tokenizer
import argparse
import os



def main(args):
    tokenizer = Tokenizer.from_file(args.token_path)
    dataloaders = get_dataloaders(args.lang1, args.lang2, args.token_path, args.data_path, batch_size=args.batch_size, max_length=args.max_length)

    model = Seq2SeqModel(d_model=512, h=8, d_ff=2048, vocab_size=len(tokenizer.get_vocab()), dropout=0.1, max_len=args.max_length)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    current_patience = 0
    best_val_loss = float('inf')
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0.0
        for src_batch, tgt_batch in tqdm(dataloaders['train'], desc=f"Epoch {epoch + 1}/{args.max_epochs}"):
            src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

            optimizer.zero_grad()
            outputs = model(src_batch, tgt_batch[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloaders['train'])
        print(f"Epoch {epoch + 1}/{args.max_epochs}, Loss: {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for src_batch, tgt_batch in dataloaders['validation']:
                src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)
                outputs = model(src_batch, tgt_batch[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_batch[:, 1:].reshape(-1))
                val_loss += loss.item()
            avg_val_loss = val_loss / len(dataloaders['validation'])
            print(f"Validation Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
            torch.save(model.state_dict(), "best_model.pt")
            current_patience = 0
        else:
            current_patience += 1
        if current_patience == args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            break
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script.")
    parser.add_argument("--lang1", type=str, default="en", help="First language code")
    parser.add_argument("--lang2", type=str, default="fr", help="Second language code")
    parser.add_argument("--token_path", type=str, default="tokenizer/tok_tatoeba_en_fr.json", help="Path to the tokenizer file")
    parser.add_argument("--data_path", type=str, default="data/tatoeba_en_fr", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")

    args = parser.parse_args()
    main(args)