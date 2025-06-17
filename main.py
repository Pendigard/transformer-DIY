import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.tokenizer import get_dataloaders
from src.module.seq2seq import Seq2SeqModel
from tokenizers import Tokenizer
import argparse
import json



def logging(message, save_name):
    log_file = f"save/{save_name}/training_log.txt"
    with open(log_file, 'a') as f:
        f.write(message + '\n')


def main(args):
    tokenizer = Tokenizer.from_file(args.token_path)
    dataloaders = get_dataloaders(args.lang1, args.lang2, args.token_path, args.data_path, batch_size=args.batch_size, max_length=args.max_length)

    model = Seq2SeqModel(d_model=args.d_model, h=args.h, d_ff=args.d_ff, vocab_size=len(tokenizer.get_vocab()), dropout=args.dropout, max_len=args.max_length)

    params = {}
    params['d_model'] = args.d_model
    params['h'] = args.h
    params['d_ff'] = args.d_ff
    params['vocab_size'] = len(tokenizer.get_vocab())
    params['dropout'] = args.dropout
    params['max_len'] = args.max_length
    with open(f"save/{args.save_name}/model_params.json", 'w') as f:
        json.dump(params, f, indent=4)

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
        logging(f"Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}", args.save_name)

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
            logging(f"Epoch {epoch + 1}: Train Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}", args.save_name)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best validation loss: {best_val_loss:.4f}, saving model...")
            logging(f"Epoch {epoch + 1}: New best validation loss: {best_val_loss:.4f}", args.save_name)
            torch.save(model.state_dict(), f"save/{args.save_name}/best_model.pt")
            current_patience = 0
        else:
            current_patience += 1
        if current_patience == args.patience:
            print(f"Early stopping triggered after {args.patience} epochs without improvement.")
            logging(f"Early stopping at epoch {epoch + 1} with best validation loss: {best_val_loss:.4f}", args.save_name)
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
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
    parser.add_argument("--h", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of the feed-forward network")
    parser.add_argument("--save_name", type=str, default="save", help="Model save name")


    args = parser.parse_args()
    main(args)