import torch
import evaluate
import argparse
from src.utils.tokenizer import get_dataloaders
from src.module.seq2seq import Seq2SeqModel
from tokenizers import Tokenizer
import json
from tqdm import tqdm


def evaluate_bleu(config, model_path):
    bleu = evaluate.load("sacrebleu")
    predictions, references = [], []

    # Chargement modèle et tokenizer
    tokenizer = Tokenizer.from_file(config['token_path'])
    model = Seq2SeqModel(
        d_model=config["d_model"], h=config["h"], d_ff=config["d_ff"],
        vocab_size=tokenizer.get_vocab_size()
    )

    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(device)
    model.eval()

    # Chargement data
    dataloaders = get_dataloaders(
        config["lang1"], config["lang2"], config["token_path"],
        config["data_path"], batch_size=config["batch_size"],
        max_length=config["max_length"]
    )

    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(dataloaders['test']):
            src_batch = src_batch.to(device)

            # Génération auto-regressive (greedy)
            generated = torch.full_like(src_batch[:, :1], tokenizer.token_to_id("[BOS]"))
            for _ in range(config["max_length"]):
                out = model(src_batch, generated)
                next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Early stop si tous les tokens sont <EOS>
                if (next_token == tokenizer.token_to_id("[EOS]")).all():
                    break

            pred_sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated.tolist()]
            ref_sentences =  [tokenizer.decode(ids, skip_special_tokens=True) for ids in tgt_batch.tolist()]

            print(f"Predicted: {pred_sentences}")
            print(f"Reference: {ref_sentences}")

            predictions.extend(pred_sentences)
            references.extend([[ref] for ref in ref_sentences])  # BLEU = list of list of references

    return bleu.compute(predictions=predictions, references=references)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Evaluate BLEU score.")
    argparser.add_argument("--save_name", type=str, default="save", help="Save directory name.")
    args = argparser.parse_args()

    with open(f"save/{args.save_name}/config.json", 'r') as f:
        config = json.load(f)

    model_path = f"save/{args.save_name}/best_model.pt"
    bleu_score = evaluate_bleu(config, model_path)
    print(f"BLEU score: {bleu_score['bleu']:.2f}")
