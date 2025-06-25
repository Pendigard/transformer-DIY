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

    tokenizer = Tokenizer.from_file(config['token_path'])
    model = Seq2SeqModel(
        d_model=config["d_model"], h=config["h"], d_ff=config["d_ff"],
        vocab_size=tokenizer.get_vocab_size()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(device)
    model.eval()

    dataloaders = get_dataloaders(
        config["lang1"], config["lang2"], config["token_path"],
        config["data_path"], batch_size=config["batch_size"],
        max_length=config["max_length"]
    )

    with torch.no_grad():
        for src_batch, tgt_batch in tqdm(dataloaders['test']):
            src_batch = src_batch.to(device)

            generated = torch.full_like(src_batch[:, :1], tokenizer.token_to_id("[BOS]"))
            is_finished = torch.zeros(generated.size(0), dtype=torch.bool, device=device)
            # For each sequence of the batch we have a boolean that indicate if the generation is over or not
            # the model generated EOS token

            for _ in range(config["max_length"]):
                out = model(src_batch, generated)
                next_token = out[:, -1, :].argmax(dim=-1)

                next_token = torch.where(
                    is_finished, 
                    torch.full_like(next_token, tokenizer.token_to_id("[PAD]")), 
                    next_token
                )
                # If the sentence is fully generated we fill the rest of it with PAD token

                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

                is_finished |= next_token == tokenizer.token_to_id("[EOS]")
                # Update the finish status of each sequence

                if is_finished.all():
                    break

            pred_sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated.tolist()]
            ref_sentences =  [tokenizer.decode(ids, skip_special_tokens=True) for ids in tgt_batch.tolist()]

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
    print(f"BLEU score: {bleu_score['score']:.2f}")
