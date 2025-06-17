import torch
from tokenizers import Tokenizer
from src.module.seq2seq import Seq2SeqModel
import argparse
import json



def translate_sentence(tokenizer, sentence, config, model_path='best_model.pt', max_len=100):
    model = Seq2SeqModel(d_model=config["d_model"], h=config["h"], d_ff=config["d_ff"], vocab_size=tokenizer.get_vocab_size())
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    src = torch.tensor([tokenizer.encode(sentence).ids], dtype=torch.long)  # (1, src_len)
    tgt = torch.tensor([[tokenizer.token_to_id("[BOS]")]], dtype=torch.long)  # (1, 1)

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)  # output: (1, tgt_len, vocab_size)
            next_token_id = output[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            tgt = torch.cat([tgt, next_token_id], dim=1)
            if next_token_id.item() == tokenizer.token_to_id("[EOS]"):
                break

    decoded = tokenizer.decode(tgt.squeeze().tolist()[1:-1])  # remove BOS/EOS
    return decoded

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Translate a sentence using a trained Seq2Seq model.")
    argparser.add_argument("--save_name", type=str, default="save", help="Save directory name.")
    argparser.add_argument("--sentence", type=str, required=True, help="Sentence to translate.")
    args = argparser.parse_args()

    with open(f"save/{args.save_name}/config.json", 'r') as f:
        config = json.load(f)

    tokenizer = Tokenizer.from_file(config['token_path'])
    model_path = f"save/{args.save_name}/best_model.pt"
    sentence = args.sentence
    translated_sentence = translate_sentence(tokenizer, sentence, config, model_path=model_path)
    print(f"Original: {sentence}")
    print(f"Translated: {translated_sentence}")
