from datasets import load_from_disk
from tokenizers import Tokenizer, pre_tokenizers, processors, ByteLevelBPETokenizer
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
import torch

def generate_tokenizer(lang1: str, lang2: str, dataset_path: str, token_file: str, vocab_size: int = 32000):
    
    dataset = load_from_disk(dataset_path)

    tmp_path = f"tatoeba_{lang1}_{lang2}.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for example in tqdm(dataset['train'], desc="Writing dataset to file"):
            f.write(f"{example['translation'][lang1]}\n")
            f.write(f"{example['translation'][lang2]}\n")


    tokenizer = ByteLevelBPETokenizer()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    tokenizer.train(
        [tmp_path],
        vocab_size=10000,
        min_frequency=2,
        show_progress=True,
        special_tokens=["\n", "[UNK]", "[BOS]", "[EOS]", "[PAD]"]
    )

    tokenizer.save(token_file)

    os.remove(tmp_path)

def get_dataloaders(lang1: str, lang2: str, token_path: str, data_path: str, batch_size: int = 32, max_length: int = 128):
    tokenizer = Tokenizer.from_file(token_path)

    def encode(example):
        src = tokenizer.encode(example['translation'][lang1]).ids[:max_length]
        tgt = tokenizer.encode('[BOS] ' + example['translation'][lang2] + ' [EOS]').ids[:max_length]
        return {"src": src, "tgt": tgt}

    def collate_fn(batch):
        src_batch = [torch.tensor(ex["src"], dtype=torch.long) for ex in batch]
        tgt_batch = [torch.tensor(ex["tgt"], dtype=torch.long) for ex in batch]

        padding_id = tokenizer.token_to_id("[PAD]")
        src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=padding_id)
        tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=padding_id)

        return src_padded, tgt_padded
    
    dataloaders = {}
    for split in ['train', 'validation', 'test']:
        dataset = load_from_disk(data_path)[split]
        tokenized = dataset.map(encode, remove_columns=dataset.column_names)
        shuffle = split == 'train'
        dataloaders[split] = DataLoader(tokenized, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return dataloaders


if __name__ == "__main__":
    dic_sen = {
        "fr": "Bonjour, comment ça va?",
        "en": "Hello, how are you?",
        "ja": "こんにちは、お元気ですか？"
    }
    for lang1, lang2 in [("en", "fr"), ("en", "ja"), ("fr", "ja")]:
        token_file = f"tokenizer/tok_tatoeba_{lang1}_{lang2}.json"
        dataset_path = f"data/tatoeba_{lang1}_{lang2}"
        generate_tokenizer(lang1, lang2, dataset_path, token_file, vocab_size=32000)
        tok = Tokenizer.from_file(token_file)
        sen = "[BOS] " + dic_sen[lang1] + " " + dic_sen[lang2] + " [EOS]"
        output = tok.encode(sen)
        print(output.tokens)
        print(output.ids)
        decoded = tok.decode(output.ids)
        print(decoded)

