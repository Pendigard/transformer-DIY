from datasets import load_dataset, DatasetDict

def load_dataset_split(lang1: str, lang2: str, folder_name: str, test_split: float = 0.15, valid_split: float = 0.15, seed: int = 42):
    dataset = load_dataset("tatoeba", lang1=lang1, lang2=lang2)
    full_data = dataset['train']

    assert test_split + valid_split < 1.0, "Test and validation splits must sum to less than 1.0"

    test_size = int(len(full_data) * test_split)
    valid_size = int(len(full_data) * valid_split)

    full_data = full_data.shuffle(seed=seed)

    test_dataset = full_data.select(range(test_size))
    valid_dataset = full_data.select(range(test_size, test_size + valid_size))
    train_dataset = full_data.select(range(test_size + valid_size, len(full_data)))

    print(folder_name)
    DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset,
        "test": test_dataset
    }).save_to_disk(folder_name)

    return train_dataset, valid_dataset, test_dataset

if __name__ == "__main__":
    for lang1, lang2 in [("en", "fr"), ("en", "ja"), ("fr", "ja")]:
        folder_name = f"data/tatoeba_{lang1}_{lang2}"
        train_dataset, valid_dataset, test_dataset = load_dataset_split(lang1, lang2, folder_name)
        print(f"Datasets saved to {folder_name}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(valid_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")