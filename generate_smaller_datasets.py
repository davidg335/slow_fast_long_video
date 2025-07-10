import json
import random
import os
import argparse

"""
To run this file run
python generate_smaller_datasets.py {dataset}
where dataset is msvd, msrvtt, breakfast
"""


def sample_annotations(input_path, n, seed=42):
    """
    Sample 1/n * 100% of entries from a JSON annotation dict.

    Args:
        input_path (str): Path to the input JSON file.
        n (int): Inverse of sampling ratio (1/n * 100% kept).
        seed (int): Random seed for reproducibility.
    """
    assert os.path.exists(input_path), f"Input file does not exist: {input_path}"
    assert n > 0, "n must be a positive integer."

    # Compute output path by appending "_small" before ".json"
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_small{ext}"

    # Load original JSON dict
    with open(input_path, 'r') as f:
        data = json.load(f)

    total_entries = len(data)
    sample_size = max(1, total_entries // n)

    # Sample
    random.seed(seed)
    sampled_keys = random.sample(list(data.keys()), sample_size)
    sampled_data = {k: data[k] for k in sampled_keys}

    # Save to new JSON file (overwrite mode)
    with open(output_path, 'w') as f:
        json.dump(sampled_data, f, indent=2)

    # Verbose output
    print("=== Dataset Sampling Summary ===")
    print(f"Input dataset:       {input_path}")
    print(f"Total entries:       {total_entries}")
    print(f"Sampling fraction:   1/{n} ({100 / n:.2f}%)")
    print(f"Entries sampled:     {sample_size}")
    print(f"Output file written: {output_path}")
    print("================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a smaller subset of the dataset annotations.")
    parser.add_argument("dataset", type=str, help="Dataset name (e.g., 'breakfast', 'msvd', 'msrvtt')")
    parser.add_argument("train_or_test",type=str,help="Training, validation, or test")
    args = parser.parse_args()
    dataset = args.dataset.lower()        # 'breakfast', 'msvd', or 'msrvtt'
    train_or_test = args.train_or_test.lower()  # 'train', 'val', or 'test'

    #n chosen such that the run time is roughly 15 minutes
    if dataset == 'breakfast':
        n = 1
        if train_or_test == 'val':
            input_path = "./data/breakfast/annotation/val.json"
            sample_annotations(input_path, n=n)
        elif train_or_test == 'test':
            input_path = "./data/breakfast/annotation/test.json"
            sample_annotations(input_path, n=n)
        elif train_or_test == 'train':
            input_path = "./data/breakfast/annotation/train.json"
            sample_annotations(input_path, n=n)
        else:
            print("Error: Invalid split for 'breakfast'. Choose from 'train', 'val', or 'test'.")

    elif dataset == 'msvd':
        if train_or_test == 'val':
            sample_annotations("./data/msvd/annotation/qa_val.json", n=1)
            sample_annotations("./data/msvd/annotation/cap_val.json", n=6)
        elif train_or_test == 'test':
            sample_annotations("./data/msvd/annotation/qa_test.json", n=2)
            sample_annotations("./data/msvd/annotation/cap_test.json", n=39)
        elif train_or_test == 'train':
            sample_annotations("./data/msvd/annotation/qa_train.json", n=5)   
            sample_annotations("./data/msvd/annotation/cap_train.json", n=98)  # may be wrong
        else:
            print("Error: Invalid split for 'msvd'. Choose from 'train', 'val', or 'test'.")

    elif dataset == 'msrvtt':
        if train_or_test == 'val':
            sample_annotations("./data/msrvtt/annotation/qa_val.json", n=2)
            sample_annotations("./data/msrvtt/annotation/cap_val.json", n=14)
        elif train_or_test == 'test':
            sample_annotations("./data/msrvtt/annotation/qa_test.json", n=12)
            sample_annotations("./data/msrvtt/annotation/cap_test.json", n=83)
        elif train_or_test == 'train':
            sample_annotations("./data/msrvtt/annotation/qa_train.json", n=4)    # Placeholder: make sure files exist
            sample_annotations("./data/msrvtt/annotation/cap_train.json", n=15)  # Placeholder: make sure files exist
        else:
            print("Error: Invalid split for 'msrvtt'. Choose from 'train', 'val', or 'test'.")

    else:
        print("Error: Dataset not found. Choose from 'breakfast', 'msvd', or 'msrvtt'.")
