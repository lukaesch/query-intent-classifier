#!/usr/bin/env python3
"""
Generate embeddings for training data using the local embedding service.
"""

import json
import requests
import numpy as np
from tqdm import tqdm

EMBEDDING_URL = "http://localhost:5001/predictions"
BATCH_SIZE = 32  # Process in batches for efficiency

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of texts."""
    response = requests.post(
        EMBEDDING_URL,
        json={"input": {"inputs": texts}},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()
    return result["output"]

def process_data(input_file: str, output_file: str):
    """Process a data file and add embeddings."""
    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Generating embeddings for {len(data)} examples...")

    # Process in batches
    all_embeddings = []
    for i in tqdm(range(0, len(data), BATCH_SIZE)):
        batch = data[i:i + BATCH_SIZE]
        texts = [ex["query"] for ex in batch]
        embeddings = get_embeddings(texts)
        all_embeddings.extend(embeddings)

    # Add embeddings to data
    for i, ex in enumerate(data):
        ex["embedding"] = all_embeddings[i]

    # Save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Saved to {output_file}")

    # Also save as numpy arrays for easier training
    X = np.array([ex["embedding"] for ex in data])
    y = np.array([1 if ex["label"] == "question" else 0 for ex in data])

    np_output = output_file.replace(".json", "")
    np.save(f"{np_output}_X.npy", X)
    np.save(f"{np_output}_y.npy", y)
    print(f"Saved numpy arrays: {np_output}_X.npy, {np_output}_y.npy")

def main():
    # Check if embedding service is running
    try:
        response = requests.get("http://localhost:5001/", timeout=5)
        print("Embedding service is running")
    except:
        print("ERROR: Embedding service not running at localhost:5001")
        print("Start it with: docker start embedding-model")
        return

    process_data("train_data.json", "train_embeddings.json")
    process_data("test_data.json", "test_embeddings.json")

    print("\nDone! Next step: python train_classifier.py")

if __name__ == "__main__":
    main()
