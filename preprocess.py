# preprocess.py
import kagglehub
import os
import re
import nltk
import pickle
import csv
from collections import Counter
import shutil

nltk.download('punkt')

DATASET_ID = "grafstor/simple-dialogs-for-chatbot"


def find_dataset_file(path):
    """Return the best available dataset file from a Kaggle download directory."""
    candidates = []
    for root, _, files in os.walk(path):
        for name in files:
            candidates.append(os.path.join(root, name))

    # 1) Legacy format expected by older code.
    for fp in candidates:
        if os.path.basename(fp).lower() == "dialogs.txt":
            return fp

    # 2) CSV fallback for datasets that ship structured files.
    preferred_csv = [
        "sft_combined.csv",
        "sft_500_starter.csv",
        "sft_100_hindi.csv",
        "sft_template.csv",
    ]
    lower_map = {os.path.basename(fp).lower(): fp for fp in candidates}
    for name in preferred_csv:
        if name in lower_map:
            return lower_map[name]

    # 3) Final fallback: any txt/csv file.
    for fp in candidates:
        if fp.lower().endswith((".txt", ".csv")):
            return fp

    raise FileNotFoundError(
        "No supported dataset file found. Expected dialogs.txt or a CSV dataset file."
    )


def extract_pairs(file_path):
    """Extract question-answer pairs from either TXT (tab-separated) or CSV files."""
    ext = os.path.splitext(file_path)[1].lower()
    pairs = []

    if ext == ".txt":
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    if ext == ".csv":
        with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            if not fields:
                return pairs

            # Known column mappings used by common chatbot datasets.
            known_mappings = [
                ("Instruction", "Response"),
                ("input_text", "target_text"),
                ("prompt", "desired_response"),
                ("question", "answer"),
                ("input", "output"),
                ("query", "response"),
            ]

            q_col, a_col = None, None
            for q_name, a_name in known_mappings:
                if q_name in fields and a_name in fields:
                    q_col, a_col = q_name, a_name
                    break

            if q_col is None or a_col is None:
                # Heuristic fallback if column names are non-standard.
                lower_to_original = {c.lower(): c for c in fields}
                q_keywords = ("instruction", "prompt", "question", "input", "query")
                a_keywords = ("response", "answer", "target", "output")

                for key, original in lower_to_original.items():
                    if q_col is None and any(k in key for k in q_keywords):
                        q_col = original
                    if a_col is None and any(k in key for k in a_keywords):
                        a_col = original

            if q_col is None or a_col is None:
                raise ValueError(
                    f"Could not infer Q/A columns in CSV: {file_path}. Columns: {fields}"
                )

            for row in reader:
                q = str(row.get(q_col, "") or "")
                a = str(row.get(a_col, "") or "")
                if q.strip() and a.strip():
                    pairs.append((q, a))
        return pairs

    raise ValueError(f"Unsupported dataset file type: {ext}")

def download_dataset():
    """Download the dataset from Kaggle using kagglehub."""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download(DATASET_ID)
    file_path = find_dataset_file(path)
    print(f"Dataset resolved: {file_path}")
    return file_path

def clean_text(text):
    """Lowercase, remove non-alphanumeric characters (keep spaces)."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def build_vocab(file_path, min_freq=2):
    """Read question-answer pairs, clean them, and build vocabulary."""
    pairs = []
    word_counts = Counter()

    raw_pairs = extract_pairs(file_path)
    for q, a in raw_pairs:
        q = clean_text(q)
        a = clean_text(a)
        if not q or not a:
            continue
        pairs.append((q, a))
        for w in q.split() + a.split():
            word_counts[w] += 1
    
    # Special tokens
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for w, cnt in word_counts.items():
        if cnt >= min_freq:
            vocab[w] = len(vocab)
    
    return pairs, vocab

if __name__ == "__main__":
    dataset_path = download_dataset()
    print("Building vocabulary...")
    pairs, vocab = build_vocab(dataset_path)
    
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    with open('pairs.pkl', 'wb') as f:
        pickle.dump(pairs, f)
    
    # Copy dataset to current directory for convenience.
    local_copy_name = os.path.basename(dataset_path)
    shutil.copy(dataset_path, local_copy_name)
    
    print(f"Done. Vocabulary size: {len(vocab)}, Pairs: {len(pairs)}")