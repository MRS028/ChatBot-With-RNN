# ChatBot with Attention Seq2Seq (PyTorch)

End-to-end chatbot project built with a bidirectional LSTM encoder, Luong-style attention, and an LSTM decoder.

Current workflow:

1. Download and preprocess a Kaggle dataset.
2. Build training pairs and vocabulary.
3. Train encoder/decoder checkpoints on CUDA.
4. Chat interactively with a hybrid response pipeline.

## Project Structure

- `preprocess.py`: downloads dataset from Kaggle and creates `pairs.pkl` + `vocab.pkl`
- `model.py`: encoder, attention, and decoder definitions
- `train.py`: model training and checkpoint saving
- `chat.py`: interactive chatbot inference
- `requirements.txt`: pinned Python dependencies

Generated artifacts:

- `pairs.pkl`, `vocab.pkl`
- `encoder.pt`, `decoder.pt` (best validation checkpoint)
- `encoder_last.pt`, `decoder_last.pt` (latest epoch checkpoint)

## Dataset Handling

By default, preprocessing downloads:

- `grafstor/simple-dialogs-for-chatbot`

Supported input formats (auto-detected):

- `dialogs.txt` with tab-separated question/answer rows
- CSV files with known or inferred Q/A columns

CSV mappings recognized directly include:

- `Instruction` -> `Response`
- `input_text` -> `target_text`
- `prompt` -> `desired_response`
- `question` -> `answer`
- `input` -> `output`
- `query` -> `response`

## Requirements

- Windows + Python 3.14 (tested)
- NVIDIA GPU with CUDA-capable PyTorch build (required for `train.py` and `chat.py`)
- Kaggle access configured for `kagglehub`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If CUDA is not detected, install a CUDA wheel in the same environment (adjust CUDA version as needed):

```powershell
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
```

Verify environment:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

## Run

### 1) Preprocess

```powershell
python preprocess.py
```

Outputs:

- `pairs.pkl`
- `vocab.pkl`
- local copy of the resolved dataset file

### 2) Train

```powershell
python train.py
```

Behavior:

- exits immediately if CUDA is unavailable
- logs train/validation loss each epoch
- saves both latest and best checkpoints

### 3) Chat

```powershell
python chat.py
```

Type `quit`, `exit`, or `bye` to stop.

## Response Strategy

The chatbot uses a hybrid pipeline in this order:

1. Rule-based responses for greetings and short intents.
2. Retrieval from training pairs via token-overlap similarity.
3. Neural decoding fallback with top-k sampling, temperature, and repetition penalties.

This typically gives more stable replies than pure greedy generation.

## Troubleshooting

### `ModuleNotFoundError: No module named 'kagglehub'`

Use the venv interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe preprocess.py
```

### `CUDA GPU is required, but no CUDA device was detected`

`train.py` and `chat.py` are CUDA-only in the current implementation.

Checklist:

1. Run `nvidia-smi` and confirm your GPU is visible.
2. Confirm `torch.cuda.is_available()` returns `True` in the same venv.
3. Reinstall CUDA-enabled PyTorch for your CUDA version.

### Bot replies are repetitive or low-quality

Try:

1. Re-run preprocessing and retrain with improved data.
2. Ask clearer, longer prompts.
3. Tune decoding/retrieval constants in `chat.py` (for example `TOP_K`, `TEMPERATURE`, `RETRIEVAL_THRESHOLD`).

## Quick Commands

```powershell
python preprocess.py
python train.py
python chat.py
```
