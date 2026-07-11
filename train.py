# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import pickle
import random
import numpy as np
from model import Encoder, Decoder

# Reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Hyperparameters
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
TEACHER_FORCING_RATIO = 0.5
MAX_LENGTH = 50
VALID_SPLIT = 0.1
GRAD_CLIP = 1.0
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required, but no CUDA device was detected. "
        "Install a CUDA-enabled PyTorch build and check NVIDIA drivers."
    )

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")

# Load data
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
with open('pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}, Number of pairs: {len(pairs)}")

class ChatDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        q_ids = [self.vocab['<SOS>']] + [self.vocab.get(w, self.vocab['<UNK>']) for w in q.split()] + [self.vocab['<EOS>']]
        a_ids = [self.vocab['<SOS>']] + [self.vocab.get(w, self.vocab['<UNK>']) for w in a.split()] + [self.vocab['<EOS>']]
        q_ids = q_ids[:MAX_LENGTH]
        a_ids = a_ids[:MAX_LENGTH]
        return torch.LongTensor(q_ids), torch.LongTensor(a_ids)

def collate_fn(batch):
    q_batch, a_batch = zip(*batch)
    q_padded = torch.nn.utils.rnn.pad_sequence(q_batch, batch_first=True, padding_value=0)
    a_padded = torch.nn.utils.rnn.pad_sequence(a_batch, batch_first=True, padding_value=0)
    return q_padded, a_padded

dataset = ChatDataset(pairs, vocab)

if len(dataset) < 2:
    raise ValueError("Not enough training pairs. Need at least 2 examples to train.")

val_size = max(1, int(len(dataset) * VALID_SPLIT))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# Models
encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <PAD>
encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)


def run_batch(q_tensor, a_tensor, teacher_forcing_ratio, training=True):
    q_tensor = q_tensor.to(DEVICE)
    a_tensor = a_tensor.to(DEVICE)

    encoder_outputs, hidden, cell = encoder(q_tensor)
    decoder_input = a_tensor[:, 0:1]  # first token is <SOS>
    target = a_tensor[:, 1:]          # shift target

    if target.size(1) == 0:
        return None, 0

    total_loss = 0.0
    non_pad_tokens = int((target != 0).sum().item())

    for t in range(target.size(1)):
        logits, hidden, cell = decoder(decoder_input, encoder_outputs, hidden, cell)
        total_loss = total_loss + criterion(logits, target[:, t])

        if training and random.random() < teacher_forcing_ratio:
            decoder_input = target[:, t:t+1]
        else:
            decoder_input = logits.argmax(dim=1, keepdim=True)

    denom = max(1, target.size(1))
    total_loss = total_loss / denom
    return total_loss, non_pad_tokens

def train_epoch():
    encoder.train()
    decoder.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (q_tensor, a_tensor) in enumerate(train_loader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss, _ = run_batch(q_tensor, a_tensor, TEACHER_FORCING_RATIO, training=True)
        if loss is None:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    if num_batches == 0:
        return float('inf')
    return total_loss / num_batches


@torch.no_grad()
def evaluate_epoch():
    encoder.eval()
    decoder.eval()
    total_loss = 0.0
    num_batches = 0

    for q_tensor, a_tensor in val_loader:
        loss, _ = run_batch(q_tensor, a_tensor, teacher_forcing_ratio=0.0, training=False)
        if loss is None:
            continue
        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        return float('inf')
    return total_loss / num_batches

print("Starting training...")
best_val_loss = float('inf')
for epoch in range(1, EPOCHS+1):
    train_loss = train_epoch()
    val_loss = evaluate_epoch()
    print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(encoder.state_dict(), 'encoder_last.pt')
    torch.save(decoder.state_dict(), 'decoder_last.pt')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(encoder.state_dict(), 'encoder.pt')
        torch.save(decoder.state_dict(), 'decoder.pt')
        print(f"New best checkpoint saved at epoch {epoch}")

print("Training complete!")