# chat.py
import os
import re
import torch
import pickle
from model import Encoder, Decoder

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required, but no CUDA device was detected. "
        "Use the same CUDA-enabled environment used for training."
    )

DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")

EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
MAX_LENGTH = 50
TEMPERATURE = 0.9
TOP_K = 8
RETRIEVAL_THRESHOLD = 0.22
MIN_INPUT_CHARS = 3
MIN_DECODER_TOKENS = 3
MIN_AVG_CONFIDENCE = 0.12
BOT_NAME = "RRR28 Bot"

GREETINGS = {
    "hi", "hello", "hey", "yo", "hola", "namaste", "hii", "heya"
}

SHORT_INPUT_RESPONSES = {
    "yes": "Nice. Tell me more.",
    "yeah": "Great. Tell me more.",
    "yep": "Got it. What would you like to discuss next?",
    "no": "Okay. Could you share a bit more detail?",
    "nope": "No problem. What should we talk about instead?",
    "why": "Can you add a little context so I can answer properly?",
    "ok": "Alright.",
    "okay": "Alright.",
    "thanks": "You are welcome.",
    "thx": "You are welcome.",
}

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
with open('pairs.pkl', 'rb') as f:
    pairs = pickle.load(f)

vocab_size = len(vocab)
idx_to_word = {idx: word for word, idx in vocab.items()}


def clean_text(text):
    """Lowercase and remove non-alphanumeric characters (keep spaces)."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def resolve_checkpoint(preferred, fallback):
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"Checkpoint not found. Tried '{preferred}' and '{fallback}'."
    )


def build_retrieval_index(raw_pairs):
    index = []
    for q, a in raw_pairs:
        q_clean = clean_text(q)
        a_clean = clean_text(a)
        q_tokens = set(q_clean.split())
        if q_tokens and a_clean:
            index.append((q_tokens, a_clean))
    return index


def retrieve_response(cleaned_input, retrieval_index, threshold=RETRIEVAL_THRESHOLD):
    input_tokens = set(cleaned_input.split())
    if not input_tokens:
        return None

    best_score = 0.0
    best_answer = None

    for q_tokens, answer in retrieval_index:
        inter = len(input_tokens & q_tokens)
        if inter == 0:
            continue
        if len(input_tokens) >= 3 and inter < 2:
            continue
        union = len(input_tokens | q_tokens)
        score = inter / max(1, union)
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score >= threshold:
        return best_answer
    return None


def should_ask_for_more_detail(cleaned_input, tokens):
    if cleaned_input in GREETINGS:
        return False
    if cleaned_input in SHORT_INPUT_RESPONSES:
        return False
    if len(cleaned_input) < MIN_INPUT_CHARS:
        return True
    return False


def low_quality_generation(decoded_words, avg_confidence):
    if len(decoded_words) < MIN_DECODER_TOKENS:
        return True
    if avg_confidence < MIN_AVG_CONFIDENCE:
        return True
    return False


def intent_response(cleaned_input):
    if cleaned_input in SHORT_INPUT_RESPONSES:
        return SHORT_INPUT_RESPONSES[cleaned_input]

    if "your name" in cleaned_input or cleaned_input == "name" or cleaned_input == "name?":
        return f"My name is {BOT_NAME}."

    if "who are you" in cleaned_input:
        return f"I am {BOT_NAME}, your chatbot assistant."

    if "how are you" in cleaned_input:
        return "I am doing well. How can I help you today?"

    if "i am your creator" in cleaned_input or "im your creator" in cleaned_input:
        return "Nice to meet you, creator. What should I learn next?"

    return None

encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

encoder_ckpt = resolve_checkpoint('encoder.pt', 'encoder_last.pt')
decoder_ckpt = resolve_checkpoint('decoder.pt', 'decoder_last.pt')
encoder.load_state_dict(torch.load(encoder_ckpt, map_location=DEVICE))
decoder.load_state_dict(torch.load(decoder_ckpt, map_location=DEVICE))
encoder.eval()
decoder.eval()
retrieval_index = build_retrieval_index(pairs)
print(f"Model loaded successfully from {encoder_ckpt} and {decoder_ckpt}.")

def predict_response(sentence, max_length=MAX_LENGTH):
    cleaned = clean_text(sentence)
    tokens = cleaned.split()
    if not tokens:
        return "Please type a valid message."

    if cleaned in GREETINGS:
        return "Hello! Ask me a question and I will try to answer."

    rule_reply = intent_response(cleaned)
    if rule_reply is not None:
        return rule_reply

    retrieved = retrieve_response(cleaned, retrieval_index)
    if retrieved is not None:
        return retrieved

    if should_ask_for_more_detail(cleaned, tokens):
        return "Could you type a slightly longer question so I can answer better?"

    ids = [vocab.get(w, vocab['<UNK>']) for w in tokens]
    ids = [vocab['<SOS>']] + ids + [vocab['<EOS>']]
    input_tensor = torch.LongTensor(ids).unsqueeze(0).to(DEVICE)
    
    with torch.inference_mode():
        encoder_outputs, hidden, cell = encoder(input_tensor)
        decoder_input = torch.LongTensor([[vocab['<SOS>']]]).to(DEVICE)
        decoded_words = []
        generated_ids = set()
        confidence_scores = []
        
        for _ in range(max_length):
            logits, hidden, cell = decoder(decoder_input, encoder_outputs, hidden, cell)

            # Penalize immediate repetition.
            step_logits = logits.clone()
            for token_id in generated_ids:
                step_logits[0, token_id] -= 1.0

            full_probs = torch.softmax(step_logits, dim=-1)
            confidence_scores.append(float(full_probs.max(dim=-1).values.item()))

            step_logits = step_logits / max(TEMPERATURE, 1e-6)
            topk_vals, topk_idx = torch.topk(step_logits, k=min(TOP_K, step_logits.size(-1)), dim=-1)
            probs = torch.softmax(topk_vals, dim=-1)
            sampled = torch.multinomial(probs.squeeze(0), num_samples=1)
            predicted_idx = topk_idx.squeeze(0)[sampled].item()

            if predicted_idx == vocab['<EOS>']:
                break
            generated_ids.add(predicted_idx)
            decoded_words.append(idx_to_word.get(predicted_idx, '<UNK>'))
            decoder_input = torch.LongTensor([[predicted_idx]]).to(DEVICE)

    if not decoded_words:
        return "I am not sure how to respond yet."

    avg_confidence = sum(confidence_scores) / max(1, len(confidence_scores))
    if low_quality_generation(decoded_words, avg_confidence):
        return "I am still learning. Please ask a clearer question with a bit more detail."

    return ' '.join(decoded_words)

print("\nChatbot ready! Type 'quit' to exit.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("Bot: Goodbye!")
        break
    if not user_input.strip():
        continue
    response = predict_response(user_input)
    print(f"Bot: {response}")