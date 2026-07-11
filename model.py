# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.num_layers = num_layers
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=lstm_dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch, seq_len, hidden_size*2)
        # hidden: (num_layers*2, batch, hidden_size)
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Luong's concat attention
        self.attn = nn.Linear(hidden_size + hidden_size*2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch, hidden_size) - forward hidden of last layer
        # encoder_outputs: (batch, seq_len, hidden_size*2)
        batch_size, seq_len, _ = encoder_outputs.size()
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat((decoder_hidden_expanded, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(combined))
        attention_weights = F.softmax(self.v(energy).squeeze(-1), dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(embed_size + hidden_size*2, hidden_size,
                            num_layers, batch_first=True, dropout=lstm_dropout)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size + hidden_size*2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        # Bridge bidirectional encoder states (2*hidden) to decoder states (hidden).
        self.state_bridge_h = nn.Linear(hidden_size * 2, hidden_size)
        self.state_bridge_c = nn.Linear(hidden_size * 2, hidden_size)

    def _bridge_bidir_state(self, state, bridge_layer):
        # state: (num_layers*2, batch, hidden_size) -> (num_layers, batch, hidden_size)
        num_directions_times_layers, batch_size, hidden_size = state.size()
        layers = num_directions_times_layers // 2
        state = state.view(layers, 2, batch_size, hidden_size)
        forward_state = state[:, 0, :, :]
        backward_state = state[:, 1, :, :]
        merged = torch.cat((forward_state, backward_state), dim=2)
        merged = merged.reshape(layers * batch_size, hidden_size * 2)
        bridged = torch.tanh(bridge_layer(merged))
        return bridged.view(layers, batch_size, hidden_size)

    def _prepare_decoder_state(self, hidden, cell):
        # Accept both encoder bidirectional state shape and already-decoder-shaped state.
        if hidden.size(0) == self.num_layers * 2:
            hidden = self._bridge_bidir_state(hidden, self.state_bridge_h)
            cell = self._bridge_bidir_state(cell, self.state_bridge_c)

        if hidden.size(0) != self.num_layers or cell.size(0) != self.num_layers:
            raise ValueError(
                f"Unexpected hidden/cell shape for decoder: hidden={tuple(hidden.shape)}, "
                f"cell={tuple(cell.shape)}, expected first dim {self.num_layers}."
            )

        return hidden, cell
        
    def forward(self, x, encoder_outputs, hidden, cell):
        # x: (batch, 1) - current input token
        embedded = self.embedding(x)  # (batch, 1, embed_size)

        hidden, cell = self._prepare_decoder_state(hidden, cell)
        
        # Use hidden state from the last decoder layer for attention.
        last_hidden = hidden[-1]  # (batch, hidden_size)
        
        context, _ = self.attention(last_hidden, encoder_outputs)
        context = context.unsqueeze(1)  # (batch, 1, hidden_size*2)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        rnn_input = self.dropout(rnn_input)
        
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        output = output.squeeze(1)  # (batch, hidden_size)
        context = context.squeeze(1)  # (batch, hidden_size*2)
        combined = torch.cat((output, context), dim=1)
        logits = self.fc(combined)
        return logits, hidden, cell