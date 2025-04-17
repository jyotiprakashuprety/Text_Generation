import streamlit as st
import torch
import pickle
import numpy as np

# Load mappings
with open("data/char2idx.pkl", "rb") as f:
    char2idx = pickle.load(f)
with open("data/idx2char.pkl", "rb") as f:
    idx2char = pickle.load(f)

vocab_size = len(char2idx)

# Define the model class (matching the saved model architecture)
class CharModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, rnn_type="LSTM"):
        super().__init__()
        self.rnn_type = rnn_type
        self.embedding = torch.nn.Embedding(vocab_size, embed_size)
        if rnn_type == "LSTM":
            self.rnn = torch.nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = torch.nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out.squeeze(1))
        return out, hidden


# Load model dynamically
def load_model(path, rnn_type):
    model = CharModel(vocab_size, hidden_size=512, rnn_type=rnn_type)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# Text generation
def generate_text(model, start_str, char2idx, idx2char, gen_size=500, temperature=1.0):
    input_eval = torch.tensor([[char2idx[c] for c in start_str if c in char2idx]], dtype=torch.long)
    text_generated = []
    hidden = None

    with torch.no_grad():
        for _ in range(gen_size):
            predictions, hidden = model(input_eval[:, -1:], hidden)
            predictions = predictions.div(temperature).exp()
            predicted_id = torch.multinomial(predictions, 1).item()

            input_eval = torch.cat([input_eval, torch.tensor([[predicted_id]])], dim=1)
            text_generated.append(idx2char[predicted_id])

    return start_str + ''.join(text_generated)

# Streamlit UI
st.title("📜 Shakespearean Text Generator")
start_input = st.text_input("Enter seed text", "To be, or not to be")
length = st.slider("Length of generated text", 100, 1000, 300)
temperature = st.slider("Creativity (temperature)", 0.5, 2.0, 1.0)

model_choice = st.radio("Choose model type", ["LSTM", "GRU"])
model_path = "data/model/best_model_lstm.pth" if model_choice == "LSTM" else "data/model/best_model_gru.pth"

if st.button("Generate"):
    model = load_model(model_path, rnn_type=model_choice)
    output = generate_text(model, start_input, char2idx, idx2char, gen_size=length, temperature=temperature)
    st.text_area("Generated Shakespearean Text", output, height=300)
