
#  Text Generation using RNNs (GRU & LSTM)

This project focuses on building a character-level text generation model using Shakespeare's works. We implemented and compared GRU and LSTM-based RNNs using PyTorch, performed hyperparameter tuning, and deployed a web app using Streamlit.

##  Live Demo
**Streamlit App:** *[Insert your deployment link here]*

##  Project Structure
```
project/
│
├── data/
│   ├── shakespeare.txt
│   ├── char2idx.pkl
│   └── idx2char.pkl
│
├── models/
│   ├── best_model_gru.pth
│   └── best_model_lstm.pth
│
├── notebooks/
│   └── text_generation_experiments.ipynb
│
├── app.py                  # Streamlit web app
├── train.py                # Model training and tuning
├── generate.py             # Text generation script
├── README.md
└── requirements.txt
```

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/text-generation-rnn.git
cd text-generation-rnn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

##  Project Steps Followed

### Step 1: Dataset Selection
- Shakespeare text corpus (`shakespeare.txt`)
- Sequential data (character-level)

### Step 2: Data Cleaning
- Removed non-ASCII characters
- Normalized whitespace and line breaks

### Step 3: Data Exploration
- Frequency distribution of characters and words
- Sentence length distribution
- Character type breakdown

### Step 4: Feature Engineering
- Created `char2idx` and `idx2char` mappings
- Encoded text into integer sequences

### Step 5: Model Selection
- Implemented both **GRU** and **LSTM** models in PyTorch
- Architecture: Embedding → RNN (GRU/LSTM) → Linear

### Step 6: Hyperparameter Tuning
- Grid search over:
  - Embedding size, hidden size, dropout, learning rate, batch size
- Evaluation: validation accuracy, loss, and perplexity

### Step 7: Model Saving
- Saved best performing models:
  - `best_model_gru.pth`
  - `best_model_lstm.pth`

### Step 8: Web App
- Created a Streamlit app to:
  - Select model (GRU or LSTM)
  - Enter seed text
  - Generate Shakespeare-style text

### Step 9: GitHub Upload
- All source code and files are included here

### Step 10: Cloud Deployment
- Hosted on Streamlit Cloud *(link above)*

## 📊 Visualizations Included
- Validation Loss vs Epoch
- Validation Accuracy vs Epoch
- Perplexity vs Epoch

## 📌 Uniqueness
- Though Shakespeare is a common dataset, we ensured uniqueness by:
  - Comparing both GRU and LSTM on validation metrics
  - Calculating combined score (accuracy/perplexity)
  - Visualizing model performance over epochs

## 📄 License
This project is for educational purposes.
