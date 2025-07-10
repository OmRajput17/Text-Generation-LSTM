# 🧠 LSTM Text Generator (Pride and Prejudice)

This project is an LSTM-based text generation model trained on *Pride and Prejudice* by Jane Austen. It uses a custom-trained Word2Vec embedding layer and generates text based on a given prompt using top-k sampling with temperature control.

---

## 📁 Project Structure

```
textgen_project/
├── src/
│   ├── train.py              # Train the model
│   ├── generate.py           # Load and generate text
│   └── utils/
│       └── text_utils.py     # Preprocessing utilities
├── models/
│   └── lstm_text_gen_pride.h5  # Trained LSTM model
├── tokenizer/
│   └── tokenizer.pkl         # Tokenizer object
├── vectors/
│   └── word2vec_pride.model  # Gensim Word2Vec embeddings
├── data/
│   └── pride_and_prejudice.txt # Training corpus
├── main.py                   # Entry point to generate text
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 How to Run

### ✅ 1. Clone the repo & install dependencies
         
```bash
git clone https://github.com/OmRajput17/Text-Generation-LSTM.git
cd textgen_lstm
pip install -r requirements.txt
```

### ✅ 2. Generate text from trained model

```bash
python main.py
```

Output:
```
She was quite determined to meet him again but the invitation...
```

> Modify the prompt and temperature settings directly in `main.py` or via `generate.py`.

---

## 🛠 How It Works

- ✅ **Text Cleaning & Lemmatization** using NLTK
- ✅ **Word Embeddings** with custom-trained Gensim Word2Vec
- ✅ **Sequence Generation** using Tokenizer
- ✅ **LSTM + Bidirectional Layer** trained on word sequences
- ✅ **Top-k + Temperature Sampling** for creative, coherent output

---

## 📦 Dependencies

```txt
tensorflow==2.12.0
keras==2.12.0
gensim==4.3.2
nltk==3.8.1
numpy==1.23.5
h5py==3.9.0
```

---

## ✍️ Sample Output

Prompt: `She was quite`

Output:
```
She was quite certain that he would return the next morning with her father...
```

---

## 🧠 Future Ideas

- Add CLI input (`--prompt "Start here"`)
- Streamlit web interface
- Export generations to `.txt` or `.pdf`
- Train on your own dataset

---

## 📜 License

MIT License. Content from Project Gutenberg is in the public domain.