import numpy as np
import pickle
from keras.models import load_model
from gensim.models import Word2Vec

# ---------------------------
# Load Model and Tokenizer
# ---------------------------
def load_resources():
    model = load_model("models/lstm_text_gen_pride.h5")

    with open("tokenizer/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

# ---------------------------
# Sampling with temperature and top-k
# ---------------------------
def sample_top_k_with_temperature(preds, k=10, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    
    # Get top-k probabilities and indices
    top_k_indices = preds.argsort()[-k:][::-1]
    top_k_probs = preds[top_k_indices]

    # Temperature scaling
    top_k_probs = np.log(top_k_probs + 1e-10) / temperature
    top_k_probs = np.exp(top_k_probs)
    top_k_probs /= np.sum(top_k_probs)

    return np.random.choice(top_k_indices, p=top_k_probs)

# ---------------------------
# Text generation function
# ---------------------------
def generate_text(prompt, tokenizer, model, seq_length=20, num_words=30, top_k=10, temperature=1.0):
    result = []
    input_text = prompt.lower()

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = token_list[-seq_length:]  # clip to seq_length
        token_list = np.pad(token_list, (seq_length - len(token_list), 0))
        token_list = token_list.reshape(1, seq_length)

        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_id = sample_top_k_with_temperature(predicted_probs, k=top_k, temperature=temperature)

        # Convert index back to word
        next_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                next_word = word
                break

        input_text += ' ' + next_word
        result.append(next_word)

    return prompt + ' ' + ' '.join(result)
