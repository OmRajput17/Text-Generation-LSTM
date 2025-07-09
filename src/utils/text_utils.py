import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter

def get_wordnet_pos(tag):
    """Map POS tag to WordNet POS tag."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(raw_text: str, remove_rare: bool = True, min_freq: int = 2):
    """
    Preprocess text: clean, tokenize, lemmatize, and optionally remove rare words.
    
    Parameters:
        raw_text (str): Raw input text
        remove_rare (bool): Whether to remove rare words
        min_freq (int): Minimum frequency threshold for keeping words

    Returns:
        List[str]: Preprocessed tokens
    """
    # Clean text
    text = raw_text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
    ]

    # Remove rare words if enabled
    if remove_rare:
        word_freq = Counter(lemmatized_tokens)
        lemmatized_tokens = [word for word in lemmatized_tokens if word_freq[word] >= min_freq]

    return lemmatized_tokens
