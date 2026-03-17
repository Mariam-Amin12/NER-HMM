from sklearn.base import defaultdict
import nltk
nltk.download('punkt')
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.stem import PorterStemmer



def threshold_data(sequences, threshold):
    word_counts = defaultdict(int)
    for sequence in sequences:
        for token, state in sequence:
            word_counts[token] += 1

    unk_sequences = []
    for sequence in sequences:
        unk_sequence = [(token if word_counts[token] > threshold else 'UNK', state) for token, state in sequence]
        unk_sequences.append(unk_sequence)

    return unk_sequences

def preprocessing(sentence):
    tokens =nltk.word_tokenize(sentence)
    # doc = nlp(sentence)
    # tokens = [token.text for token in doc if not token.is_punct]
    # print(tokens)
    stemmer = PorterStemmer()
    stems = [stemmer.stem(token) for token in tokens]
    # print(stems)

    return stems


