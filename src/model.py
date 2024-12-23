import os
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from collections import Counter

class NGramm:
    def __init__(self, load_path = None):
        self.__ngramms = Counter()
        self.__vocab_size = 0

    def train(self, path, n_gramm = 3, stop_words = [
            '.', ',', ';', ':', '\'',  '\"', '!', '?', 
            '(', ')', '[', ']', '{', '}', '<', '>', '-', 
            '_', '=', '+', '*', '/', '\\', '|', '~', '`', 
            '^', '&', '0', '1', '2', '3', '4', '5', '6', 
            '7', '8', '9', '@', '#', '$', '%', '—', '\'\'',
            '""', '``'
    ]):
        tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

        for index, filename in enumerate(os.listdir(path)):
            print(f"[{index + 1}/{len(os.listdir(path))}] Train on {filename}")

            train_file = os.path.join(path, filename)
            if os.path.isfile(train_file):
                with open(train_file, 'r', encoding='utf-8') as file:
                    text = re.sub(r'(?<=\w)-(?=\w)', ' ', file.read())
                    
                    tokens = tokenizer.tokenize(text.lower())

                    filtered_tokens = [token for token in tokens if token not in stop_words and not re.match(r'^\d+$', token)]
                    file_n_grams = list(ngrams(filtered_tokens, n_gramm))

                    self.__ngramms.update(file_n_grams)
                    self.__vocab_size = len(set(tokens))

        print("Saving " + str(n_gramm) + "-grams...")
        with open(str(n_gramm) + '_grams.txt', 'w', encoding='utf-8') as file:
            for n_gram, count in self.__ngramms.items():
                file.write(f"{' '.join(n_gram)}\t{count}\n")

        print("Processing and saving is done")

        return self.__ngramms
    

    def predict(self, word, length_output = 3):

        predicted_ngramms = []
        for ngramm, count in self.__ngramms.items():
            if word == ngramm[0]:
                predicted_ngramms.append((ngramm, count))

        predicted_ngramms.sort(key=lambda x: x[1], reverse=True)
        predicted_ngramms = predicted_ngramms[:length_output]

        print(predicted_ngramms)

