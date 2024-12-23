import os
import re
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from collections import Counter

class NGramm:
    def __init__(self, load_path = None):
        self.__ngramms = Counter()
        self.__ngramms_size = 0
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

        if n_gramm < 2:
            self.__ngramms_size = 2
        else:
            self.__ngramms_size = n_gramm

        print("Training...")
        train_dir = os.path.join(path, 'train')

        for index, filename in enumerate(os.listdir(train_dir)):
            print(f"[{index + 1}/{len(os.listdir(train_dir))}] Train on {filename}")

            train_file = os.path.join(train_dir, filename)
            if os.path.isfile(train_file):
                with open(train_file, 'r', encoding='utf-8') as file:
                    text = re.sub(r'(?<=\w)-(?=\w)', ' ', file.read())
                    
                    tokens = tokenizer.tokenize(text.lower())

                    filtered_tokens = [token for token in tokens if token not in stop_words and not re.match(r'^\d+$', token)]
                    file_n_grams = list(ngrams(filtered_tokens, self.__ngramms_size))

                    self.__ngramms.update(file_n_grams)
                    self.__vocab_size = len(set(tokens))

        print("Saving " + str(self.__ngramms_size) + "-grams...")
        with open(str(self.__ngramms_size) + '_grams.txt', 'w', encoding='utf-8') as file:
            for n_gram, count in self.__ngramms.items():
                file.write(f"{' '.join(n_gram)}\t{count}\n")

        print("Processing and saving is done")

        return self.__ngramms
    

    def predict(self, proposal, length_output = 3):
        
        words = proposal.lower().split()
        context_length = len(words)

        predicted_ngramms = Counter()
        for ngramm, count in self.__ngramms.items():

            flag = True
            for index, word in enumerate(words):
                if word != ngramm[index]:
                    flag = False
                    break
            
            if flag:
                ngramm_length = context_length + 1
                predicted_ngramms.update({ngramm[0:ngramm_length]: count})

        predicted_ngramms = list(predicted_ngramms.items())

        predicted_ngramms.sort(key=lambda x: x[1], reverse=True)
        predicted_ngramms = predicted_ngramms[:length_output]

        return predicted_ngramms
