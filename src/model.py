import os
import re
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from collections import Counter
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

class NGramm:
    def __init__(self, load_path = None):
        self.__ngramms = Counter()
        self.__ngramms_size = 0
        self.__vocab_size = 0

    def save(self, save_path=None):
        """Сохраняет модель в файл."""
        if save_path is None:
            save_path = f"{self.__ngramms_size}_grams.pkl"
        print(f"Saving {self.__ngramms_size}-grams model to {save_path}...")
        with open(save_path, 'wb') as file:
            pickle.dump({
                "ngramms": self.__ngramms,
                "ngramms_size": self.__ngramms_size,
                "vocab_size": self.__vocab_size
            }, file)
        print("Model saved successfully.")

    def load(self, load_path):
        """Загружает модель из файла."""
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"File {load_path} not found.")
        print(f"Loading model from {load_path}...")
        with open(load_path, 'rb') as file:
            data = pickle.load(file)
            self.__ngramms = data["ngramms"]
            self.__ngramms_size = data["ngramms_size"]
            self.__vocab_size = data["vocab_size"]
        print("Model loaded successfully.")

    def __train(self, path, n_gramm, stop_words):
        
        tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

        if n_gramm < 2:
            self.__ngramms_size = 2
        else:
            self.__ngramms_size = n_gramm

            if os.path.isfile(path):
                with open(path, 'r', encoding='utf-8') as file:
                    text = re.sub(r'(?<=\w)-(?=\w)', ' ', file.read())
                    
                    tokens = tokenizer.tokenize(text.lower())

                    filtered_tokens = [token for token in tokens if token not in stop_words and not re.match(r'^\d+$', token)]
                    file_n_grams = list(ngrams(filtered_tokens, self.__ngramms_size))

                    self.__ngramms.update(file_n_grams)
                    self.__vocab_size = len(set(tokens))

        return self.__ngramms
    

    def __validation(self, path, n_gramm):
        validation_dir = os.path.join(path, 'val')

        correct = 0
        total = 0

        for index, filename in enumerate(os.listdir(validation_dir)):
            print(f"[{index + 1}/{len(os.listdir(validation_dir))}] Validation on {filename}")

            validation_file = os.path.join(validation_dir, filename)
            if os.path.isfile(validation_file):
                with open(validation_file, 'r', encoding='utf-8') as file:
                    text = re.sub(r'[-.,;:\'\"!?(){}\[\]<>_+=*/\\|~^&«»@#$%\—\'\'""``\d]+', ' ', file.read())

                    words = text.lower().split()


                with tqdm(total=len(words), desc='Validation Progress') as pbar:
                    proposal = []
                    for index, word in enumerate(words):
                        if len(proposal) == n_gramm:
                            proposal.pop(0)
                        proposal.append(word)


                        string = ' '.join(proposal)
                        result = self.predict(string)

                        expected_next_word = proposal[-1]
                        if result:
                            predicted_next_word = result[0][0][-1]

                            if predicted_next_word == expected_next_word:
                                correct += 1
                        total += 1
                        pbar.update(1)

        print("Validation is done")
        return correct / total if total > 0 else 0

    def train(self, path, n_gramm=3, stop_words=[
            '.', ',', ';', ':', '\'', '\"', '!', '?', 
            '(', ')', '[', ']', '{', '}', '<', '>', '-', 
            '_', '=', '+', '*', '/', '\\', '|', '~', '`', 
            '^', '&', '0', '1', '2', '3', '4', '5', '6', 
            '7', '8', '9', '@', '#', '$', '%', '—', '\'\'',
            '""', '``', '«»'
        ]):
        print("Training...")

        train_dir = os.path.join(path, 'train')
        file_list = os.listdir(train_dir)
        total_files = len(file_list)

        validation_thresholds = [int(total_files * p) for p in [0.25, 0.50, 0.75, 1.0]]
        current_threshold_index = 0

        percentages = []
        accuracies = []

        for index, filename in enumerate(file_list, start=1):
            print(f"[{index}/{total_files}] Train on {filename}")

            train_file = os.path.join(train_dir, filename)
            self.__train(train_file, n_gramm, stop_words)

            if (current_threshold_index < len(validation_thresholds) 
                    and index >= validation_thresholds[current_threshold_index]):
                percentage = (validation_thresholds[current_threshold_index] / total_files) * 100
                print(f"Validation at {percentage:.0f}%...")
                results = self.__validation(path, n_gramm)
                print(f"Accuracy at {percentage:.0f}%: {results:.4f}")
                
                percentages.append(percentage)
                accuracies.append(results * 100)
                current_threshold_index += 1

        plt.figure()
        plt.plot(percentages, accuracies, marker='o', label="Validation Accuracy")
        plt.title("Validation Accuracy During Training")
        plt.xlabel("Percentage of Dataset Used for Training (%)")
        plt.ylabel("Accuracy (%)")
        plt.grid()
        plt.legend()

        output_path = os.path.join(path, "validation_accuracy.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Validation accuracy graph saved at {output_path}")


        self.save()
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
