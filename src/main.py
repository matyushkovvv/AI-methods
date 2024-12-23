from model import NGramm

if __name__ == "__main__":
    model = NGramm()

    bigramma = model.train('C:\development\Python\Методы_ИИ\coursework\src\\resources', n_gramm=2)
    model.predict('Кирилл', length_output=3)
