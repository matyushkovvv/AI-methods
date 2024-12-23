from model import NGramm

if __name__ == "__main__":
    model = NGramm()

    bigramma = model.train('C:\development\Python\Методы_ИИ\coursework\src\\resources', n_gramm=2)
    print(model.predict('созвонимся', length_output=3))

    # TODO: Нужно добавить в модель сохранение и загрузку модели
    # TODO: Нужно добавить сглаживание Лапласса
    # TODO: Нужно сделать GUI
