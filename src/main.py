from model import NGramm
import tkinter as tk
from tkinter import scrolledtext

def start_gui(model, text_length):
    """Функция для запуска GUI."""
    root = tk.Tk()
    root.title("Предсказание следующего слова")

    def predict_next_word():
        input_text = text_input.get("1.0", tk.END).strip()
        if input_text:
            input_list = input_text.lower().split()
            input_text = ' '.join(input_list[-text_length:])

            predictions = model.predict(input_text, length_output=5)
            recommendations.delete(1.0, tk.END)
            if predictions:
                for i, (ngram, count) in enumerate(predictions, start=1):
                    recommendations.insert(tk.END, f"{i}. {' '.join(ngram)} (count: {count})\n")
            else:
                recommendations.insert(tk.END, "Нет рекомендаций.")
        else:
            recommendations.delete(1.0, tk.END)
            recommendations.insert(tk.END, "Введите текст для предсказания.")

    # Поле для ввода текста
    text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=5)
    text_input.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

    # Кнопка для получения предсказаний
    predict_button = tk.Button(root, text="Предсказать", command=predict_next_word)
    predict_button.grid(row=1, column=0, padx=10, pady=10)

    # Поле для отображения рекомендаций
    recommendations = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, state=tk.NORMAL)
    recommendations.grid(row=2, column=0, padx=10, pady=10, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    model = NGramm()

    n_gramm = 3
    model.train('C:\development\Python\Методы_ИИ\coursework\src\\resources', n_gramm=n_gramm)
    model.save()
    # model.load('C:\development\Python\Методы_ИИ\coursework\src\\2_grams.pkl')

    start_gui(model, text_length=n_gramm - 1)
