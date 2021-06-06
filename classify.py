import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder


class DataLoader:   # Загрузка и предобработка данных
    def __init__(self, data_path: str, raw: bool = False):
        self.data_path = data_path
        self.raw = raw  # Нужна ли предобработка
        self.data = None
        self.X = None
        self.Y = None
        self.labels = None

    def load_data(self):
        self.data = pd.read_csv(self.data_path)     # Считывание данных

    def preproc_data(self):
        if self.raw:    # Предобработка данных
            cols = list(self.data.columns[1:4])
            res_col = self.data.columns[-2]
            encoder = LabelEncoder()
            for col in cols:
                self.data[f'num_{col}'] = np.array(encoder.fit_transform(self.data[col]))
            self.data[f'num_{res_col}'] = np.array(encoder.fit_transform(self.data[res_col]))
            self.labels = self.data['label']
            self.data = self.data.drop(['flag', 'service', 'protocol_type', 'label'])
        else:
            self.labels = self.data['label']
            self.data = self.data.drop(['label'], axis=1)

        self.Y = np.array(self.data['num_label'])   # Разбиение данных
        self.X = np.array(self.data.drop(['num_label'], axis=1))

        scaler = Normalizer().fit(self.X)   # Нормализация
        self.X = scaler.transform(self.X)

        self.X = self.X.reshape(len(self.data), 1, 39)  # Изменение формата данных

    def get_x(self):
        return self.X

    def get_y(self):
        return self.Y

    def get_labels(self):
        return self.labels


if __name__ == '__main__':
    # Инициализация парсера
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/test10.csv')
    parser.add_argument('--model', type=str, default='model')
    opt = parser.parse_args()

    # Загрузка и обработка данных
    data_load = DataLoader(opt.data)
    data_load.load_data()
    data_load.preproc_data()

    # Отделение лейблов от данных
    X = data_load.get_x()
    Y = data_load.get_y()
    labels = data_load.get_labels()

    # Инициализация модели
    model = tf.keras.models.load_model(opt.model)

    # Классификация
    predict = model.predict(X)

    # Подсчёт точности
    loss, accuracy = model.evaluate(X, tf.keras.utils.to_categorical(Y, 4))

    # Формирование результатов
    results = np.argmax(predict, axis=1)

    d = {'labels': labels, 'num_labels': Y, 'results ': results}
    df = pd.DataFrame(d)

    # Вывод результатов
    print(df)
    print("\nAccuracy: %.2f%%" % (accuracy * 100))



