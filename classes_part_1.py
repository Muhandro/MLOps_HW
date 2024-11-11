from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
import category_encoders as ce
from sklearn.model_selection import train_test_split

# %%
class DataParserABC(ABC):
    @abstractmethod
    def parse_data_types(self, data):
        """
        Абстрактный метод для определения типов данных в датасете.

        :param data: DataFrame с исходными данными.
        :return: Словарь, где ключи — названия столбцов, а значения — типы данных.
        """
        pass


# %%
class DataPreprocessorABC(ABC):
    @abstractmethod
    def label_encoding(self, data, columns):
        """
        Абстрактный метод для выполнения label_encoding.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :return: обработанный DataFrame.
        """
        pass

    @abstractmethod
    def one_hot_encoding(self, data, columns):
        """
        Абстрактный метод для выполнения one_hot_encoding.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :return: обработанный DataFrame.
        """
        pass

    @abstractmethod
    def target_encoding(self, data, columns, target):
        """
        Абстрактный метод для выполнения target_encoding.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :param target: Название целевой переменной.
        :return: обработанный DataFrame.
        """
        pass

    @abstractmethod
    def frequency_encoding(self, data, columns):
        """
        Абстрактный метод для выполнения frequency_encoding.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :return: обработанный DataFrame.
        """
        pass

    @abstractmethod
    def binary_encoding(self, data, columns):
        """
        Абстрактный метод для выполнения binary_encoding.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :return: обработанный DataFrame.
        """
        pass

    @abstractmethod
    def hashing_encoding(self, data, columns, n_components=8):
        """
        Абстрактный метод для выполнения hashing_encoding.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :param n_components: Количество компонентов для хеширования.
        :return: обработанный DataFrame.
        """
        pass

    @abstractmethod
    def standard_scaling(self, data, columns):
        """
        Абстрактный метод для выполнения standard_scaling.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для масштабирования.
        :return: обработанный DataFrame.
        """
        pass

    @abstractmethod
    def min_max_scaling(self, data, columns):
        """
        Абстрактный метод для выполнения min_max_scaling.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для масштабирования.
        :return: обработанный DataFrame.
        """
        pass

    @abstractmethod
    def robust_scaling(self, data, columns):
        """
        Абстрактный метод для выполнения robust_scaling.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для масштабирования.
        :return: обработанный DataFrame.
        """
        pass


# %%
class DataSplitterABC(ABC):
    @abstractmethod
    def split_data(self, data, target, test_size, random_state, **kwargs):
        """
        Абстрактный метод для разделения данных на обучающую и тестовую выборки.

        :param data: DataFrame с признаками.
        :param target: Название целевой переменной.
        :param test_size: Размер тестовой выборки.
        :param random_state: Случайное состояние для воспроизводимости.
        :return: Кортеж (X_train, X_test, y_train, y_test)
        """
        pass


# %%
class RandomSplitter(DataSplitterABC):
    def split_data(self, data, target, test_size, random_state, **kwargs):
        """
        Разделяет данные на обучающую и тестовую выборки случайным образом.

        :param data: DataFrame с признаками и целевой переменной.
        :param target: Название целевой переменной.
        :param test_size: Размер тестовой выборки.
        :param random_state: Случайное состояние.
        :return: Кортеж (X_train, X_test, y_train, y_test)
        """
        X = data.drop(columns=[target])
        y = data[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, **kwargs
        )
        return X_train, X_test, y_train, y_test


# %%
class DataParser(DataParserABC):
    def parse_data_types(self, dataset):
        """
        Определяет типы данных для каждого столбца в датасете.

        :param dataset: DataFrame с исходными данными.
        :return: Словарь, где ключи — названия столбцов, а значения — типы данных.
        """
        data_types = {column: dataset[column].dtype for column in dataset.columns}
        return data_types


# %%
class DataPreprocessor(DataPreprocessorABC):
    def label_encoding(self, data, columns):
        """
        Применяет label_encoding к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :return: DataFrame.
        """
        le = LabelEncoder()
        for col in columns:
            data[col] = le.fit_transform(data[col])
        return data

    def one_hot_encoding(self, data, columns):
        """
        Применяет one_hot_encoding к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :return: DataFrame.
        """
        return pd.get_dummies(data, columns=columns)

    def target_encoding(self, data, columns, target):
        """
        Применяет target_encoding к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :param target: Название целевой переменной.
        :return: DataFrame.
        """
        te = ce.TargetEncoder(cols=columns)
        data[columns] = te.fit_transform(data[columns], data[target])
        return data

    def frequency_encoding(self, data, columns):
        """
        Применяет frequency_encoding к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :return: DataFrame.
        """
        for col in columns:
            freq_encoding = data[col].value_counts() / len(data)
            data[col] = data[col].map(freq_encoding)
        return data

    def binary_encoding(self, data, columns):
        """
        Применяет binary_encoding к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :return: DataFrame.
        """
        be = ce.BinaryEncoder(cols=columns)
        data = be.fit_transform(data)
        return data

    def hashing_encoding(self, data, columns, n_components=8):
        """
        Применяет hashing_encoding к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для кодирования.
        :param n_components: Количество компонентов для хеширования.
        :return: DataFrame.
        """
        he = ce.HashingEncoder(cols=columns, n_components=n_components)
        data = he.fit_transform(data)
        return data

    def standard_scaling(self, data, columns):
        """
        Применяет standard_scaling к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для масштабирования.
        :return: DataFrame.
        """
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])
        return data

    def min_max_scaling(self, data, columns):
        """
        Применяет min_max_scaling к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для масштабирования.
        :return: DataFrame.
        """
        scaler = MinMaxScaler()
        data[columns] = scaler.fit_transform(data[columns])
        return data

    def robust_scaling(self, data, columns):
        """
        Применяет robust_scaling к указанным столбцам.

        :param data: DataFrame с данными.
        :param columns: Список столбцов для масштабирования.
        :return: DataFrame.
        """
        scaler = RobustScaler()
        data[columns] = scaler.fit_transform(data[columns])
        return data