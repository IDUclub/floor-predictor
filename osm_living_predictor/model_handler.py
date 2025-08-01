import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split

class ModelHandler:
    def __init__(self, model_path, target_col='is_living', df=None):
        self.model_path = model_path
        self.model = None
        self.target_col = target_col
        self.df = df

    def load_model_from_file(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
                self.feature_order = self.model.feature_names_in_
            print("[ModelHandler] Модель успешно загружена из файла.")
        else:
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

    def save_model(self):
        if self.model is None:
            raise ValueError("Невозможно сохранить пустую модель.")
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[ModelHandler] Модель сохранена: {self.model_path}")

    def set_model(self, model):
        """
        Установить внешнюю модель (например, обученную в ноутбуке).
        """
        self.model = model
        print("[ModelHandler] Внешняя модель установлена.")

    def train_model(self, X=None, y=None, cv=5, save=True):
        """
        Обучает модель (по умолчанию — DecisionTreeClassifier), использует self.df если X, y не переданы.
        """
        if X is None or y is None:
            if not hasattr(self, 'df') or self.df is None:
                raise AttributeError("Нужно передать X и y или задать self.df")
            df = self.df.copy()
            X = df.drop(columns=[self.target_col, 'geometry'], errors='ignore')
            y = df[self.target_col].astype(int)

        self.model.fit(X, y)

        self.feature_order = self.model.feature_names_in_

        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"[CV] Accuracy: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")

        if save:
            self.save_model()


    def train_test_split(self, test_size=0.3, random_state=42):
        """
        Делит данные на обучающую и тестовую выборки.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        if not hasattr(self, 'df') or self.df is None:
            raise AttributeError("Для использования train_test_split необходимо задать self.df")

        df = self.df.copy()
        X = df.drop(columns=[self.target_col, 'geometry'], errors='ignore')
        y = df[self.target_col].astype(int)

        stratify = y if y.nunique() > 1 else None  # избежать ошибки, если только 1 класс
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    def predict(self, df, map_labels: bool = False):
        """
        Предсказывает значения на входном DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Данные (можно с geometry), из которых будут удалены нечисловые признаки.
        map_labels : bool
            Если True, то метки 0/1 будут заменены на 'NON_RES' / 'RES'

        Returns
        -------
        df : pd.DataFrame
            DataFrame с добавленным столбцом 'predicted'
        """
        if self.model is None:
            raise RuntimeError("Модель не установлена. Используйте set_model(), load_model_from_file() или train_model().")

        df = df.copy()
        X = df.drop(columns=['geometry'], errors='ignore')

        # Проверка порядка признаков
        missing_cols = set(self.feature_order) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_order)
        if missing_cols:
            raise ValueError(f"Отсутствуют признаки: {missing_cols}")
        if extra_cols:
            X = X[self.feature_order]  # удалить лишние признаки
        else:
            X = X[self.feature_order]  # упорядочить


        df['predicted'] = self.model.predict(X)

        if map_labels:
            mapping = {1: 'RES', 0: 'NON_RES'}
            df['predicted'] = df['predicted'].map(mapping).fillna(df['predicted'])

        return df



