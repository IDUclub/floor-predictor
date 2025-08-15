import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split

class ModelHandler:
    """
    A class for handling machine learning models, including loading, saving, training, and making predictions.
    
    Attributes
    ----------
    model_path : str
        Path to the model file
    model : object
        The machine learning model
    target_col : str, optional
        Name of the target column (default is 'is_living')
    df : pd.DataFrame, optional
        DataFrame containing the data
    feature_order : list
        List of feature names in the order expected by the model
    """
    
    def __init__(self, model_path, target_col='is_living', df=None):
        """
        Initialize the ModelHandler.

        Parameters
        ----------
        model_path : str
            Path to the model file
        target_col : str, optional
            Name of the target column (default is 'is_living')
        df : pd.DataFrame, optional
            DataFrame containing the data (default is None)
        """
        self.model_path = model_path
        self.model = None
        self.target_col = target_col
        self.df = df

    def load_model_from_file(self):
        """
        Load a model from a file using pickle.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist at the specified path
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
                self.feature_order = self.model.feature_names_in_
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def save_model(self):
        """
        Save the current model to a file using pickle.

        Raises
        ------
        ValueError
            If no model is currently loaded
        """
        if self.model is None:
            raise ValueError("Cannot save an empty model.")
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

    def set_model(self, model):
        """
        Set an external model (e.g., trained in a notebook).

        Parameters
        ----------
        model : object
            The machine learning model to set
        """
        self.model = model

    def train_model(self, X=None, y=None, cv=5, save=True):
        """
        Train the model (default is DecisionTreeClassifier). Uses self.df if X and y are not provided.

        Parameters
        ----------
        X : pd.DataFrame, optional
            Training data features
        y : pd.Series, optional
            Training data labels
        cv : int, optional
            Number of cross-validation folds (default is 5)
        save : bool, optional
            Whether to save the model after training (default is True)

        Raises
        ------
        AttributeError
            If neither X/y nor self.df are provided
        """
        if X is None or y is None:
            if not hasattr(self, 'df') or self.df is None:
                raise AttributeError("Must provide X and y or set self.df")
            df = self.df.copy()
            X = df.drop(columns=[self.target_col, 'geometry'], errors='ignore')
            y = df[self.target_col].astype(int)

        self.model.fit(X, y)

        self.feature_order = self.model.feature_names_in_

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        print(f"[CV] Accuracy: mean={np.mean(cv_scores):.4f}, std={np.std(cv_scores):.4f}")

        if save:
            self.save_model()

    def train_test_split(self, test_size=0.3, random_state=42):
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        test_size : float, optional
            Proportion of the dataset to include in the test split (default is 0.3)
        random_state : int, optional
            Random seed for reproducibility (default is 42)

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test

        Raises
        ------
        AttributeError
            If self.df is not set
        X_train, X_test, y_train, y_test
        """
        if not hasattr(self, 'df') or self.df is None:
            raise AttributeError("train_test_split requires self.df to be set")

        df = self.df.copy()
        X = df.drop(columns=[self.target_col, 'geometry'], errors='ignore')
        y = df[self.target_col].astype(int)

        stratify = y if y.nunique() > 1 else None  # avoid error if only 1 class
        stratify = y if y.nunique() > 1 else None  # избежать ошибки, если только 1 класс
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    def predict(self, df, map_labels: bool = False):
        """
        Make predictions on input DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Data (can include geometry) from which non-numeric features will be removed
        map_labels : bool, optional
            If True, labels 0/1 will be replaced with 'NON_RES' / 'RES' (default is False)

        Returns
        -------
        pd.DataFrame
            DataFrame with added 'predicted' column

        Raises
        ------
        RuntimeError
            If no model is loaded
        ValueError
            If required features are missing from the input DataFrame
        """
        if self.model is None:
            raise RuntimeError("No model is set. Use set_model(), load_model_from_file() or train_model().")

        df = df.copy()
        X = df.drop(columns=['geometry'], errors='ignore')

        # Check feature order
        # Проверка порядка признаков
        missing_cols = set(self.feature_order) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_order)
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")
        if extra_cols:
            X = X[self.feature_order]  # remove extra features
        else:
            X = X[self.feature_order]  # reorder


        df['predicted'] = self.model.predict(X)

        if map_labels:
            mapping = {1: 'RES', 0: 'NON_RES'}
            df['predicted'] = df['predicted'].map(mapping).fillna(df['predicted'])

        return df



