import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class DataPreprocessor:
    def __init__(self, config):
        self.train_config = config["train"]
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()
        self.X_train_scaled, self.X_test_scaled = self.standarization()
        self.save_preprocessed_data()

    def preprocess_data(self):
        # Normalize training dataset
        # prepare
        raw_data = pd.read_parquet(self.train_config["source_data"])

        # handle outliers, drop the rows that has the inf values
        raw_data = raw_data[~np.isinf(raw_data).any(axis=1)]
        inf_or_neg_inf_mask = np.isinf(raw_data) | (raw_data == -np.inf)
        print("Total number of inf or -inf values after dropping the inf:")
        print(inf_or_neg_inf_mask.sum().sum())

        # handle NANs, use multivariate imputation by chained equations
        print("Before the MIC imputation....")
        print(raw_data.isnull().sum().sum())  # created by cursor
        imputer_mice = IterativeImputer(random_state=0)
        raw_data = pd.DataFrame(
            imputer_mice.fit_transform(raw_data), columns=raw_data.columns
        )
        print("NaN values after MICE imputation:")
        print(raw_data.isna().sum().sum())

        # extract labels from raw data
        Y = raw_data.gesture
        X = raw_data.drop(columns=["user", "gesture"])

        # split raw data to test and evaluation dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
        return X_train, X_test, y_train, y_test

    def standarization(self):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

    def save_preprocessed_data(self):
        train_dir = self.train_config.get("train_data_root_path")
        test_dir = self.train_config.get("test_data_root_path")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        joblib.dump(self.X_train_scaled, f"{train_dir}/X.joblib")
        joblib.dump(self.X_test_scaled, f"{test_dir}/X.joblib")
        joblib.dump(self.y_train, f"{train_dir}/y.joblib")
        joblib.dump(self.y_test, f"{test_dir}/y.joblib")


class GestureData(Dataset):
    def __init__(self, X, Y):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values
        self.X = torch.FloatTensor(X)
        # PyTorch's loss functions, like CrossEntropyLoss, expect class labels to be in the range 0 to (num_classes - 1).
        self.Y = torch.LongTensor(Y) - 1

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
