import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class DataPreprocessor:
    def __init__(self, model_version=None, config=None, train=True):
        """
        Input:
          config (dict, optional): Configuration dictionary
        """
        self.model_version = model_version
        self.train_config = config.get("train") if config else None
        self.scaler = StandardScaler()
        self.imputer_mice = IterativeImputer(random_state=0)

        if train:
            self.process_training_data()

    def process_training_data(self):
        """Process the training data when config is provided."""
        raw_data = pd.read_parquet(self.train_config["source_data"])
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(raw_data)
        self.X_train_scaled, self.X_test_scaled = self.standarization(
            self.X_train, self.X_test
        )
        self.save_preprocessed_data()

    def preprocess_data(self, raw_data: pd.DataFrame, fit=True):
        """
        preprocess data, remove nulls and extreme values,i.e. inf from the raw data
        """
        # handle outliers, drop the rows that has the inf values
        raw_data = raw_data[~np.isinf(raw_data).any(axis=1)]
        inf_or_neg_inf_mask = np.isinf(raw_data) | (raw_data == -np.inf)
        print("Total number of inf or -inf values after dropping the inf:")
        print(inf_or_neg_inf_mask.sum().sum())

        # Extract labels and drop 'user' and 'gesture' columns before imputation
        Y = None
        if "gesture" in raw_data.columns and "user" in raw_data.columns:
            Y = raw_data["gesture"]
            X = raw_data.drop(columns=["user", "gesture"])
        else:
            X = raw_data

        # handle NANs, use multivariate imputation by chained equations
        print("Before the MICE imputation....")
        print(X.isnull().sum().sum())
        if fit:
            X = pd.DataFrame(self.imputer_mice.fit_transform(X), columns=X.columns)
        else:
            X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        print("NaN values after MICE imputation:")
        print(X.isna().sum().sum())

        return X, Y

    def split_data(self, raw_data):
        """
        Split input data to train and test set
        """
        X, Y = self.preprocess_data(raw_data)
        # split raw data to test and evaluation dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
        return X_train, X_test, y_train, y_test

    def standarization(self, X_train, X_test):
        """
        apply standardriztion to the input data
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def save_preprocessed_data(self):
        """
        save preprocessed data to corresponding files
        """
        train_dir = (
            self.train_config.get("train_data_root_path")
            + "/"
            + str(self.model_version)
        )
        test_dir = (
            self.train_config.get("test_data_root_path") + "/" + str(self.model_version)
        )
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        joblib.dump(self.X_train_scaled, f"{train_dir}/X.joblib")
        joblib.dump(self.X_test_scaled, f"{test_dir}/X.joblib")
        joblib.dump(self.y_train, f"{train_dir}/y.joblib")
        joblib.dump(self.y_test, f"{test_dir}/y.joblib")
        joblib.dump(self.scaler, f"{train_dir}/scaler.joblib")
        joblib.dump(self.imputer_mice, f"{train_dir}/imputer.joblib")

    @classmethod
    def load_for_inference(cls, model_version, config, train=False):
        """
        Load saved scaler and imputer for inference
        """
        preprocessor = cls(model_version=model_version, config=config, train=train)
        # base path to the data processor like for imputation and standarization
        base_path = (
            preprocessor.train_config.get("train_data_root_path")
            + "/"
            + str(preprocessor.model_version)
        )
        preprocessor.scaler = joblib.load(os.path.join(base_path, "scaler.joblib"))
        preprocessor.imputer = joblib.load(os.path.join(base_path, "imputer.joblib"))
        return preprocessor

    def prepare_for_inference(self, new_data):
        """
        Prepare new data points for inference
        """
        # Preprocess the data without fitting
        preprocessed_data, _ = self.preprocess_data(new_data, fit=False)

        # Standardize the data
        preprocessed_data_scaled = self.scaler.transform(preprocessed_data)

        return preprocessed_data_scaled


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
