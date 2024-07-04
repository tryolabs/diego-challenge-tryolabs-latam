from pathlib import Path
from typing import List, Tuple, Union

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class DelayModel:

    TOP_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air",
    ]

    DELAY_THRESHOLD_MINUTES = 1

    MODEL_FILE_NAME = "model.joblib"
    MODEL_PATH = Path("challenge/models")

    def __init__(self):
        self._model: Union[BaseEstimator, ClassifierMixin] = self._load_model(
            self.complete_model_path
        )

    @property
    def complete_model_path(self) -> Path:
        """
        Returns the complete model path including the model file name.

        Returns
        -------
        Path
            Complete path to the model file.
        """
        return self.MODEL_PATH / self.MODEL_FILE_NAME

    def _get_minute_diff(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate the difference in minutes between two datetime columns in a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing two columns:
            - 'Fecha-O': datetime or str, a datetime column or string in the format '%Y-%m-%d %H:%M:%S'
            - 'Fecha-I': datetime or str, a datetime column or string in the format '%Y-%m-%d %H:%M:%S'

        Returns
        -------
        pd.Series
            A Series containing the difference in minutes between 'Fecha-O' and 'Fecha-I' for each row.
        """
        try:
            fecha_o = pd.to_datetime(data["Fecha-O"])
            fecha_i = pd.to_datetime(data["Fecha-I"])
            return (fecha_o - fecha_i).dt.total_seconds() / 60
        except (ValueError, KeyError) as e:
            raise ValueError("Invalid input data or date format") from e

    def _create_one_hot_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates one-hot encoded features for the specified columns in the input
        DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the columns "OPERA", "TIPOVUELO",
            and "MES".

        Returns
        -------
        pd.DataFrame
            A DataFrame with one-hot encoded features for "OPERA", "TIPOVUELO", and "MES" columns.
        """
        one_hot_features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        return one_hot_features

    def _create_delay_target(
        self, data: pd.DataFrame, target_column: str
    ) -> pd.DataFrame:
        """
        Creates a delay target column based on the difference in minutes.

        Parameters
        ----------
        data : pd.DataFrame
            The input DataFrame containing the necessary columns for delay
            calculation.
        target_column : str
            The name of the column to be created for delay target.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the specified target column indicating whether the
        delay exceeds the threshold (1 if delay > threshold, else 0).

        """
        data["min_diff"] = self._get_minute_diff()
        delay_target = np.where(data["min_diff"] > self.DELAY_THRESHOLD_MINUTES, 1, 0)
        return pd.DataFrame({target_column: delay_target}, index=data.index)

    def _load_model(self, path: str):
        """
        Load a saved model from the given path.

        Parameters
        ----------
        path : str
            Path to the saved model file.
        """
        self._model = joblib.load(path)

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        target = None

        if target_column:
            target = self._create_delay_target(data)

        features = self._create_one_hot_features(data)

        # Ensure all expected feature columns are present
        features = features.reindex(columns=self.TOP_FEATURES, fill_value=0)

        return features, target

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return
