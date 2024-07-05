import logging
from pathlib import Path
from typing import List, Tuple, Union

import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


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
        "OPERA_Copa Air"
    ]

    DATA_SPLITTING_RANDOM_STATE = 42
    TEST_SIZE = 0.33

    MODEL_RANDOM_STATE = 1
    LEARNING_RATE = 0.01

    DELAY_THRESHOLD_MINUTES = 15

    MODEL_FILE_NAME = Path("model.json")
    MODEL_PATH = Path("models")

    def __init__(self):
        self._model: Union[BaseEstimator, ClassifierMixin] = self._load_model(
            self.complete_model_path
        )
        self.target_column = None

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
        Calculate the difference in minutes between two datetime columns in a
        DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing two columns:
            - 'Fecha-O': datetime or str, a datetime column or string in the
            format '%Y-%m-%d %H:%M:%S'
            - 'Fecha-I': datetime or str, a datetime column or string in the
            format '%Y-%m-%d %H:%M:%S'

        Returns
        -------
        pd.Series
            A Series containing the difference in minutes between 'Fecha-O'
        and 'Fecha-I' for each row.
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
        data["min_diff"] = self._get_minute_diff(data)
        delay_target = np.where(data["min_diff"] > self.DELAY_THRESHOLD_MINUTES, 1, 0)
        return pd.DataFrame({target_column: delay_target}, index=data.index)

    def _load_model(self, model_path: Path) -> Union[BaseEstimator, ClassifierMixin]:
        """
        Load a saved XGBoost model from the given JSON path.

        Parameters
        ----------
        path : Path
            Path to the saved model JSON file.
        """
        logging.info(f"Loading model from {model_path}")

        try:
            # Check if the file exists and is a file (not a directory)
            if not model_path.is_file():
                raise FileNotFoundError(f"No file found at {model_path}")

            # Check if the file has a .json extension
            if model_path.suffix.lower() != ".json":
                logging.warning(
                    f"File {model_path} does not have a .json extension. Attempting to load anyway."
                )

            # Create a new XGBClassifier instance
            model = XGBClassifier()

            # Load the model from the JSON
            model.load_model(str(model_path))

            logging.info("Model loaded successfully")

            return model

        except FileNotFoundError as e:
            logging.error(f"Model file not found: {e}")
        except json.JSONDecodeError:
            logging.error(
                f"Error decoding JSON from {model_path}. Make sure it's a valid JSON file."
            )
        except PermissionError:
            logging.error(f"Permission denied when trying to read {model_path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")

    def _calculate_class_weights(self, target: pd.Series) -> dict:
        """
        Calculate class weights for balancing classes in a dataset.

        Parameters
        ----------
        target : pd.Series
            The target column containing class labels.

        Returns
        -------
        dict
            A dictionary where keys are class labels and values are the
        corresponding weights.
        """
        # Get unique classes
        classes = np.unique(target)

        # Compute class weights
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=target
        )

        # Create a dictionary mapping class labels to weights
        class_weights = dict(zip(classes, weights))

        print(class_weights)

        return class_weights

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
            self.target_column = target_column
            target = self._create_delay_target(data, target_column)

        features = self._create_one_hot_features(data)

        # Ensure all expected feature columns are present
        features = features.reindex(columns=self.TOP_FEATURES, fill_value=0)

        if target_column:
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=self.TEST_SIZE,
            random_state=self.DATA_SPLITTING_RANDOM_STATE,
        )

        target_series = target[self.target_column]
        class_weights = self._calculate_class_weights(target=target_series)

        # For binary classification, XGBoost uses scale_pos_weight
        scale_pos_weight = class_weights[1] / class_weights[0]

        print(scale_pos_weight)

        # Initialize and train the model
        model = XGBClassifier(
            random_state=self.MODEL_RANDOM_STATE,
            learning_rate=self.LEARNING_RATE,
            scale_pos_weight=scale_pos_weight,
        )

        model.fit(x_train, y_train)

        logging.info("Finished training, calculating test metrics...")

        # Predictions
        y_pred = model.predict(x_test)

        # Metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        logging.info(f"Classification Report:\n{class_report}")

        # Ensure the directory exists
        self.complete_model_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the model
        model.save_model(self.complete_model_path)
        self._model = model

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            logging.warning("Model wasn't found.")
            self._load_model(self.complete_model_path)

        predictions = self._model.predict(features)
        return predictions.tolist()
