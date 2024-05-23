import os
from Mobile_Price_Predictor.logging import logger
import pandas as pd
from Mobile_Price_Predictor.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_data(self, df, m_dep_threshold=0.5, px_height_threshold=65, sc_w_threshold=2.54):
        """
        Apply transformations to the data.

        Parameters:
        df (DataFrame): The data to transform.
        is_train (bool): Whether the data is training data or test data.
        m_dep_threshold (float): The threshold for 'm_dep' column.
        px_height_threshold (int): The threshold for 'px_height' column.
        sc_w_threshold (float): The threshold for 'sc_w' column.

        Returns:
        DataFrame: The transformed data.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected input to be a pandas DataFrame")

        df.loc[df["m_dep"] < m_dep_threshold, "m_dep"] = m_dep_threshold
        df.loc[df["px_height"] < px_height_threshold, "px_height"] = px_height_threshold
        df.loc[df["sc_w"] < sc_w_threshold, "sc_w"] = sc_w_threshold

        return df

    def convert(self):
        # Read the data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        logger.info("Data read successfully")

        # Transform the data
        train_data = self.transform_data(train_data )
        test_data = self.transform_data(test_data)
        test_data.drop(['id'], axis=1, inplace=True)
        logger.info("Data transformed successfully")
        
        train_data.to_csv(os.path.join(self.config.root_dir, "train_data.csv"))
        test_data.to_csv(os.path.join(self.config.root_dir, "test_data.csv"))
        logger.info("Train and Test data made successfully")
