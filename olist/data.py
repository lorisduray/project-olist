import os
import pandas as pd


class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys are 'sellers', 'orders', 'order_items' etc...
        Its values are pandas.DataFrames loaded from csv files
        """

        root_dir = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(root_dir, "data", "csv")

        file_names = [f for f in os.listdir(csv_path) if f.endswith(".csv")]

        key_names = [
            key_name.replace("olist_", "").replace("_dataset", "").replace(".csv", "")
            for key_name in file_names
        ]

        # Create the dictionary
        data = {}
        for k, f in zip(key_names, file_names):
            data[k] = pd.read_csv(os.path.join(csv_path, f))

        return data
