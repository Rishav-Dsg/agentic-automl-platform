import pandas as pd

class ProfilerAgent:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def profile_dataset(self):
        profile = {}

        profile["rows"] = self.df.shape[0]
        profile["columns"] = self.df.shape[1]

        numerical_cols = self.df.select_dtypes(
            include=['int64', 'float64']
        ).columns.tolist()
        categorical_cols = self.df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        profile["numerical_columns"] = numerical_cols
        profile["categorical_columns"] = categorical_cols

        missing_values = self.df.isnull().sum().to_dict()
        profile["missing_values"] = missing_values
        profile["duplicate_rows"] = self.df.duplicated().sum()

        if profile["rows"] < 1000:
            profile["dataset_size"] = "small"
        elif profile["rows"] < 100000:
            profile["dataset_size"] = "medium"
        else:
            profile["dataset_size"] = "large"

        profile["sparsity"] = (
            self.df.isnull().sum().sum()
            /
            (self.df.shape[0] * self.df.shape[1])
        )

        return profile