class TaskAgent:
    def __init__(self, df):
        self.df = df

    def detect_target_column(self):
        possible_targets = []

        for column in self.df.columns:
            unique_values = self.df[column].nunique()

            if unique_values < 20:
                possible_targets.append(column)
        return possible_targets

    def detect_task_type(self, target_column):
        unique_values = self.df[target_column].nunique()

        if unique_values < 20:
            return "classification"

        return "regression"

    def detect_id_columns(self):
        id_columns = []

        valid_patterns = [
            "id",
            "_id",
            "userid",
            "user_id",
            "customer_id",
            "product_id"
        ]

        for column in self.df.columns:

            column_lower = column.lower().strip()

            if column_lower in valid_patterns:
                id_columns.append(column)

        return id_columns