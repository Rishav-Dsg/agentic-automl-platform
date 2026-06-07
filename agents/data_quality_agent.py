import pandas as pd

class DataQualityAgent:
    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def analyze(self):
        findings = []
        missing = self.df.isnull().sum().sum()

        if missing > 0:
            findings.append(
                f"{missing} missing values detected."
            )
        else:
            findings.append(
                "No missing values detected."
            )

        duplicates = self.df.duplicated().sum()

        if duplicates:
            findings.append(
                f"{duplicates} duplicate rows detected."
            )

        numeric_cols = self.df.select_dtypes(
            include="number"
        ).columns
        corr_matrix = self.df[
            numeric_cols
        ].corr().abs()

        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] > 0.9):
                    high_corr.append(
                        (
                            corr_matrix.columns[i],
                            corr_matrix.columns[j]
                        )
                    )

        if high_corr:
            findings.append(
                f"{len(high_corr)} highly correlated feature pairs detected."
            )
        target_dist = (
            self.df[self.target_column]
            .value_counts(normalize=True)
        )

        if target_dist.max() > 0.7:
            findings.append(
                "Potential class imbalance detected."
            )
        return findings