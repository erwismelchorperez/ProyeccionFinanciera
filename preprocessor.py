class FinancialPreprocessor:
    def __init__(self, df, target_column, date_column='Year'):
        self.df = df
        self.target_column = target_column
        self.date_column = date_column

    def preprocess(self):
        df = self.df.dropna()
        X = df.drop(columns=[self.target_column, self.date_column])
        y = df[self.target_column]
        return X, y, df[self.date_column]