import pandas as pd


class LoadData:

    def __init__(self, file_path):
        self.file_path = file_path

    def data(self):
        # SYNTAX FIX: header=True is invalid — header takes an int (row index) or None.
        # header=0 means the first row is the column header, which is the default/correct behaviour.
        dataset = pd.read_csv(self.file_path, header=0)

        return dataset
