import pandas as pd
import numpy as np

# SYNTAX FIX: Removed the module-level line `dataset = loading_dataset.LoadData()`
# It served no purpose here and crashed on import because file_path argument was missing.
# The import of loading_dataset is also removed since it is not needed in this file.


class PreProcess:

    def __init__(self, data):
        self.data = data.copy()

    def DataInfo(self):
        columns_names = self.data.columns
        data_shape = self.data.shape
        data_info = self.data.info()

        print("Columns Names: ", columns_names)
        print("Data Shape: ", data_shape)
        print("Data Info: ", data_info)

    def ColNames(self):
        column_names = {
            'Hipertension': 'Hypertension',
            'Handcap': 'Handicap',
            # LOGICAL FIX: The target column is named 'No-show' with a hyphen.
            # A hyphen in a column name breaks attribute-style access and many downstream operations.
            # Renaming it here so all other code can reference it cleanly as 'NoShow'.
            'No-show': 'NoShow'
        }
        self.data.rename(columns=column_names, inplace=True)
        return self.data

    def DropCols(self, cols):
        self.data.drop(columns=cols, inplace=True)
        print(f"Columns dropped: {cols}")
        return self.data

    def RemoveRows(self):
        # SYNTAX FIX: .drop() needs index labels, not the filtered rows themselves.
        # Added .index so we pass the row index numbers to drop.
        # LOGICAL FIX: Original only dropped Age == 0 but the dataset also contains
        # negative ages which are equally invalid. Changed to drop Age <= 0.
        self.data.drop(self.data[self.data["Age"] <= 0].index, inplace=True)
        return self.data

    def TimeCorrection(self):
        # LOGICAL FIX: Original code converted 'AppointmentID' (a numeric ID) to datetime.
        # The actual date column is 'AppointmentDay'. Corrected both column names.
        self.data['AppointmentDay'] = pd.to_datetime(self.data['AppointmentDay'])
        self.data['ScheduledDay'] = pd.to_datetime(self.data['ScheduledDay'])

        # Days between when the appointment was booked and when it is scheduled.
        self.data["DaysInAdvance"] = (
            self.data['AppointmentDay'].dt.date - self.data['ScheduledDay'].dt.date
        ).apply(lambda x: x.days)

        # Total hours between booking and appointment.
        self.data["HoursInAdvance"] = (
            self.data['AppointmentDay'] - self.data['ScheduledDay']
        ).dt.total_seconds() / 3600

        # SYNTAX FIX: '$M' and '$S' were dollar signs instead of '%M' and '%S'.
        # LOGICAL FIX: Original condition `6 <= x >= 13` is a chained comparison that
        # evaluates as (6 <= x) AND (x >= 13), meaning x >= 13 — Morning was never matched.
        # Corrected to proper range checks using AND logic.
        self.data['PartOfDay'] = (
            self.data['ScheduledDay'].dt.hour.apply(
                lambda x: 'Morning' if 6 <= x < 13
                else 'Afternoon' if 13 <= x < 19
                else 'Evening'
            )
        )

        print("******** Three new columns created *****")
        print(self.data.head(10))

        return self.data
