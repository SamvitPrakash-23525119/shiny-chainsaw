from src.models.observation import Observation
from src.utils.csv_loader import load


class ObservationBuilder:
    """Build `Observation` objects from loaded tabular data.

    Converts a DataFrame loaded from CSV into a list of model objects used
    by the preprocessing pipeline.
    """

    def retrieveObservations(self, data_path):
        """Load a dataset from disk and build observation objects.

        Args:
            data_path (str): Path to the CSV file to load.

        Returns:
            List[Observation]: Observations built from the CSV rows.
        """
        df = load(file_path=data_path)
        observations = self._build(df)

        return observations

    def _build(self, df):
        """Convert a DataFrame into a list of `Observation` objects.

        Args:
            df (pandas.DataFrame): Input DataFrame with observation columns.

        Returns:
            List[Observation]: One observation per row in the DataFrame.
        """

        observations = []

        for _, row in df.iterrows():

            observation = Observation(
                employee_department=row["employee_department"],
                employee_campus=row["employee_campus"],
                employee_position=row["employee_position"],
                employee_seniority_years=row["employee_seniority_years"],
                is_contractor=row["is_contractor"],
                employee_classification=row["employee_classification"],
                has_foreign_citizenship=row["has_foreign_citizenship"],
                has_criminal_record=row["has_criminal_record"],
                has_medical_history=row["has_medical_history"],
                employee_origin_country=row["employee_origin_country"],

                total_printed_pages=row["total_printed_pages"],
                num_printed_pages_off_hours=row["num_printed_pages_off_hours"],

                total_files_burned=row["total_files_burned"],
                burned_from_other=row["burned_from_other"],

                is_abroad=row["is_abroad"],
                trip_day_number=row["trip_day_number"],
                hostility_country_level=row["hostility_country_level"],

                num_entries=row["num_entries"],
                num_unique_campus=row["num_unique_campus"],
                late_exit_flag=row["late_exit_flag"],
                entry_during_weekend=row["entry_during_weekend"],

                is_malicious=row["is_malicious"]
            )

            observations.append(observation)

        return observations