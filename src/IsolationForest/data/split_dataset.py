import random
import pandas as pd
from typing import List, Tuple
from src.models.user_entity import UserEntity
from src.config.file_paths import TRAINING_DATA_PATH, TESTING_DATA_PATH, CLEANED_DATA_PATH
from src.pipeline.threat_detection_pipeline import ThreatDetectionPipeline



class SplitDataset:
    """Helper to split the cleaned dataset into train and test CSVs.

    This class wraps the project's `ThreatDetectionPipeline` preprocessing
    step, converts `UserEntity` objects into flat rows and writes the
    resulting train/test CSVs to the paths defined in configuration.
    """
    def __init__(self):
        """Create a SplitDataset and initialize the internal pipeline.

        Initializes a `ThreatDetectionPipeline` instance used for
        preprocessing the cleaned dataset into `UserEntity` objects.
        """
        self.pipeline = ThreatDetectionPipeline()

    def split_and_save(self):
        """Split the dataset into train/test and save as CSV files.

        The method runs preprocessing to obtain `UserEntity` objects, splits
        them according to the default train/test ratio, converts each set to a
        pandas `DataFrame`, and writes the CSVs to the configured paths.
        """
        entities = self._get_entities()
        train_entities, test_entities = self._split(entities)

        train_df = self._to_dataframe(train_entities)
        test_df = self._to_dataframe(test_entities)

        train_df.to_csv(TRAINING_DATA_PATH, index=False)
        test_df.to_csv(TESTING_DATA_PATH, index=False)

    def _get_entities(self) -> List[UserEntity]:
        """Run pipeline preprocessing and return `UserEntity` objects.

        Returns:
            List[UserEntity]: List of user entities produced by the
                pipeline's preprocessing step applied to the cleaned data
                CSV.
        """
        return self.pipeline._preprocessing(CLEANED_DATA_PATH)

    def _split(
        self,
        entities: List[UserEntity],
        train_ratio: float = 0.7,
        seed: int = 42
    ) -> Tuple[List[UserEntity], List[UserEntity]]:
        """Split a list of `UserEntity` objects into train and test lists.

        The split is performed at the entity level (not observation level).

        Args:
            entities (List[UserEntity]): List of user entities to split.
            train_ratio (float): Fraction of entities to place in the train
                set. Defaults to 0.7.
            seed (int): Random seed for reproducible shuffling. Defaults to
                42.

        Returns:
            Tuple[List[UserEntity], List[UserEntity]]: `(train_entities,
                test_entities)` lists.
        """

        rng = random.Random(seed)

        # shuffle ENTITIES (not observations)
        shuffled = entities.copy()
        rng.shuffle(shuffled)

        split_index = int(len(shuffled) * train_ratio)

        train_entities = shuffled[:split_index]
        test_entities = shuffled[split_index:]

        return train_entities, test_entities

    def _to_dataframe(self, entities: List[UserEntity]) -> pd.DataFrame:
        """Convert a list of `UserEntity` objects into a pandas DataFrame.

        Each `Observation` in every `UserEntity` becomes a single row in the
        resulting DataFrame with identity fields duplicated for each
        observation.

        Args:
            entities (List[UserEntity]): User entities to flatten.

        Returns:
            pd.DataFrame: DataFrame with one row per observation.
        """

        rows = []

        for entity in entities:

            sig = entity.identity

            for obs in entity.observations:

                row = {
                    "employee_department": sig.employee_department,
                    "employee_campus": sig.employee_campus,
                    "employee_position": sig.employee_position,
                    "employee_seniority_years": sig.employee_seniority_years,
                    "is_contractor": sig.is_contractor,
                    "employee_classification": sig.employee_classification,
                    "has_foreign_citizenship": sig.has_foreign_citizenship,
                    "has_criminal_record": sig.has_criminal_record,
                    "has_medical_history": sig.has_medical_history,
                    "employee_origin_country": sig.employee_origin_country,
                    "hostility_country_level": sig.hostility_country_level,
                    "total_printed_pages": obs.total_printed_pages,
                    "num_printed_pages_off_hours": obs.num_printed_pages_off_hours,
                    "total_files_burned": obs.total_files_burned,
                    "burned_from_other": obs.burned_from_other,
                    "is_abroad": obs.is_abroad,
                    "trip_day_number": obs.trip_day_number,
                    "num_entries": obs.num_entries,
                    "num_unique_campus": obs.num_unique_campus,
                    "late_exit_flag": obs.late_exit_flag,
                    "entry_during_weekend": obs.entry_during_weekend,
                    "is_malicious": obs.is_malicious
                }

                rows.append(row)

        return pd.DataFrame(rows)