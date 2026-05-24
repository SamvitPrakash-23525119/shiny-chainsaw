import numpy as np

from src.IsolationForest.models.aggregated_features import AggregatedFeatures

FEATURE_NAMES = [
    "employee_seniority_years",
    "is_contractor",
    "has_foreign_citizenship",
    "has_criminal_record",
    "has_medical_history",
    "avg_total_printed_pages",
    "avg_off_hours_print_ratio",
    "avg_files_burned",
    "avg_burned_from_other",
    "avg_num_entries",
    "avg_unique_campus_ratio",
    "late_exit_ratio",
    "weekend_entry_ratio",
    "avg_trip_duration",
    "avg_hostility_country_level",
    "abroad_ratio",
    "risk_score",
]


class FeatureVectorizer:
    """Convert aggregated feature objects into numeric vectors."""

    def vectorize(
        self,
        features: AggregatedFeatures
    ) -> np.ndarray:
        """Convert aggregated features into a fixed-length numpy vector.

        Args:
            features (AggregatedFeatures): Aggregated features for a user.

        Returns:
            np.ndarray: Dense float vector representation of the features.
        """

        return np.array([
            features.employee_seniority_years,
            features.is_contractor,
            features.has_foreign_citizenship,
            features.has_criminal_record,
            features.has_medical_history,
            features.avg_total_printed_pages,
            features.avg_off_hours_print_ratio,
            features.avg_files_burned,
            features.avg_burned_from_other,
            features.avg_num_entries,
            features.avg_unique_campus_ratio,
            features.late_exit_ratio,
            features.weekend_entry_ratio,
            features.avg_trip_duration,
            features.avg_hostility_country_level,
            features.abroad_ratio,
            features.risk_score

        ], dtype=float)