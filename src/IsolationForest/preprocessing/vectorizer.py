import numpy as np

from src.models.aggregated_features import AggregatedFeatures


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