import numpy as np
from statistics import mean
from src.models.user_entity import UserEntity
from src.models.aggregated_features import AggregatedFeatures


class FeatureAggregator:
    """Aggregate observations and identity fields into summary features.

    This helper computes averaged behavioural metrics for a user entity and
    derives a heuristic risk score from the aggregated data.
    """
    CRIMINAL_HISTORY_WEIGHT = 1.5
    CONTRACTOR_WEIGHT = 0.5
    FOREIGN_CITIZENSHIP_HOSTILE_COUNTRY_WEIGHT = 1.0
    OFF_HOURS_PRINTING_WEIGHT = 2.0

    def aggregate(
        self,
        entity: UserEntity
    ) -> AggregatedFeatures:
        """Aggregate a user's observations into a single feature vector.

        Args:
            entity (UserEntity): User entity containing identity metadata
                and observations to aggregate.

        Returns:
            AggregatedFeatures: Computed aggregate metrics and risk score.
        """

        observations = entity.observations
        identity = entity.identity

        avg_total_printed_pages = mean(
            o.total_printed_pages for o in observations
        )

        avg_files_burned = mean(
            o.total_files_burned for o in observations
        )

        avg_burned_from_other = mean(
            o.burned_from_other for o in observations
        )

        avg_num_entries = mean(
            o.num_entries for o in observations
        )

        avg_trip_duration = mean(
            o.trip_day_number for o in observations
        )

        avg_hostility_country_level = mean(
            o.hostility_country_level for o in observations
        )

        off_hours_ratios = []

        for o in observations:
            if o.total_printed_pages > 0:
                ratio = (
                    o.num_printed_pages_off_hours /
                    o.total_printed_pages
                )
            else:
                ratio = 0.0

            off_hours_ratios.append(ratio)

        avg_off_hours_print_ratio = mean(off_hours_ratios)

        campus_ratios = []

        for o in observations:
            if o.num_entries > 0:
                ratio = (
                    o.num_unique_campus /
                    o.num_entries
                )
            else:
                ratio = 0.0

            campus_ratios.append(ratio)

        avg_unique_campus_ratio = mean(campus_ratios)

        late_exit_ratio = mean(
            float(o.late_exit_flag)
            for o in observations
        )

        weekend_entry_ratio = mean(
            float(o.entry_during_weekend)
            for o in observations
        )

        abroad_ratio = mean(
            float(o.is_abroad)
            for o in observations
        )

        risk_score = 0.0

        # criminal history
        risk_score += self.CRIMINAL_HISTORY_WEIGHT if identity.has_criminal_record else 0.0

        # contractor
        risk_score += self.CONTRACTOR_WEIGHT if identity.is_contractor else 0.0

        # foreign citizenship & hostile countries
        if (identity.has_foreign_citizenship and avg_hostility_country_level < identity.hostility_country_level):
            risk_score += self.FOREIGN_CITIZENSHIP_HOSTILE_COUNTRY_WEIGHT

        # excessive off-hours printing
        risk_score += avg_off_hours_print_ratio * self.OFF_HOURS_PRINTING_WEIGHT

        # file burning behavior
        risk_score += np.log1p(avg_files_burned)

        return AggregatedFeatures(
            employee_seniority_years=identity.employee_seniority_years,
            is_contractor=float(identity.is_contractor),
            has_foreign_citizenship=float(identity.has_foreign_citizenship),
            has_criminal_record=float(identity.has_criminal_record),
            has_medical_history=float(identity.has_medical_history),
            avg_total_printed_pages=avg_total_printed_pages,
            avg_off_hours_print_ratio=avg_off_hours_print_ratio,
            avg_files_burned=avg_files_burned,
            avg_burned_from_other=avg_burned_from_other,
            avg_num_entries=avg_num_entries,
            avg_unique_campus_ratio=avg_unique_campus_ratio,
            late_exit_ratio=late_exit_ratio,
            weekend_entry_ratio=weekend_entry_ratio,
            avg_trip_duration=avg_trip_duration,
            avg_hostility_country_level=avg_hostility_country_level,
            abroad_ratio=abroad_ratio,
            risk_score=risk_score
        )