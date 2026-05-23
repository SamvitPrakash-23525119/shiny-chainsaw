from dataclasses import dataclass


@dataclass
class AggregatedFeatures:
    """Dataclass representing aggregated per-user features.

    This structure holds averaged or aggregated behavioural features for a
    single user, computed across that user's observations. These fields are
    used by the detection pipeline and models to characterise user
    behaviour and compute risk scores.

    Attributes:
        employee_seniority_years (int): Years of seniority for the employee.
        is_contractor (bool): Whether the employee is a contractor.
        has_foreign_citizenship (bool): Whether the employee holds foreign
            citizenship.
        has_criminal_record (bool): Whether the employee has a criminal
            record.
        has_medical_history (bool): Whether the employee has a medical
            history flag.
        avg_total_printed_pages (float): Average total printed pages per
            observation.
        avg_off_hours_print_ratio (float): Average ratio of off-hours
            printing to total printing.
        avg_files_burned (float): Average number of files burned.
        avg_burned_from_other (float): Average number of files burned that
            came from other users.
        avg_num_entries (float): Average number of entries (e.g., accesses).
        avg_unique_campus_ratio (float): Average ratio of unique campus
            visits.
        late_exit_ratio (float): Fraction of observations with late exits.
        weekend_entry_ratio (float): Fraction of observations with weekend
            entries.
        avg_trip_duration (float): Average duration of trips (if applicable).
        avg_hostility_country_level (float): Average hostility level of
            countries associated with the user's origin or travel.
        abroad_ratio (float): Fraction of observations where the user was
            abroad.
        risk_score (float): Computed risk score for the user (higher is
            riskier).
    """
    employee_seniority_years: int
    is_contractor: bool
    has_foreign_citizenship: bool
    has_criminal_record: bool
    has_medical_history: bool
    avg_total_printed_pages: float
    avg_off_hours_print_ratio: float
    avg_files_burned: float
    avg_burned_from_other: float
    avg_num_entries: float
    avg_unique_campus_ratio: float
    late_exit_ratio: float
    weekend_entry_ratio: float
    avg_trip_duration: float
    avg_hostility_country_level: float
    abroad_ratio: float
    risk_score: float