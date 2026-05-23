from dataclasses import dataclass

@dataclass
class Observation:
    """Dataclass representing a single observation event.

    Contains both identity fields (employee metadata) and per-observation
    behavioral features used by the detection pipeline.

    Attributes:
        employee_department (str): Department name.
        employee_campus (str): Campus identifier.
        employee_position (str): Job position/title.
        employee_seniority_years (int): Years of seniority.
        is_contractor (bool): Whether the employee is a contractor.
        employee_classification (int): Numeric classification code.
        has_foreign_citizenship (bool): Whether the employee has foreign
            citizenship.
        has_criminal_record (bool): Whether the employee has a criminal
            record flag.
        has_medical_history (bool): Whether the employee has a medical
            history flag.
        employee_origin_country (str): Country of origin.
        hostility_country_level (int): Numeric hostility level for country
            context.

        total_printed_pages (int): Number of pages printed during the
            observation period.
        num_printed_pages_off_hours (int): Pages printed outside normal
            hours.
        total_files_burned (int): Number of files burned.
        burned_from_other (int): Files burned that originated from other
            users.
        is_abroad (bool): Whether the observation occurred abroad.
        trip_day_number (int): Day number of a trip.
        num_entries (int): Number of entries/access events.
        num_unique_campus (int): Number of unique campuses visited.
        late_exit_flag (bool): Whether a late exit was recorded.
        entry_during_weekend (bool): Whether entry occurred on weekend.
        is_malicious (bool | None): Label indicating whether the
            observation is malicious; may be `None` for unlabeled data.
    """
    employee_department: str
    employee_campus: str
    employee_position: str
    employee_seniority_years: int
    is_contractor: bool
    employee_classification: int
    has_foreign_citizenship: bool
    has_criminal_record: bool
    has_medical_history: bool
    employee_origin_country: str
    hostility_country_level: int
    
    total_printed_pages: int
    num_printed_pages_off_hours: int
    total_files_burned: int
    burned_from_other: int
    is_abroad: bool
    trip_day_number: int
    num_entries: int
    num_unique_campus: int
    late_exit_flag: bool
    entry_during_weekend: bool
    is_malicious: bool | None