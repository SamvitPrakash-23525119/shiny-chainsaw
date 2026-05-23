from dataclasses import dataclass

@dataclass(frozen=True)
class IdentitySignature:
    """Immutable dataclass representing an employee identity signature.

    The identity signature gathers static attributes about an employee
    used to contextualise observations and aggregated features.

    Attributes:
        employee_department (str): Department name.
        employee_campus (str): Campus identifier.
        employee_position (str): Job position or title.
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