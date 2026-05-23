from typing import List, Dict
from src.models.observation import Observation
from src.models.identity_signature import IdentitySignature
from src.models.user_entity import UserEntity


class IdentificationResolver:
    """Group observations by identity signature into user entities.

    The resolver extracts a stable identity signature from each observation
    and combines observations that belong to the same user.
    """

    def _build_signature(self, obs: Observation) -> IdentitySignature:
        """Build an identity signature from a single observation.

        Args:
            obs (Observation): Observation containing identity fields.

        Returns:
            IdentitySignature: Immutable identity signature built from the
                observation's identity metadata.
        """
        return IdentitySignature(
            employee_department=obs.employee_department,
            employee_campus=obs.employee_campus,
            employee_position=obs.employee_position,
            employee_seniority_years=obs.employee_seniority_years,
            is_contractor=obs.is_contractor,
            employee_classification=obs.employee_classification,
            has_foreign_citizenship=obs.has_foreign_citizenship,
            has_criminal_record=obs.has_criminal_record,
            has_medical_history=obs.has_medical_history,
            employee_origin_country=obs.employee_origin_country,
            hostility_country_level=obs.hostility_country_level
        )

    def group_observations(
        self,
        observations: List[Observation]
    ) -> List[UserEntity]:
        """Group observations into user entities keyed by identity.

        Args:
            observations (List[Observation]): Observations to group.

        Returns:
            List[UserEntity]: User entities with observations attached and
                maliciousness propagated where any observation is marked
                malicious.
        """

        entity_map: Dict[IdentitySignature, UserEntity] = {}

        for obs in observations:

            sig = self._build_signature(obs)

            if sig not in entity_map:
                entity_map[sig] = UserEntity(identity=sig)

            if obs.is_malicious:
                entity_map[sig].isMalicious = True

            entity_map[sig].observations.append(obs)

        return list(entity_map.values())