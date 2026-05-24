import numpy as np
from numpy import ndarray
from typing import List
from dataclasses import dataclass, field
from .observation import Observation
from .identity_signature import IdentitySignature
from .aggregated_features import AggregatedFeatures

@dataclass
class UserEntity:
    """Represents a user and all data collected for that user.

    A `UserEntity` groups an identity signature, a list of observations,
    a vector representation, aggregated features and the maliciousness
    label used by the pipeline.

    Attributes:
        identity (IdentitySignature): Static identity metadata for the user.
        observations (List[Observation]): Observations associated with the
            user.
        vector (ndarray): Vector representation of the user entity.
        aggregated_features (AggregatedFeatures): Aggregated features for
            the user, or `None` if not computed yet.
        isMalicious (bool): Flag indicating whether the user is malicious.
    """
    identity: IdentitySignature
    observations: List[Observation] = field(default_factory=list)
    vector: ndarray = field(default_factory=lambda: np.array([]))
    aggregated_features: AggregatedFeatures = None
    isMalicious: bool = False