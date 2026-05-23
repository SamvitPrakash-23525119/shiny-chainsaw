from dataclasses import dataclass
from .user_entity import UserEntity

@dataclass
class Prediction:
    """Prediction result for a single user entity.

    Attributes:
        user_entity (UserEntity): The user entity that was evaluated.
        anomaly_score (float): Anomaly score assigned by the detection
            model.
        is_malicious (bool): Binary prediction indicating whether the user
            is considered malicious.
    """
    user_entity: UserEntity
    anomaly_score: float
    is_malicious: bool
    