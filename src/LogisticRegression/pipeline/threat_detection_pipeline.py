import numpy as np
from src.IsolationForest.preprocessing.identification_resolver import IdentificationResolver
from src.IsolationForest.preprocessing.observation_builder import ObservationBuilder
from src.IsolationForest.preprocessing.feature_aggregator import FeatureAggregator
from src.IsolationForest.preprocessing.vectorizer import FeatureVectorizer
from src.IsolationForest.models.prediction import Prediction
from src.IsolationForest.utils.pickler import Pickler
from src.LogisticRegression.training.trainer import Trainer
from src.LogisticRegression.config.file_paths import (
    LR_MODEL_PATH,
    TRAINING_DATA_PATH,
    TESTING_DATA_PATH,
)

class ThreatDetectionPipeline:
    """Orchestrates preprocessing, training, model loading and testing for
    the logistic regression threat detector.

    The pipeline reuses the shared preprocessing components from the
    IsolationForest module and wires them together with the logistic
    regression trainer and model persistence utilities.
    """

    def __init__(self):
        """Initialise pipeline components and default state.

        Creates helper instances for preprocessing, training and persistence
        and initialises the loaded model and performance state to ``None``.
        """
        self.pickler = Pickler(file_path=LR_MODEL_PATH)
        self.identification_resolver = IdentificationResolver()
        self.observation_builder = ObservationBuilder()
        self.feature_aggregator = FeatureAggregator()
        self.vectorizer = FeatureVectorizer()
        self.trainer = Trainer(iterations=30)
        self.model = None
        self.performance = None

    def load_model(self, force_retrain: bool = False, data_path: str = None) -> bool:
        """Load a saved model or retrain it when requested.

        If a ``data_path`` is provided the model is loaded from that path.
        Otherwise the default persisted model path is used via ``Pickler``.
        When ``force_retrain`` is ``True`` and no ``data_path`` is provided,
        the model is trained from the default training data.

        Args:
            force_retrain (bool): Whether to retrain instead of loading a
                persisted model. Defaults to ``False``.
            data_path (str | None): Optional path to a saved model file.

        Returns:
            bool: ``True`` if a model was loaded successfully or retraining
                completed, otherwise ``False``.
        """
        if not force_retrain or data_path is not None:
            loaded = self.pickler.load(data_path)

            if loaded is not None:
                self.model = loaded['model']
                self.performance = loaded['training_result']
                return True
            else:
                return False

        return self._train_model()

    def get_metrics(self):
        """Return the performance metrics for the currently loaded model.

        Returns:
            EvaluationResult | None: The stored performance result if a model
                has been loaded or trained, otherwise ``None``.
        """
        if self.model is None:
            return None

        return self.performance

    def test_csv(self, data_path: str = TESTING_DATA_PATH):
        """Run inference on a CSV file and return structured predictions.

        The CSV is preprocessed into user entities, transformed into feature
        vectors, classified by the loaded model and converted into
        ``Prediction`` objects.

        Args:
            data_path (str): Path to the CSV file to evaluate. Defaults to
                the configured testing data path.

        Returns:
            List[Prediction]: Predictions containing each user entity, its
                predicted maliciousness label and probability score.
        """
        if self.model is None:
            self.load_model()

        user_entities = self._preprocessing(data_path)
        X = np.array([entity.vector for entity in user_entities])

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        formal_predictions = []

        for entity, pred, proba in zip(user_entities, predictions, probabilities):
            formal_predictions.append(Prediction(
                user_entity=entity,
                anomaly_score=float(proba[1]),
                is_malicious=bool(pred),
            ))

        return formal_predictions

    def _preprocessing(self, data_path: str):
        """Convert a CSV file into fully prepared user entities.

        Args:
            data_path (str): Path to the CSV file to preprocess.

        Returns:
            List[UserEntity]: Preprocessed user entities ready for model
                training or inference.
        """
        observations = self.observation_builder.retrieveObservations(data_path)
        user_entities = self.identification_resolver.group_observations(observations)

        for entity in user_entities:
            entity.aggregated_features = self.feature_aggregator.aggregate(entity)
            entity.vector = self.vectorizer.vectorize(entity.aggregated_features)

        return user_entities

    def _train_model(
        self,
        train_path: str = TRAINING_DATA_PATH,
        test_path: str = TESTING_DATA_PATH,
        max_iter_range: tuple = None,
        class_weight_options: list = None,
        random_state_range: tuple = None,
    ) -> bool:
        """Train the model using preprocessed entities and return success.

        Args:
            train_path (str): Path to the training CSV file. Defaults to the
                configured training data path.
            test_path (str): Path to the testing CSV file. Defaults to the
                configured testing data path.
            max_iter_range (tuple | None): Optional range for the maximum
                solver iteration count to pass through to the trainer.
            class_weight_options (list | None): Optional list of class
                weighting strategies to pass through to the trainer.
            random_state_range (tuple | None): Optional range for the random
                seed to pass through to the trainer.

        Returns:
            bool: ``True`` if training produced a model, otherwise ``False``.
        """
        train_entities = self._preprocessing(train_path)
        test_entities = self._preprocessing(test_path)

        X_train = np.array([entity.vector for entity in train_entities])
        y_train = np.array([entity.isMalicious for entity in train_entities])
        X_test = np.array([entity.vector for entity in test_entities])
        y_test = np.array([entity.isMalicious for entity in test_entities])

        kwargs = {}
        if max_iter_range is not None:
            kwargs['max_iter_range'] = max_iter_range
        if class_weight_options is not None:
            kwargs['class_weight_options'] = class_weight_options
        if random_state_range is not None:
            kwargs['random_state_range'] = random_state_range

        self.performance = self.trainer.train(X_train, y_train, X_test, y_test, **kwargs)
        self.model = self.trainer.get_model()

        if self.model is None:
            return False

        return True
