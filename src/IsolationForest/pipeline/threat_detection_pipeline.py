import numpy as np
from src.IsolationForest.preprocessing.identification_resolver import IdentificationResolver
from src.IsolationForest.preprocessing.observation_builder import ObservationBuilder
from src.IsolationForest.preprocessing.feature_aggregator import FeatureAggregator
from src.IsolationForest.preprocessing.vectorizer import FeatureVectorizer
from src.IsolationForest.models.prediction import Prediction
from src.IsolationForest.training.trainer import Trainer
from src.IsolationForest.utils.pickler import Pickler
from src.IsolationForest.config.file_paths import BEST_MODEL_PATH, TRAINING_DATA_PATH, TESTING_DATA_PATH

class ThreatDetectionPipeline:
    """Orchestrates preprocessing, training, model loading and testing.

    The pipeline wires together the preprocessing helpers, trainer and model
    persistence utilities required to train and evaluate the threat
    detection system.
    """

    def __init__(self):
        """Initialize pipeline components and default state.

        Creates helper instances for preprocessing, training and persistence
        and initialises the loaded model/performance state to `None`.
        """
        self.pickler = Pickler(file_path=BEST_MODEL_PATH)
        self.identification_resolver = IdentificationResolver()
        self.observation_builder = ObservationBuilder()
        self.feature_aggregator = FeatureAggregator()
        self.vectorizer = FeatureVectorizer()
        self.trainer = Trainer(iterations=100)
        self.model = None
        self.performance = None

    def load_model(self, force_retrain=False, data_path=None):
        """Load a saved model or retrain it when requested.

        If a `data_path` is provided, the model is loaded from that path.
        Otherwise the default persisted model path is used via `Pickler`.
        When `force_retrain` is `True` and no `data_path` is provided, the
        model is trained from the default training data.

        Args:
            force_retrain (bool): Whether to retrain instead of loading a
                persisted model. Defaults to `False`.
            data_path (str | None): Optional path to a saved model file.

        Returns:
            bool: `True` if a model was loaded successfully or retraining
                completed, otherwise `False`.
        """
        if not force_retrain or not data_path is None:
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
            EvaluationResult | None: The stored performance result if a
                model has been loaded or trained, otherwise `None`.
        """
        if self.model is None:
            return None

        return self.performance

    def test_csv(self, data_path=TESTING_DATA_PATH):
        """Run inference on a CSV file and return structured predictions.

        The CSV is preprocessed into user entities, transformed into vectors,
        scored by the loaded model and converted into `Prediction` objects.

        Args:
            data_path (str): Path to the CSV file to evaluate. Defaults to
                the configured testing data path.

        Returns:
            List[Prediction]: Predictions containing each user entity, its
                anomaly score and the binary maliciousness label.
        """
        if self.model is None:
            self.load_model()

        user_entities = self._preprocessing(data_path)
        X = np.array([entity.vector for entity in user_entities])

        scores = np.array([self.model.anomaly_score(x) for x in X])
        
        t = np.percentile(scores, self.performance.threshold)
        predictions = np.where(scores < t, 0, 1)

        formal_predictions = []

        for entity, score, pred in zip(user_entities, scores, predictions):
            formal_predictions.append(Prediction(
                user_entity=entity,
                anomaly_score=score,
                is_malicious=bool(pred)
            ))

        return formal_predictions

    def _preprocessing(self, data_path):
        """Convert a CSV file into fully prepared user entities.

        This method retrieves observations, groups them into `UserEntity`
        objects, aggregates features and computes vector representations for
        each entity.

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

    def _train_model(self, data_path=TRAINING_DATA_PATH, n_trees=None, sampling_size=None, threshold=None):
        """Train the model using preprocessed entities and return success.

        Args:
            data_path (str): Path to the training CSV file. Defaults to the
                configured training data path.
            n_trees (tuple | None): Optional range/tuple of tree counts to
                pass through to the trainer.
            sampling_size (tuple | None): Optional range/tuple of sampling
                sizes to pass through to the trainer.
            threshold (tuple | None): Optional range/tuple of threshold
                values to pass through to the trainer.

        Returns:
            bool: `True` if training produced a model, otherwise `False`.
        """
        user_entities = self._preprocessing(data_path)

        X = np.array([entity.vector for entity in user_entities])
        X_actual = np.array([entity.isMalicious for entity in user_entities])

        if n_trees is not None and sampling_size is not None and threshold is not None:
            self.performance = self.trainer.train(X, X_actual, n_trees_range=n_trees, sampling_size_range=sampling_size, threshold_range=threshold)
        else:
            self.performance = self.trainer.train(X, X_actual)
        
        self.model = self.trainer.get_model()

        if self.model is None:
            return False

        return True






        

        


        
    
