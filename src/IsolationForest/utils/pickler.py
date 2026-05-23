import pickle
import os

class Pickler:
    """Persist and restore Python objects using pickle files."""

    def __init__(self, file_path):
        """Initialize the pickler and ensure its directory exists.

        Args:
            file_path (str): Path to the pickle file used for persistence.
        """
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def save(self, obj, training_result):
        """Serialize a model and its training result to disk.

        Args:
            obj: Model object to persist.
            training_result: Training result associated with the model.
        """
        payload = {
            'model': obj,
            'training_result': training_result
        }   

        with open(self.file_path, 'wb') as f:
            pickle.dump(payload, f)

    def load(self, file_path=None):
        """Load a previously persisted object payload from disk.

        Args:
            file_path (str | None): Optional path to a pickle file. If not
                provided, the instance's default path is used.

        Returns:
            object | None: Loaded payload dictionary, or `None` if the file
                does not exist.
        """
        if file_path is None:
            file_path = self.file_path

        if not os.path.exists(file_path):
            return None

        with open(file_path, 'rb') as f:
            return pickle.load(f)