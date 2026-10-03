import os
import json
import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)


class ModelLoadingError(Exception):
    """Raised when an ML model cannot be loaded due to corruption or missing file."""
    pass


class ModelIncompatibilityError(Exception):
    """Raised when an ML model fails version compatibility checks."""
    pass


class ModelService:
    """Thread-safe singleton model registry and loader with caching and error isolation."""
    _instance = None
    _models_cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._models_cache = {}
        return cls._instance

    @classmethod
    def get_base_dir(cls) -> Path:
        return Path(settings.BASE_DIR)

    @classmethod
    def get_available_models(cls) -> dict:
        """Returns catalog of model files, versions, and physical existence."""
        base_dir = cls.get_base_dir()
        prod_dir = base_dir / 'models/production'
        exp_dir = base_dir / 'models/experimental'
        legacy_dir = base_dir / 'model'

        return {
            'rfc_production': {
                'name': 'Random Forest Classifier Pipeline (Production)',
                'version': '1.0.0',
                'status': 'PRODUCTION',
                'filename': 'random_forest_pipeline_v1.joblib',
                'path': prod_dir / 'random_forest_pipeline_v1.joblib',
                'metadata_path': prod_dir / 'metadata.json',
                'exists': (prod_dir / 'random_forest_pipeline_v1.joblib').exists(),
                'type': 'pipeline',
                'default_threshold': 0.23
            },
            'mlp_experimental': {
                'name': 'Multi-Layer Perceptron (DNN Experimental)',
                'version': '1.0.0',
                'status': 'EXPERIMENTAL',
                'filename': 'mlp_neural_network_pipeline_v1.joblib',
                'path': exp_dir / 'mlp_neural_network_pipeline_v1.joblib',
                'metadata_path': exp_dir / 'mlp_neural_network_metadata.json',
                'exists': (exp_dir / 'mlp_neural_network_pipeline_v1.joblib').exists(),
                'type': 'pipeline',
                'default_threshold': 0.45
            },
            'autoencoder_experimental': {
                'name': 'Autoencoder / Anomaly Detector (Experimental)',
                'version': '1.0.0',
                'status': 'EXPERIMENTAL_REQUIRES_RETRAINING',
                'filename': 'autoencoder_anomaly_v1.joblib',
                'path': exp_dir / 'autoencoder_anomaly_v1.joblib',
                'metadata_path': exp_dir / 'autoencoder_anomaly_metadata.json',
                'exists': (exp_dir / 'autoencoder_anomaly_v1.joblib').exists(),
                'type': 'anomaly',
                'default_threshold': 70.0
            },
            'baseline_lr': {
                'name': 'Baseline Logistic Regression',
                'version': '1.0.0',
                'status': 'EXPERIMENTAL',
                'filename': 'baseline_lr_pipeline_v1.joblib',
                'path': exp_dir / 'baseline_lr_pipeline_v1.joblib',
                'metadata_path': exp_dir / 'baseline_lr_metadata.json',
                'exists': (exp_dir / 'baseline_lr_pipeline_v1.joblib').exists(),
                'type': 'pipeline',
                'default_threshold': 0.94
            },
            # Legacy pointers preserved for backward compatibility
            'rfc': {
                'name': 'Legacy Random Forest Classifier',
                'version': '0.1.0-legacy',
                'status': 'ARCHIVED',
                'filename': 'rfc_model.pkl',
                'path': legacy_dir / 'rfc_model.pkl',
                'exists': (legacy_dir / 'rfc_model.pkl').exists(),
                'type': 'sklearn',
                'default_threshold': 0.50
            },
            'dnn': {
                'name': 'Legacy Deep Neural Network',
                'version': '0.1.0-legacy',
                'status': 'ARCHIVED',
                'filename': 'dnn_model.h5',
                'path': legacy_dir / 'dnn_model.h5',
                'exists': (legacy_dir / 'dnn_model.h5').exists(),
                'type': 'keras',
                'default_threshold': 0.50
            }
        }

    @classmethod
    def get_model_metadata(cls, model_key: str) -> dict:
        """Retrieves serialized JSON metadata for the specified model key."""
        catalog = cls.get_available_models()
        if model_key not in catalog:
            return {}
        info = catalog[model_key]
        meta_path = info.get('metadata_path')
        if meta_path and meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not read metadata for {model_key}: {e}")
        return {
            'model_name': info['name'],
            'model_version': info['version'],
            'status': info['status'],
            'default_threshold': info.get('default_threshold', 0.5)
        }

    @classmethod
    def load_model(cls, model_key: str = 'rfc_production'):
        """Loads and caches a model artifact with safety guards."""
        if model_key in cls._models_cache:
            return cls._models_cache[model_key]

        catalog = cls.get_available_models()
        if model_key not in catalog:
            # Fallback alias mappings
            if model_key == 'rfc':
                if catalog['rfc_production']['exists']:
                    return cls.load_model('rfc_production')
            elif model_key == 'dnn':
                if catalog['mlp_experimental']['exists']:
                    return cls.load_model('mlp_experimental')
            raise ModelLoadingError(f"Requested model '{model_key}' is not in the system catalog.")

        info = catalog[model_key]
        model_path = info['path']

        if not model_path.exists():
            # If requesting legacy rfc but production exists, transparently use production
            if model_key == 'rfc' and catalog['rfc_production']['exists']:
                return cls.load_model('rfc_production')
            raise ModelLoadingError(f"Model artifact '{info['name']}' not found on storage ({info['filename']}).")

        try:
            if info['type'] in ('pipeline', 'sklearn', 'anomaly', 'ensemble'):
                import joblib
                model = joblib.load(model_path)
                cls._models_cache[model_key] = model
                logger.info(f"Successfully loaded and cached model: {info['name']}")
                return model
            elif info['type'] == 'keras':
                from tensorflow.keras.models import load_model
                model = load_model(str(model_path))
                cls._models_cache[model_key] = model
                logger.info(f"Successfully loaded Keras model: {info['name']}")
                return model
        except Exception as exc:
            logger.error(f"Failed loading model {model_key}: {exc}")
            raise ModelIncompatibilityError(
                f"Model '{info['name']}' could not be deserialized. "
                f"Artifact format requires migration or retraining."
            ) from exc

    @classmethod
    def clear_cache(cls):
        """Clears cached models from memory."""
        cls._models_cache.clear()
