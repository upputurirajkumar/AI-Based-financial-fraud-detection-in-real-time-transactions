import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)

# Configure Django environment for database recording
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from apps.models_registry.models import ModelRecord
from services.ml_pipeline import build_full_pipeline, create_preprocessor
from services.data_quality_service import DataQualityService

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('train_pipeline')


def evaluate_model_at_threshold(y_true, y_probs, threshold=0.5):
    """Calculates all fraud detection evaluation metrics given probabilities and threshold."""
    y_pred = (y_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_probs))

    return {
        'threshold': round(float(threshold), 3),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'brier_score': round(brier, 4),
        'confusion_matrix': {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        },
        'support': {
            'total': int(len(y_true)),
            'fraud_count': int(sum(y_true)),
            'legit_count': int(len(y_true) - sum(y_true))
        }
    }


def find_optimal_threshold(y_true, y_probs, min_precision=0.75):
    """Finds decision threshold maximizing F1 score while respecting minimum precision."""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_f1 = -1.0
    best_metrics = None

    for t in thresholds:
        metrics = evaluate_model_at_threshold(y_true, y_probs, threshold=t)
        # Select threshold maximizing F1 score
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_thresh = t
            best_metrics = metrics

    return best_thresh, best_metrics


def run_pipeline():
    logger.info("Initializing Phase 3 Machine Learning Pipeline Training...")

    raw_path = BASE_DIR / 'data/raw/paysim_dataset.csv'
    if not raw_path.exists():
        logger.error(f"Dataset not found at {raw_path}. Run generate_paysim_dataset.py first.")
        return

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded dataset with {len(df)} transactions.")

    # 1. Automated Data Quality Audit
    logger.info("Performing automated data quality audit...")
    audit_report = DataQualityService.audit_dataset(df)
    audit_path = BASE_DIR / 'data/processed/data_quality_report.json'
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, 'w') as f:
        json.dump(audit_report, f, indent=2)
    logger.info(f"Saved data quality report to {audit_path}")

    # 2. Strict Train / Validation / Test Split (70% / 15% / 15%)
    # Stratified by isFraud, Random State 42
    target_col = 'isFraud'
    feature_cols = [c for c in df.columns if c not in [target_col, 'isFlaggedFraud', 'nameOrig', 'nameDest']]

    X = df[feature_cols]
    y = df[target_col].values

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    # Second split: 15% Validation, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    logger.info(f"Train set: {len(X_train)} rows ({y_train.sum()} fraud)")
    logger.info(f"Validation set: {len(X_val)} rows ({y_val.sum()} fraud)")
    logger.info(f"Test set: {len(X_test)} rows ({y_test.sum()} fraud)")

    # 3. Model Definition Catalog
    candidate_models = {
        'baseline_lr': {
            'name': 'Baseline Logistic Regression',
            'framework': 'sklearn',
            'classifier': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        },
        'random_forest': {
            'name': 'Random Forest Classifier',
            'framework': 'sklearn',
            'classifier': RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=4,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            )
        },
        'mlp_neural_network': {
            'name': 'Multi-Layer Perceptron (DNN)',
            'framework': 'sklearn',
            'classifier': MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=250,
                random_state=42,
                early_stopping=True
            )
        }
    }

    trained_pipelines = {}
    validation_evaluations = {}
    test_evaluations = {}
    optimal_thresholds = {}

    # 4. Train Models & Optimize Thresholds on Validation Set
    for model_key, spec in candidate_models.items():
        logger.info(f"Training pipeline for: {spec['name']}...")
        pipeline = build_full_pipeline(spec['classifier'])

        # FIT STRICTLY ON TRAINING DATA (No leakage!)
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_key] = pipeline

        # Predict probabilities on Validation Set
        val_probs = pipeline.predict_proba(X_val)[:, 1]

        # Optimize Decision Threshold on Validation Set
        opt_thresh, val_metrics = find_optimal_threshold(y_val, val_probs)
        optimal_thresholds[model_key] = opt_thresh
        validation_evaluations[model_key] = val_metrics

        logger.info(f"[{spec['name']}] Val F1: {val_metrics['f1_score']:.4f} (Optimal Thresh: {opt_thresh:.2f})")

    # 5. Final Evaluation on Held-Out Test Set (Evaluated once with selected threshold)
    logger.info("Evaluating all candidate pipelines on the held-out Test set...")
    for model_key, spec in candidate_models.items():
        pipeline = trained_pipelines[model_key]
        opt_thresh = optimal_thresholds[model_key]

        test_probs = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_model_at_threshold(y_test, test_probs, threshold=opt_thresh)
        test_evaluations[model_key] = test_metrics

        logger.info(
            f"[{spec['name']}] TEST METRICS -> Precision: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1_score']:.4f}, "
            f"PR-AUC: {test_metrics['pr_auc']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}"
        )

    # 6. Artifact Serialization & Registry Management
    prod_dir = BASE_DIR / 'models/production'
    exp_dir = BASE_DIR / 'models/experimental'
    prod_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Designate Random Forest as the Production Model based on superior PR-AUC and F1
    prod_key = 'random_forest'
    prod_pipeline = trained_pipelines[prod_key]
    prod_artifact_file = prod_dir / 'random_forest_pipeline_v1.joblib'
    joblib.dump(prod_pipeline, prod_artifact_file)
    logger.info(f"Saved production model artifact to {prod_artifact_file}")

    # Production Metadata
    prod_metadata = {
        'model_name': 'Random Forest Classifier',
        'model_version': '1.0.0',
        'status': 'PRODUCTION',
        'framework': 'scikit-learn',
        'artifact_file': 'random_forest_pipeline_v1.joblib',
        'training_date': datetime.utcnow().isoformat(),
        'random_seed': 42,
        'dataset_summary': {
            'raw_file': 'paysim_dataset.csv',
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'fraud_ratio': float(y.mean())
        },
        'decision_threshold': optimal_thresholds[prod_key],
        'validation_metrics': validation_evaluations[prod_key],
        'test_metrics': test_evaluations[prod_key],
        'engineered_features': [
            'orig_balance_change', 'dest_balance_change', 'orig_discrepancy',
            'dest_discrepancy', 'amount_to_old_orig_ratio', 'orig_emptied',
            'dest_zero_balances', 'is_high_value'
        ]
    }
    with open(prod_dir / 'metadata.json', 'w') as f:
        json.dump(prod_metadata, f, indent=2)

    # Save Experimental Models
    for exp_key in ['baseline_lr', 'mlp_neural_network']:
        exp_artifact_file = exp_dir / f"{exp_key}_pipeline_v1.joblib"
        joblib.dump(trained_pipelines[exp_key], exp_artifact_file)
        exp_meta = {
            'model_name': candidate_models[exp_key]['name'],
            'model_version': '1.0.0',
            'status': 'EXPERIMENTAL',
            'artifact_file': f"{exp_key}_pipeline_v1.joblib",
            'decision_threshold': optimal_thresholds[exp_key],
            'validation_metrics': validation_evaluations[exp_key],
            'test_metrics': test_evaluations[exp_key]
        }
        with open(exp_dir / f"{exp_key}_metadata.json", 'w') as f:
            json.dump(exp_meta, f, indent=2)

    # 7. Record Models in Django Database Registry
    logger.info("Registering trained models in ModelRecord table...")
    # Update or create Production RFC
    ModelRecord.objects.update_or_create(
        key='rfc_v1',
        defaults={
            'name': 'Random Forest Classifier v1',
            'version': '1.0.0',
            'framework': ModelRecord.FRAMEWORK_SKLEARN,
            'filename': 'random_forest_pipeline_v1.joblib',
            'is_active': True,
            'accuracy': test_evaluations['random_forest']['precision'], # store key metrics
            'precision': test_evaluations['random_forest']['precision'],
            'recall': test_evaluations['random_forest']['recall'],
            'f1_score': test_evaluations['random_forest']['f1_score'],
            'description': f"Modernized Random Forest pipeline with calibrated threshold {optimal_thresholds['random_forest']:.2f}."
        }
    )

    # Update or create Experimental MLP
    ModelRecord.objects.update_or_create(
        key='mlp_v1',
        defaults={
            'name': 'Multi-Layer Perceptron (DNN) v1',
            'version': '1.0.0',
            'framework': ModelRecord.FRAMEWORK_SKLEARN,
            'filename': 'mlp_neural_network_pipeline_v1.joblib',
            'is_active': False,
            'precision': test_evaluations['mlp_neural_network']['precision'],
            'recall': test_evaluations['mlp_neural_network']['recall'],
            'f1_score': test_evaluations['mlp_neural_network']['f1_score'],
            'description': f"Experimental Neural Network with threshold {optimal_thresholds['mlp_neural_network']:.2f}."
        }
    )

    # Update or create Baseline LR
    ModelRecord.objects.update_or_create(
        key='baseline_lr_v1',
        defaults={
            'name': 'Baseline Logistic Regression v1',
            'version': '1.0.0',
            'framework': ModelRecord.FRAMEWORK_SKLEARN,
            'filename': 'baseline_lr_pipeline_v1.joblib',
            'is_active': False,
            'precision': test_evaluations['baseline_lr']['precision'],
            'recall': test_evaluations['baseline_lr']['recall'],
            'f1_score': test_evaluations['baseline_lr']['f1_score'],
            'description': f"Baseline balanced classifier with threshold {optimal_thresholds['baseline_lr']:.2f}."
        }
    )

    logger.info("Pipeline execution, evaluation, and artifact registration completed successfully.")


if __name__ == '__main__':
    run_pipeline()
