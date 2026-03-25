"""
Model training pipeline with:
  - XGBoost classifier
  - Stratified K-Fold cross-validation
  - Regularization (L1/L2, max_depth, subsampling)
  - Early stopping on validation loss
  - Model + metadata persistence
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from config.settings import settings
from utils.feature_engine import FeatureEngine, FEATURE_COLUMNS
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Trains and evaluates an XGBoost Buy/Sell classifier."""

    def __init__(self, feature_engine: FeatureEngine = None):
        self.feature_engine = feature_engine or FeatureEngine()
        self.model = None
        self.cv_results = {}

    def get_model_params(self) -> Dict:
        """
        XGBoost hyperparameters tuned for overfitting prevention.

        Regularization strategies:
          1. reg_alpha (L1) — encourages sparsity in feature weights
          2. reg_lambda (L2) — penalizes large weights
          3. max_depth=4 — limits tree complexity
          4. subsample=0.8 — trains each tree on 80% of rows
          5. colsample_bytree=0.8 — uses 80% of features per tree
          6. min_child_weight=5 — requires minimum samples per leaf
          7. early_stopping_rounds — stops when val loss plateaus
        """
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,       # L1 regularization
            "reg_lambda": 1.0,      # L2 regularization
            "gamma": 0.1,           # Min loss reduction for split
            "random_state": 42,
            "use_label_encoder": False,
            "verbosity": 0,
        }

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = None,
    ) -> Dict[str, float]:
        """
        Run stratified K-fold cross-validation.

        Stratification ensures each fold preserves the Buy/Sell class ratio,
        which is critical for financial data where classes may be imbalanced.

        Returns dict of mean scores across folds.
        """
        n_folds = n_folds or settings.cv_folds
        params = self.get_model_params()

        logger.info(
            f"Starting {n_folds}-fold stratified cross-validation "
            f"on {len(X)} samples, {len(X.columns)} features"
        )

        model = xgb.XGBClassifier(**params)

        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=42,
        )

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }

        cv_results = cross_validate(
            model, X, y,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        results = {}
        for metric in ["accuracy", "precision", "recall", "f1"]:
            test_key = f"test_{metric}"
            train_key = f"train_{metric}"
            results[f"cv_{metric}_mean"] = float(np.mean(cv_results[test_key]))
            results[f"cv_{metric}_std"] = float(np.std(cv_results[test_key]))
            results[f"train_{metric}_mean"] = float(np.mean(cv_results[train_key]))

        self.cv_results = results

        logger.info("Cross-Validation Results:")
        logger.info(f"  Accuracy:  {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
        logger.info(f"  Precision: {results['cv_precision_mean']:.4f} ± {results['cv_precision_std']:.4f}")
        logger.info(f"  Recall:    {results['cv_recall_mean']:.4f} ± {results['cv_recall_std']:.4f}")
        logger.info(f"  F1:        {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")

        # Check for overfitting: if train >> test, warn
        overfit_gap = results["train_accuracy_mean"] - results["cv_accuracy_mean"]
        if overfit_gap > 0.10:
            logger.warning(
                f"Potential overfitting detected! "
                f"Train accuracy ({results['train_accuracy_mean']:.4f}) is "
                f"{overfit_gap:.4f} higher than CV accuracy ({results['cv_accuracy_mean']:.4f})"
            )

        return results

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> xgb.XGBClassifier:
        """
        Train the final model on all available data.
        Uses early stopping with a held-out validation set (last 20% of data).
        """
        params = self.get_model_params()

        # Split last 20% as validation for early stopping
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            f"Training final model: {len(X_train)} train / {len(X_val)} validation samples"
        )

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        val_acc = accuracy_score(y_val, y_pred)
        val_f1 = f1_score(y_val, y_pred)

        logger.info(f"Validation accuracy: {val_acc:.4f}")
        logger.info(f"Validation F1:       {val_f1:.4f}")
        logger.info(f"\nClassification Report:\n{classification_report(y_val, y_pred, target_names=['Sell', 'Buy'])}")

        return self.model

    def save_model(self, model_path: str = None, metadata_path: str = None):
        """Save trained model and metadata to disk."""
        model_path = model_path or settings.model_path
        metadata_path = metadata_path or settings.metadata_path

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save XGBoost model in JSON format (portable)
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        metadata = {
            "model_version": settings.model_version,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "features": FEATURE_COLUMNS,
            "n_features": len(FEATURE_COLUMNS),
            "cv_results": self.cv_results,
            "hyperparameters": self.get_model_params(),
            "regularization": {
                "l1_alpha": self.get_model_params()["reg_alpha"],
                "l2_lambda": self.get_model_params()["reg_lambda"],
                "max_depth": self.get_model_params()["max_depth"],
                "subsample": self.get_model_params()["subsample"],
                "colsample_bytree": self.get_model_params()["colsample_bytree"],
                "min_child_weight": self.get_model_params()["min_child_weight"],
                "early_stopping": True,
            },
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    @staticmethod
    def load_model(model_path: str = None) -> xgb.XGBClassifier:
        """Load a trained model from disk."""
        model_path = model_path or settings.model_path
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
