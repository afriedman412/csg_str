# app/models/model_loader.py
from app.core.config import PIPELINE_PATH
import joblib
from app.model_api.training_structure import STRRevenuePipeline, LightGBMRegressorCV


def load_model():
    """Load the most recent trained model from disk."""
    import sys
    sys.modules["__main__"].STRRevenuePipeline = STRRevenuePipeline
    sys.modules["__main__"].LightGBMRegressorCV = LightGBMRegressorCV
    return joblib.load(PIPELINE_PATH)
