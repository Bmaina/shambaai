"""ShambaAI — Predictive crop disease intelligence for East African smallholders."""
from shambaai.pipeline.inference import ShambaAIPipeline, FarmPlot
from shambaai.models.health_scorer import FarmHealthScorer, FarmRisk
from shambaai.models.disease_cnn import CropDiseaseNet, DISEASE_CLASSES
from shambaai.models.edsr import EDSR, SpectralFidelityLoss, walds_protocol_pair

__version__ = "1.0.0"
__all__ = [
    "ShambaAIPipeline", "FarmPlot",
    "FarmHealthScorer", "FarmRisk",
    "CropDiseaseNet", "DISEASE_CLASSES",
    "EDSR", "SpectralFidelityLoss", "walds_protocol_pair",
]
