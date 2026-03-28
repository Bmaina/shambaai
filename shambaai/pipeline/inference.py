"""
ShambaAI Inference Pipeline — Three-Layer Defense™ integration.

This module is the production entry point. It wires together:
  Layer 1: Sentinel-1/2 feature extraction → FarmHealthScorer
  Layer 2: NDVI anomaly detection → pre-emptive WhatsApp alert
  Layer 3: CropDiseaseNet → on-demand photo diagnosis

Also implements:
  - Portfolio batch scoring (B2B API for FINCA Kenya, Pula Advisors)
  - WhatsApp message formatting (Twilio/WhatsApp Business API)
  - County-level heatmap generation (B2B intelligence product)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional
import json
import os

from shambaai.models.health_scorer import FarmHealthScorer, FarmRisk, SentinelFeatureExtractor
from shambaai.models.disease_cnn   import CropDiseaseNet, DISEASE_CLASSES, TREATMENT_MAP


# ── Farm Plot Data Model ──────────────────────────────────────────────────────

@dataclass
class FarmPlot:
    """Represents a single smallholder farm plot."""
    plot_id:       str
    farmer_name:   str
    phone_number:  str          # E.164 format: +254712345678
    crop:          str          # primary crop
    county:        str          # Kenya county
    latitude:      float
    longitude:     float
    area_ha:       float
    # Latest Sentinel-2 band values (reflectance, 0–1 scale)
    B2: float = 0.08; B3: float = 0.12; B4: float = 0.08
    B5: float = 0.18; B6: float = 0.22; B7: float = 0.26
    B8: float = 0.42; B8A: float = 0.44; B11: float = 0.20; B12: float = 0.14
    # Latest Sentinel-1 SAR values (dB)
    VV: float = -12.0; VH: float = -18.0
    # Temporal context
    ndvi_prev: float = 0.50
    ndvi_3mo_mean: float = 0.52
    days_since_rain: int = 7

    def sentinel_features(self) -> dict:
        return {
            "B2": self.B2, "B3": self.B3, "B4": self.B4,
            "B5": self.B5, "B6": self.B6, "B7": self.B7,
            "B8": self.B8, "B8A": self.B8A, "B11": self.B11, "B12": self.B12,
            "VV": self.VV, "VH": self.VH,
            "ndvi_prev": self.ndvi_prev,
            "ndvi_3mo_mean": self.ndvi_3mo_mean,
            "plot_area_ha": self.area_ha,
            "days_since_rain": self.days_since_rain,
        }


# ── Alert Message Formatter ───────────────────────────────────────────────────

class WhatsAppFormatter:
    """
    Formats ShambaAI alerts for WhatsApp Business API delivery.
    Messages are bilingual (English + Swahili) and actionable.
    """

    @staticmethod
    def satellite_alert(plot: FarmPlot, risk: FarmRisk) -> dict:
        """Pre-emptive satellite-triggered alert (no photo required)."""
        ndvi_computed = (plot.B8 - plot.B4) / (plot.B8 + plot.B4 + 1e-8)
        re_ndvi       = (plot.B8 - plot.B5) / (plot.B8 + plot.B5 + 1e-8)

        body = (
            f"🛰 *ShambaAI Satellite Alert*\n"
            f"Farm: {plot.plot_id} | {plot.county}\n"
            f"Crop: {plot.crop.title()}\n\n"
            f"*Farm Health Score: {risk.score:.0f}/100 — {risk.tier}*\n\n"
            f"NDVI: {ndvi_computed:.3f} (prev: {plot.ndvi_prev:.3f})\n"
            f"Red-Edge NDVI: {re_ndvi:.3f}\n"
            f"Days since rain: {plot.days_since_rain}\n\n"
            f"📋 *Action:* {risk.action}\n\n"
            f"_{risk.swahili_msg}_\n\n"
            f"📸 *Reply with a photo of your crop for instant disease diagnosis.*\n"
            f"💊 *Reply DAWA for nearest agro-dealer location.*"
        )
        return {
            "to": plot.phone_number,
            "body": body,
            "type": "satellite_alert",
            "risk_tier": risk.tier,
            "plot_id": plot.plot_id,
        }

    @staticmethod
    def disease_diagnosis(
        plot: FarmPlot,
        predictions: list[dict],
        risk: FarmRisk,
    ) -> dict:
        """Photo-triggered disease diagnosis response."""
        if not predictions:
            body = (
                f"📸 *ShambaAI Diagnosis*\n"
                f"Farm: {plot.plot_id}\n\n"
                f"✅ No disease detected with high confidence.\n"
                f"Your {plot.crop} looks healthy from this photo.\n\n"
                f"🛰 *Satellite score: {risk.score:.0f}/100 — {risk.tier}*\n"
                f"Continue monitoring. Next satellite check: 6 days."
            )
        else:
            top = predictions[0]
            treatment = top.get("treatment", {})
            body = (
                f"📸 *ShambaAI Diagnosis*\n"
                f"Farm: {plot.plot_id} | {plot.crop.title()}\n\n"
                f"🔬 *Detected:* {top['class'].replace('_', ' ').title()}\n"
                f"   Confidence: {top['confidence']*100:.0f}%\n\n"
                f"💊 *Treatment:*\n"
                f"   {treatment.get('action', 'Consult agro-dealer')}\n"
                f"   Chemical: {treatment.get('chemical', 'N/A')}\n"
                f"   Dose: {treatment.get('dose', 'N/A')}\n"
                f"   Cost: ~KES {treatment.get('cost_kes', '?')}\n\n"
                f"🌍 *Swahili:*\n"
                f"   {treatment.get('swahili', '')}\n\n"
                f"🛰 *Satellite score: {risk.score:.0f}/100 — {risk.tier}*\n\n"
                f"_Reply DAWA for nearest agro-dealer. Reply HELP for extension officer._"
            )
        return {
            "to": plot.phone_number,
            "body": body,
            "type": "disease_diagnosis",
            "top_disease": predictions[0]["class"] if predictions else "healthy",
            "plot_id": plot.plot_id,
        }

    @staticmethod
    def welcome_message(farmer_name: str, phone: str) -> dict:
        body = (
            f"🌱 *Karibu ShambaAI!* Welcome, {farmer_name}!\n\n"
            f"Your farm is now monitored by satellite every 6 days.\n\n"
            f"*What you get (KES 200/month):*\n"
            f"✅ Satellite alerts 2–3 weeks before disease appears\n"
            f"✅ Instant photo diagnosis (40+ diseases)\n"
            f"✅ Treatment advice in Swahili\n"
            f"✅ Works offline — no internet needed for diagnosis\n\n"
            f"📸 *To diagnose a sick plant:* Send a clear photo now.\n"
            f"🛰 *Satellite alerts:* Automatic every 6 days.\n\n"
            f"_Tunawasiliana! We are connected._"
        )
        return {"to": phone, "body": body, "type": "welcome"}


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class ShambaAIPipeline:
    """
    Production inference pipeline integrating all three ShambaAI layers.

    Usage:
        pipeline = ShambaAIPipeline()
        pipeline.load_models()

        # Satellite monitoring pass (runs nightly via GEE)
        risk = pipeline.score_plot(farm_plot)

        # Photo diagnosis (triggered by farmer WhatsApp message)
        results = pipeline.diagnose_image("path/to/photo.jpg")
    """

    def __init__(self, model_dir: str = "models/"):
        self.model_dir     = model_dir
        self.health_scorer = FarmHealthScorer()
        self.disease_model: Optional[CropDiseaseNet] = None
        self.formatter     = WhatsAppFormatter()
        self._scorer_ready = False

    def load_models(self, train_if_missing: bool = True) -> "ShambaAIPipeline":
        """Load or train models."""
        scorer_path = os.path.join(self.model_dir, "health_scorer.joblib")

        if os.path.exists(scorer_path):
            self.health_scorer = FarmHealthScorer.load(scorer_path)
            print(f"[Pipeline] FarmHealthScorer loaded from {scorer_path}")
        elif train_if_missing:
            print("[Pipeline] Training FarmHealthScorer on synthetic data...")
            os.makedirs(self.model_dir, exist_ok=True)
            self.health_scorer.fit(verbose=True)
            self.health_scorer.save(scorer_path)
        self._scorer_ready = True

        # Disease CNN (load if checkpoint exists)
        cnn_path = os.path.join(self.model_dir, "disease_cnn.pt")
        self.disease_model = CropDiseaseNet()
        if os.path.exists(cnn_path):
            import torch
            self.disease_model.load_state_dict(
                torch.load(cnn_path, map_location="cpu")
            )
            print(f"[Pipeline] CropDiseaseNet loaded from {cnn_path}")
        else:
            print("[Pipeline] CropDiseaseNet: using ImageNet pretrained (no fine-tune yet)")
            print("  → To fine-tune: run scripts/train_disease_cnn.py")
        self.disease_model.eval()
        return self

    def score_plot(self, plot: FarmPlot) -> FarmRisk:
        """Layer 1+2: Satellite scoring for a single farm plot."""
        assert self._scorer_ready, "Call load_models() first"
        return self.health_scorer.predict(plot.sentinel_features(), crop=plot.crop)

    def score_portfolio(
        self, plots: list[FarmPlot]
    ) -> pd.DataFrame:
        """
        Layer 1+2: Batch portfolio scoring. B2B product for FINCA Kenya,
        Pula Advisors, ACRE Africa.

        Returns DataFrame with one row per plot, including:
          health_score, risk_tier, action, alert_message (EN + SW)
        """
        assert self._scorer_ready, "Call load_models() first"
        records = []
        for plot in plots:
            feats = SentinelFeatureExtractor.compute_features(**plot.sentinel_features())
            records.append(feats)
        feat_df = pd.DataFrame(records)
        risks   = self.health_scorer.predict_batch(feat_df, crop="mixed")

        rows = []
        for plot, risk in zip(plots, risks):
            rows.append({
                "plot_id":      plot.plot_id,
                "farmer":       plot.farmer_name,
                "county":       plot.county,
                "crop":         plot.crop,
                "area_ha":      plot.area_ha,
                "lat":          plot.latitude,
                "lon":          plot.longitude,
                "health_score": round(risk.score, 1),
                "risk_tier":    risk.tier,
                "action":       risk.action,
                "days_to_act":  risk.days_to_act,
                "alert_en":     risk.alert_msg,
                "alert_sw":     risk.swahili_msg,
            })
        df = pd.DataFrame(rows).sort_values("health_score")
        return df

    def diagnose_image(
        self,
        image_path: str,
        plot: Optional[FarmPlot] = None,
    ) -> list[dict]:
        """
        Layer 3: On-device CNN diagnosis from a farm photo.
        Returns top-3 disease predictions with treatment recommendations.
        """
        import torch
        from PIL import Image
        from shambaai.models.disease_cnn import get_transforms

        img = Image.open(image_path).convert("RGB")
        tf  = get_transforms("val")
        x   = tf(img).unsqueeze(0)
        return self.disease_model.predict(x, top_k=3)

    def full_diagnosis(
        self,
        plot: FarmPlot,
        image_path: Optional[str] = None,
    ) -> dict:
        """
        Complete Three-Layer Defense™ response for a single farm.
        Combines satellite score + optional photo diagnosis.
        Returns formatted WhatsApp message dict.
        """
        risk = self.score_plot(plot)

        if image_path:
            predictions = self.diagnose_image(image_path, plot)
            msg = self.formatter.disease_diagnosis(plot, predictions, risk)
        else:
            predictions = []
            msg = self.formatter.satellite_alert(plot, risk)

        return {
            "whatsapp_message": msg,
            "risk": {
                "score": risk.score,
                "tier":  risk.tier,
                "action": risk.action,
                "days_to_act": risk.days_to_act,
            },
            "disease_predictions": predictions,
            "plot_id": plot.plot_id,
        }

    def county_heatmap_data(
        self, plots: list[FarmPlot]
    ) -> dict:
        """
        Generate county-level heatmap data for B2B intelligence dashboard.
        Returns GeoJSON-compatible structure for TerraSignal API.
        """
        df = self.score_portfolio(plots)
        counties = df.groupby("county").agg(
            avg_score=("health_score", "mean"),
            critical_count=("risk_tier", lambda x: (x == "CRITICAL").sum()),
            high_count=("risk_tier", lambda x: (x == "HIGH").sum()),
            total_farms=("plot_id", "count"),
            total_ha=("area_ha", "sum"),
        ).reset_index()

        features = []
        for _, row in counties.iterrows():
            features.append({
                "type": "Feature",
                "properties": {
                    "county":          row["county"],
                    "avg_health_score": round(row["avg_score"], 1),
                    "critical_farms":  int(row["critical_count"]),
                    "high_risk_farms": int(row["high_count"]),
                    "total_farms":     int(row["total_farms"]),
                    "total_ha":        round(row["total_ha"], 1),
                    "risk_level":      (
                        "CRITICAL" if row["avg_score"] < 35
                        else "HIGH" if row["avg_score"] < 55
                        else "MODERATE" if row["avg_score"] < 75
                        else "STABLE"
                    ),
                },
                "geometry": None,  # In production: add county polygon centroid
            })

        return {
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "total_farms": int(df.shape[0]),
                "total_ha": round(df["area_ha"].sum(), 1),
                "avg_health_score": round(df["health_score"].mean(), 1),
                "critical_count": int((df["risk_tier"] == "CRITICAL").sum()),
            }
        }
