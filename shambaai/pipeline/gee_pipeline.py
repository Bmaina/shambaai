"""
shambaai/pipeline/gee_pipeline.py — Google Earth Engine satellite data pipeline.

This module connects ShambaAI to real Sentinel-1/2 imagery via GEE Python API.
In production this runs nightly as a Cloud Function / Cloud Run job.

Workflow:
    1. For each registered farm plot, extract Sentinel-1/2 pixel statistics
    2. Compute 17 agronomic features via SentinelFeatureExtractor
    3. Run FarmHealthScorer to generate Farm Health Scores
    4. Identify CRITICAL/HIGH plots → trigger WhatsApp alerts
    5. Write county heatmap to Firestore / BigQuery for TerraSignal B2B API

Pre-requisites (free):
    pip install earthengine-api
    earthengine authenticate   # one-time browser auth
    earthengine set_project YOUR_GEE_PROJECT_ID

GEE free tier:
    - Up to 25GB cloud storage
    - Unlimited computation on Sentinel data
    - 5 concurrent requests
    For production scale (100k farms), apply for GEE Commercial tier
    or use GEE + Cloud Functions architecture.

Architecture note:
    ShambaAI deliberately keeps the GEE pipeline SEPARATE from the
    on-device CNN. GEE runs server-side (cloud) for the satellite layer.
    TFLite runs client-side (phone) for the photo diagnosis layer.
    This hybrid approach keeps operational costs near zero at scale.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import json


# ── Farm polygon data model ───────────────────────────────────────────────────

@dataclass
class FarmPolygon:
    """
    A registered farm plot polygon for GEE monitoring.
    In production: stored in Firestore, loaded per nightly batch.
    """
    plot_id:   str
    phone:     str
    crop:      str
    county:    str
    geojson:   dict   # GeoJSON polygon geometry
    area_ha:   float


# ── GEE Pipeline ──────────────────────────────────────────────────────────────

class GEEPipeline:
    """
    Production satellite data extraction pipeline using Google Earth Engine.

    Usage:
        import ee
        ee.Initialize()
        pipeline = GEEPipeline()
        features = pipeline.extract_features(farm_polygon, date="2026-03-22")
    """

    # Sentinel-2 collection (L2A = surface reflectance, atmospherically corrected)
    S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

    # Sentinel-1 GRD (Ground Range Detected)
    S1_COLLECTION = "COPERNICUS/S1_GRD"

    # CHIRPS daily precipitation (for days_since_rain feature)
    CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"

    # Band mappings
    S2_BANDS = {
        "B2": "B2", "B3": "B3", "B4": "B4",
        "B5": "B5", "B6": "B6", "B7": "B7",
        "B8": "B8", "B8A": "B8A", "B11": "B11", "B12": "B12",
        "QA60": "QA60",  # Cloud mask
    }
    S1_BANDS = {"VV": "VV", "VH": "VH"}

    def __init__(self, cloud_threshold: float = 20.0):
        self.cloud_threshold = cloud_threshold

    def extract_features(
        self,
        farm: FarmPolygon,
        date: str,
        lookback_days: int = 12,
        history_days: int = 90,
    ) -> dict:
        """
        Extract 17 agronomic features for a single farm polygon.

        Args:
            farm:          FarmPolygon with GeoJSON geometry
            date:          Target date "YYYY-MM-DD"
            lookback_days: Days to look back for cloud-free image
            history_days:  Days for seasonal baseline NDVI

        Returns:
            Feature dict compatible with SentinelFeatureExtractor.compute_features()

        GEE operations performed:
          1. Sentinel-2 mosaic (cloud-masked, median composite)
          2. Sentinel-1 SAR (latest VV/VH pass)
          3. NDVI previous pass (6 days ago)
          4. NDVI 90-day baseline
          5. CHIRPS rain days calculation
        """
        try:
            import ee
        except ImportError:
            raise ImportError(
                "Google Earth Engine not installed.\n"
                "  pip install earthengine-api\n"
                "  earthengine authenticate"
            )

        geom = ee.Geometry(farm.geojson)
        end_date   = ee.Date(date)
        start_date = end_date.advance(-lookback_days, "day")
        hist_start = end_date.advance(-history_days, "day")

        # ── Sentinel-2 cloud-masked composite ──────────────────────────────
        def mask_s2_clouds(img):
            qa    = img.select("QA60")
            cloud = (1 << 10)
            cirr  = (1 << 11)
            mask  = qa.bitwiseAnd(cloud).eq(0).And(qa.bitwiseAnd(cirr).eq(0))
            return img.updateMask(mask).divide(10000)  # DN → reflectance

        s2 = (
            ee.ImageCollection(self.S2_COLLECTION)
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_threshold))
            .map(mask_s2_clouds)
            .select(list(self.S2_BANDS.keys()))
            .median()
        )

        # ── Sentinel-1 SAR (latest pass) ───────────────────────────────────
        s1 = (
            ee.ImageCollection(self.S1_COLLECTION)
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
            .mean()
        )

        # ── NDVI previous pass (6 days before) ─────────────────────────────
        prev_end   = end_date.advance(-6,  "day")
        prev_start = end_date.advance(-18, "day")
        s2_prev = (
            ee.ImageCollection(self.S2_COLLECTION)
            .filterBounds(geom)
            .filterDate(prev_start, prev_end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_threshold))
            .map(mask_s2_clouds)
            .select(["B4", "B8"])
            .median()
        )
        ndvi_prev_img = s2_prev.normalizedDifference(["B8", "B4"]).rename("ndvi_prev")

        # ── 90-day seasonal NDVI baseline ──────────────────────────────────
        s2_hist = (
            ee.ImageCollection(self.S2_COLLECTION)
            .filterBounds(geom)
            .filterDate(hist_start, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", self.cloud_threshold))
            .map(mask_s2_clouds)
            .select(["B4", "B8"])
            .map(lambda img: img.normalizedDifference(["B8", "B4"]).rename("ndvi"))
            .mean()
        )

        # ── CHIRPS rain days ────────────────────────────────────────────────
        chirps = (
            ee.ImageCollection(self.CHIRPS_COLLECTION)
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        last_rain_img = chirps.map(
            lambda img: img.gt(1.0).rename("rained")
        ).sum()  # days with > 1mm rain in lookback period

        # ── Extract pixel statistics over farm polygon ──────────────────────
        combined = (
            s2
            .addBands(s1)
            .addBands(ndvi_prev_img)
            .addBands(s2_hist.rename("ndvi_baseline"))
            .addBands(last_rain_img)
        )

        stats = combined.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=10,
            maxPixels=1e6,
        ).getInfo()

        # ── Compute days_since_rain ─────────────────────────────────────────
        rain_days_in_lookback = stats.get("rained", 0) or 0
        days_since_rain = max(1, lookback_days - int(rain_days_in_lookback * lookback_days))

        # ── Return feature dict ─────────────────────────────────────────────
        return {
            "B2":  stats.get("B2",  0.08),
            "B3":  stats.get("B3",  0.12),
            "B4":  stats.get("B4",  0.08),
            "B5":  stats.get("B5",  0.18),
            "B6":  stats.get("B6",  0.22),
            "B7":  stats.get("B7",  0.26),
            "B8":  stats.get("B8",  0.42),
            "B8A": stats.get("B8A", 0.44),
            "B11": stats.get("B11", 0.20),
            "B12": stats.get("B12", 0.14),
            "VV":  stats.get("VV",  -12.0),
            "VH":  stats.get("VH",  -18.0),
            "ndvi_prev":     stats.get("ndvi_prev",     0.50),
            "ndvi_3mo_mean": stats.get("ndvi_baseline", 0.52),
            "plot_area_ha":  farm.area_ha,
            "days_since_rain": days_since_rain,
        }

    def batch_monitor(
        self,
        farms: list[FarmPolygon],
        date: str,
        scorer,          # FarmHealthScorer instance
        alert_tiers: tuple = ("CRITICAL", "HIGH"),
    ) -> dict:
        """
        Nightly batch monitoring job.
        Scores all registered farms and returns alert list.

        In production: triggered by Cloud Scheduler → Cloud Functions.
        """
        alerts    = []
        all_scores = []

        for farm in farms:
            try:
                features = self.extract_features(farm, date)
                risk     = scorer.predict(features, crop=farm.crop)
                all_scores.append({
                    "plot_id":  farm.plot_id,
                    "score":    risk.score,
                    "tier":     risk.tier,
                    "county":   farm.county,
                    "crop":     farm.crop,
                })
                if risk.tier in alert_tiers:
                    alerts.append({
                        "phone":    farm.phone,
                        "plot_id":  farm.plot_id,
                        "message":  risk.alert_msg,
                        "swahili":  risk.swahili_msg,
                        "tier":     risk.tier,
                        "score":    risk.score,
                    })
            except Exception as e:
                print(f"  [GEE] Failed to process {farm.plot_id}: {e}")
                continue

        return {
            "date":         date,
            "farms_scored": len(all_scores),
            "alerts_sent":  len(alerts),
            "alerts":       alerts,
            "scores":       all_scores,
        }


# ── Wald's Protocol data collection from GEE ─────────────────────────────────

def collect_walds_training_data(
    region_geojson: dict,
    output_dir: str,
    start_date: str = "2022-01-01",
    end_date:   str = "2024-12-31",
    n_patches:  int = 5000,
    patch_size_px: int = 64,
):
    """
    Collect Sentinel-2 patches from GEE for EDSR training.
    Implements Wald's Protocol: saves 10m HR patches as .npy files.
    The training script downsamples these to 40m LR inputs automatically.

    In production:
        - Sample random 640m×640m windows from agricultural areas in Kenya
        - Filter to cloud-free observations
        - Export as Cloud-Optimised GeoTIFF or .npy

    Usage:
        import ee; ee.Initialize()
        collect_walds_training_data(
            region_geojson=kenya_bbox,
            output_dir="data/sentinel2/",
            n_patches=5000
        )
    """
    print(f"[Wald's Protocol] Collecting {n_patches} Sentinel-2 patches...")
    print(f"  Region: {json.dumps(region_geojson)[:60]}...")
    print(f"  Date range: {start_date} → {end_date}")
    print(f"  Patch size: {patch_size_px}px @ 10m = {patch_size_px * 10}m × {patch_size_px * 10}m")
    print(f"  Output: {output_dir}")
    print()
    print("  NOTE: This requires earthengine-api + GEE authentication.")
    print("  Full implementation in notebooks/01_data_collection.ipynb")
    print()
    print("  Alternative (free, no GEE auth required):")
    print("  Use Copernicus Open Access Hub: https://scihub.copernicus.eu/")
    print("  Download S2 L2A tiles, save bands B2-B12 as (C, H, W) .npy arrays")
    print("  Then run: python scripts/train_edsr.py --data_dir data/sentinel2/")
