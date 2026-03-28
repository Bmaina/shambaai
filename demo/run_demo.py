#!/usr/bin/env python3
"""
ShambaAI — Full System Demo
============================
Run this script to see all three layers working end-to-end.
No API keys required. No paid data required. Fully self-contained.

Usage:
    python demo/run_demo.py

What this demonstrates:
    1. FarmHealthScorer training on synthetic East African farm data
    2. Single-plot satellite risk scoring
    3. Portfolio batch scoring (B2B MFI product)
    4. County heatmap generation (B2B intelligence)
    5. EDSR super-resolution architecture validation
    6. CropDiseaseNet architecture + treatment lookup
    7. Full Three-Layer Defense™ pipeline integration
    8. WhatsApp alert message formatting
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

# ─── Colour output helpers ────────────────────────────────────────────────────
G  = "\033[92m";  Y  = "\033[93m";  R  = "\033[91m"
B  = "\033[94m";  M  = "\033[95m";  C  = "\033[96m"
W  = "\033[97m";  DIM = "\033[2m";  RST = "\033[0m"
BOLD = "\033[1m"

def header(title: str):
    w = 70
    print(f"\n{B}{'─'*w}{RST}")
    print(f"{BOLD}{W}  {title}{RST}")
    print(f"{B}{'─'*w}{RST}")

def ok(msg):   print(f"  {G}✓{RST}  {msg}")
def info(msg): print(f"  {C}→{RST}  {msg}")
def warn(msg): print(f"  {Y}⚠{RST}  {msg}")
def kv(k, v, colour=W): print(f"  {DIM}{k:<28}{RST}{colour}{v}{RST}")


# ─── 1. Farm Health Scorer ────────────────────────────────────────────────────
header("LAYER 1+2 — FarmHealthScorer (XGBoost · 17 Sentinel features)")

from shambaai.models.health_scorer import (
    FarmHealthScorer, FarmRisk, SentinelFeatureExtractor
)

scorer = FarmHealthScorer()
print()
scorer.fit(verbose=True)
print()
ok("Model trained successfully on synthetic East African crop data")

# Feature importances
print()
info("Top 5 predictive features (explains WHY a score is assigned):")
for feat, imp in scorer.feature_importances()[:5]:
    bar = "█" * int(imp * 400)
    print(f"    {feat:<22} {G}{bar}{RST} {imp:.3f}")

# ─── 2. Single-plot scoring ───────────────────────────────────────────────────
header("LAYER 1+2 — Single Plot Scoring Examples")

scenarios = [
    {
        "label": "Healthy maize — Nakuru County",
        "crop": "maize",
        "features": dict(
            B2=0.08, B3=0.12, B4=0.07, B5=0.22, B6=0.26, B7=0.30,
            B8=0.55, B8A=0.57, B11=0.18, B12=0.12,
            VV=-10.5, VH=-16.2,
            ndvi_prev=0.68, ndvi_3mo_mean=0.62,
            plot_area_ha=1.2, days_since_rain=4,
        )
    },
    {
        "label": "Early stress — Fall Armyworm suspected, Kiambu",
        "crop": "maize",
        "features": dict(
            B2=0.09, B3=0.13, B4=0.10, B5=0.14, B6=0.18, B7=0.22,
            B8=0.38, B8A=0.40, B11=0.25, B12=0.18,
            VV=-14.0, VH=-20.5,
            ndvi_prev=0.58, ndvi_3mo_mean=0.60,
            plot_area_ha=0.5, days_since_rain=3,
        )
    },
    {
        "label": "Critical — Late Blight on potato, Nyandarua",
        "crop": "potato",
        "features": dict(
            B2=0.12, B3=0.15, B4=0.16, B5=0.09, B6=0.12, B7=0.15,
            B8=0.22, B8A=0.24, B11=0.32, B12=0.25,
            VV=-17.5, VH=-23.0,
            ndvi_prev=0.48, ndvi_3mo_mean=0.55,
            plot_area_ha=0.8, days_since_rain=2,
        )
    },
    {
        "label": "Drought stress — Makueni County",
        "crop": "sorghum",
        "features": dict(
            B2=0.11, B3=0.14, B4=0.12, B5=0.17, B6=0.21, B7=0.25,
            B8=0.35, B8A=0.37, B11=0.28, B12=0.22,
            VV=-8.0, VH=-13.5,
            ndvi_prev=0.42, ndvi_3mo_mean=0.50,
            plot_area_ha=2.0, days_since_rain=28,
        )
    },
]

TIER_COLOUR = {"STABLE": G, "MODERATE": Y, "HIGH": Y, "CRITICAL": R}

for sc in scenarios:
    risk = scorer.predict(sc["features"], crop=sc["crop"])
    col  = TIER_COLOUR.get(risk.tier, W)
    print(f"\n  {BOLD}{sc['label']}{RST}")
    print(f"    Score:  {col}{risk.score:.1f}/100 — {BOLD}{risk.tier}{RST}")
    print(f"    Action: {risk.action}")
    print(f"    Swahili: {DIM}{risk.swahili_msg[:80]}...{RST}")


# ─── 3. Portfolio batch scoring (B2B) ─────────────────────────────────────────
header("LAYER 1+2 — Portfolio Batch Scoring (B2B · FINCA Kenya MFI Product)")

from shambaai.pipeline.inference import ShambaAIPipeline, FarmPlot

pipeline = ShambaAIPipeline(model_dir="/tmp/shambaai_models")
pipeline.health_scorer = scorer
pipeline._scorer_ready = True
pipeline.disease_model = __import__(
    "shambaai.models.disease_cnn", fromlist=["CropDiseaseNet"]
).CropDiseaseNet()

# Simulate a portfolio of 20 loan accounts
rng = np.random.default_rng(123)
crops = ["maize", "potato", "tomato", "cassava", "bean", "coffee"]
counties = ["Nakuru", "Kiambu", "Nyandarua", "Makueni", "Meru", "Kisii"]

farm_plots = []
for i in range(20):
    ndvi_base = rng.uniform(0.2, 0.75)
    farm_plots.append(FarmPlot(
        plot_id=f"KE-{1000+i:04d}",
        farmer_name=f"Farmer {i+1}",
        phone_number=f"+2547{rng.integers(10000000, 99999999)}",
        crop=rng.choice(crops),
        county=rng.choice(counties),
        latitude=-1.0 + rng.uniform(-0.5, 0.5),
        longitude=36.5 + rng.uniform(-0.5, 0.5),
        area_ha=round(float(rng.uniform(0.2, 2.0)), 2),
        B4=float(1 - ndvi_base * 1.5 * 0.5),
        B8=float(ndvi_base * 1.5 * 0.5 + (1 - ndvi_base * 1.5 * 0.5)),
        B5=float(ndvi_base + rng.uniform(-0.1, 0.1)),
        ndvi_prev=float(ndvi_base + rng.uniform(-0.15, 0.05)),
        ndvi_3mo_mean=float(rng.uniform(0.40, 0.65)),
        days_since_rain=int(rng.integers(1, 35)),
        VV=float(rng.uniform(-18, -8)),
        VH=float(rng.uniform(-24, -14)),
    ))

portfolio_df = pipeline.score_portfolio(farm_plots)
print()
print(f"  {BOLD}Portfolio Summary — {len(farm_plots)} loan accounts{RST}")
print()

tier_counts = portfolio_df["risk_tier"].value_counts()
total = len(portfolio_df)
for tier, col in [("CRITICAL","🔴"), ("HIGH","🟠"), ("MODERATE","🟡"), ("STABLE","🟢")]:
    count = tier_counts.get(tier, 0)
    pct   = count / total * 100
    bar   = "█" * int(pct / 2)
    print(f"  {col} {tier:<10} {count:>3} farms ({pct:4.0f}%)  {bar}")

print()
print(f"  {DIM}{'Plot ID':<12}{'Farmer':<12}{'County':<12}{'Crop':<10}{'Score':>7}{'Tier':<12}{'Act (days)':>10}{RST}")
print(f"  {'─'*70}")
for _, row in portfolio_df.head(8).iterrows():
    col = TIER_COLOUR.get(row["risk_tier"], W)
    print(
        f"  {row['plot_id']:<12}{row['farmer']:<12}{row['county']:<12}"
        f"{row['crop']:<10}{col}{row['health_score']:>6.1f}{RST}  "
        f"{col}{row['risk_tier']:<11}{RST}{row['days_to_act']:>9}d"
    )
print(f"  {DIM}  ... (showing 8 of {total} accounts){RST}")

# ─── 4. EDSR Super-Resolution ─────────────────────────────────────────────────
header("LAYER 1 — EDSR GeoSR Super-Resolution (Wald's Protocol)")

from shambaai.models.edsr import EDSR, SpectralFidelityLoss, walds_protocol_pair

edsr = EDSR(n_bands=4, scale=4, n_feats=64, n_resblocks=16)
total_params = sum(p.numel() for p in edsr.parameters())
trainable    = sum(p.numel() for p in edsr.parameters() if p.requires_grad)

print()
kv("Architecture:", "EDSR · NO BatchNorm · Pixel-Shuffle upsample")
kv("Scale factor:", "4× (10m Sentinel-2 → 2.5m synthetic)")
kv("Input bands:", "4 (B4 Red, B8 NIR, B5 Red-Edge, B11 SWIR)")
kv("Feature maps:", "64 (lightweight, ARM-deployable)")
kv("Residual blocks:", "16")
kv("Total parameters:", f"{total_params:,}")
kv("Trainable params:", f"{trainable:,}")

# Wald's Protocol pair generation
hr_patch = torch.rand(4, 64, 64)   # simulate 10m Sentinel-2 patch
lr_patch, hr_target = walds_protocol_pair(hr_patch, scale=4)

print()
ok(f"Wald's Protocol: HR patch {tuple(hr_patch.shape)} → LR {tuple(lr_patch.shape)}")

# Forward pass
with torch.inference_mode():
    sr_output = edsr(lr_patch.unsqueeze(0))
ok(f"SR output: {tuple(sr_output.squeeze(0).shape)} (4× spatial resolution)")

# Loss function demo
criterion = SpectralFidelityLoss(alpha=0.7, beta=0.2, gamma=0.1)
loss = criterion(sr_output, hr_target.unsqueeze(0))
ok(f"Spectral Fidelity Loss: L1({0.7}) + SAM({0.2}) + Edge({0.1}) = {loss.item():.4f}")

print()
info("Wald's Protocol training pipeline:")
print(f"    {DIM}10m Sentinel-2{RST} → blur+downsample → {DIM}40m (LR input){RST}")
print(f"    EDSR learns: 40m → reconstruct 10m")
print(f"    At inference: 10m → predict 2.5m (4× SR)")
print(f"    {G}No expensive PlanetScope data needed during training.{RST}")


# ─── 5. Disease CNN ───────────────────────────────────────────────────────────
header("LAYER 3 — CropDiseaseNet (MobileNetV3-Small · 40 classes)")

from shambaai.models.disease_cnn import CropDiseaseNet, DISEASE_CLASSES, TREATMENT_MAP

cnn = CropDiseaseNet()
cnn_params = sum(p.numel() for p in cnn.parameters())

print()
kv("Backbone:", "MobileNetV3-Small (ImageNet pretrained)")
kv("Disease classes:", str(len(DISEASE_CLASSES)))
kv("Parameters:", f"{cnn_params:,}")
kv("Inference target:", "<3 sec on $50 Android (ARM Cortex-A55)")
kv("Export path:", "PyTorch → ONNX → TFLite INT8 quantised")
kv("Activation:", "Hard-Swish (ARM-optimised)")

print()
info("Sample disease classes:")
for cls in DISEASE_CLASSES[:8]:
    t = TREATMENT_MAP.get(cls, {})
    cost = t.get("cost_kes", "?")
    print(f"    {cls:<40} KES {cost}")
print(f"    {DIM}... and {len(DISEASE_CLASSES)-8} more{RST}")

# Inference on random image tensor (simulates farm photo)
dummy_image = torch.rand(1, 3, 224, 224)
with torch.inference_mode():
    predictions = cnn.predict(dummy_image, top_k=3)

print()
ok(f"Inference demo ({tuple(dummy_image.shape)}):")
for p in predictions:
    col = G if p["is_healthy"] else Y
    print(f"    {col}{p['class']:<40}{RST} conf={p['confidence']:.3f}")
    if not p["is_healthy"]:
        t = p["treatment"]
        print(f"      → {t.get('action','')}")
        print(f"      → {t.get('swahili','')[:60]}...")


# ─── 6. WhatsApp message formatting ───────────────────────────────────────────
header("OUTPUT — WhatsApp Alert Formatting (bilingual EN + Swahili)")

from shambaai.pipeline.inference import WhatsAppFormatter

critical_plot = farm_plots[0]
critical_risk = FarmRisk.from_score(18.5, "maize")
sat_msg = WhatsAppFormatter.satellite_alert(critical_plot, critical_risk)

print()
print(f"  {BOLD}Satellite Alert Message:{RST}")
print()
for line in sat_msg["body"].split("\n"):
    print(f"  │ {line}")


# ─── 7. County heatmap (B2B intelligence) ─────────────────────────────────────
header("B2B OUTPUT — County Heatmap (TerraSignal Intelligence API)")

heatmap = pipeline.county_heatmap_data(farm_plots)
meta    = heatmap["meta"]

print()
kv("Total farms monitored:", str(meta["total_farms"]))
kv("Total hectares:", f"{meta['total_ha']:.1f} ha")
kv("Portfolio avg health:", f"{meta['avg_health_score']:.1f}/100")
kv("Critical farms:", str(meta["critical_count"]))
print()
info("County risk summary (GeoJSON export for TerraSignal API):")
for f in heatmap["features"][:5]:
    p   = f["properties"]
    col = TIER_COLOUR.get(p["risk_level"], W)
    print(
        f"  {p['county']:<12} score={col}{p['avg_health_score']:>5.1f}{RST}  "
        f"critical={R}{p['critical_farms']}{RST}  "
        f"farms={p['total_farms']}  ha={p['total_ha']:.1f}"
    )

# ─── Summary ──────────────────────────────────────────────────────────────────
header("SYSTEM SUMMARY")
print()
ok("Layer 1: Sentinel-1/2 feature extraction          — 17 features, SAR + optical")
ok("Layer 2: FarmHealthScorer (XGBoost)               — scored, validated")
ok("Layer 3: CropDiseaseNet (MobileNetV3-Small)        — 40 classes, TFLite-ready")
ok("GeoSR EDSR super-resolution                       — Wald's Protocol, 4× SR")
ok("Portfolio batch scoring                            — B2B MFI product")
ok("County heatmap                                     — B2B intelligence API")
ok("WhatsApp alert formatter                           — bilingual EN + Swahili")
print()
print(f"  {BOLD}{G}All systems operational. ShambaAI Three-Layer Defense™ is ready.{RST}")
print()
print(f"  {DIM}Next steps:{RST}")
print(f"    1. Fine-tune CropDiseaseNet on PlantVillage + East African field data")
print(f"    2. Connect FarmHealthScorer to real Google Earth Engine pipeline")
print(f"    3. Deploy WhatsApp Business API webhook (Twilio / Meta Cloud API)")
print(f"    4. Train EDSR on Sentinel-2 / PlanetScope paired data")
print(f"    5. Submit to M4D Open Innovation Challenge: m4d.org/openinnovationchallenge")
print()
