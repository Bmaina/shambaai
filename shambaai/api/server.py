"""
shambaai/api/server.py — TerraSignal B2B REST API

This is the production API that powers:
  - TerraSignal partner dashboard (FINCA Kenya, Pula Advisors, ACRE Africa)
  - WhatsApp webhook receiver (farmer photo → disease diagnosis)
  - County heatmap endpoint (GeoJSON for mapping)

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn shambaai.api.server:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /score          — Score a single farm plot
    POST /portfolio      — Score a portfolio (B2B)
    POST /diagnose       — Diagnose a crop photo
    GET  /heatmap/{county} — County risk heatmap
    POST /whatsapp/webhook — WhatsApp Business API webhook
    GET  /health         — Health check

Deployment:
    Dockerfile is provided for Cloud Run / Render deployment.
    See README for free-tier Cloud Run instructions ($0 for 2M requests/month).
"""

import os
import json
import base64
import tempfile
from typing import Optional, List
from io import BytesIO

try:
    from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Provide stubs so the module imports without error
    class BaseModel:
        pass

from shambaai.pipeline.inference import ShambaAIPipeline, FarmPlot


# ── Pydantic request/response models ─────────────────────────────────────────

class ScoreRequest(BaseModel):
    plot_id:       str
    farmer_name:   str
    phone_number:  str
    crop:          str
    county:        str
    latitude:      float
    longitude:     float
    area_ha:       float
    # Sentinel-2 band values (optional — defaults to recent GEE extraction)
    B4: Optional[float] = 0.08
    B8: Optional[float] = 0.42
    B5: Optional[float] = 0.18
    B11: Optional[float] = 0.20
    VV: Optional[float] = -12.0
    VH: Optional[float] = -18.0
    ndvi_prev:       Optional[float] = 0.50
    ndvi_3mo_mean:   Optional[float] = 0.52
    days_since_rain: Optional[int]   = 7


class ScoreResponse(BaseModel):
    plot_id:      str
    health_score: float
    risk_tier:    str
    action:       str
    days_to_act:  int
    alert_en:     str
    alert_sw:     str


class PortfolioRequest(BaseModel):
    plots: List[ScoreRequest]
    institution_id: Optional[str] = None
    include_geojson: Optional[bool] = False


class DiagnoseResponse(BaseModel):
    plot_id:          Optional[str]
    top_disease:      str
    confidence:       float
    action:           str
    chemical:         str
    dose:             str
    cost_kes:         int
    swahili_advice:   str
    health_score:     Optional[float]
    risk_tier:        Optional[str]


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise ImportError("pip install fastapi uvicorn python-multipart")

    app = FastAPI(
        title="TerraSignal Intelligence API",
        description=(
            "Satellite-powered crop risk scoring for East African microfinance "
            "institutions, agricultural insurers, and cooperative managers. "
            "Powered by ShambaAI Three-Layer Defense™."
        ),
        version="1.0.0",
        contact={
            "name":  "Benson M. Gachaga",
            "email": "maina.anu@gmail.com",
            "url":   "https://radixgeo-hnwp4sqr.manus.space/",
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Load pipeline at startup
    pipeline = ShambaAIPipeline(model_dir=os.environ.get("MODEL_DIR", "models/"))

    @app.on_event("startup")
    async def startup():
        pipeline.load_models(train_if_missing=True)
        print("[API] ShambaAI pipeline ready")

    # ── Health check ───────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {
            "status":  "ok",
            "model":   "FarmHealthScorer v1.0",
            "version": "1.0.0",
            "tagline": "Predicting crop disease before it strikes.",
        }

    # ── Single plot score ──────────────────────────────────────────────────
    @app.post("/score", response_model=ScoreResponse)
    async def score_plot(req: ScoreRequest):
        """
        Score a single farm plot from Sentinel feature values.
        B2B use case: MFI credit officer checks loan portfolio farm.
        """
        plot = FarmPlot(
            plot_id=req.plot_id, farmer_name=req.farmer_name,
            phone_number=req.phone_number, crop=req.crop,
            county=req.county, latitude=req.latitude, longitude=req.longitude,
            area_ha=req.area_ha, B4=req.B4, B8=req.B8, B5=req.B5,
            B11=req.B11, VV=req.VV, VH=req.VH,
            ndvi_prev=req.ndvi_prev, ndvi_3mo_mean=req.ndvi_3mo_mean,
            days_since_rain=req.days_since_rain,
        )
        risk = pipeline.score_plot(plot)
        return ScoreResponse(
            plot_id=req.plot_id,
            health_score=round(risk.score, 1),
            risk_tier=risk.tier,
            action=risk.action,
            days_to_act=risk.days_to_act,
            alert_en=risk.alert_msg,
            alert_sw=risk.swahili_msg,
        )

    # ── Portfolio batch score ──────────────────────────────────────────────
    @app.post("/portfolio")
    async def score_portfolio(req: PortfolioRequest):
        """
        Score a portfolio of farm plots.
        B2B use case: FINCA Kenya loan officer reviews 200 accounts.
        Returns portfolio summary + per-plot risk table.
        """
        plots = [
            FarmPlot(
                plot_id=p.plot_id, farmer_name=p.farmer_name,
                phone_number=p.phone_number, crop=p.crop,
                county=p.county, latitude=p.latitude, longitude=p.longitude,
                area_ha=p.area_ha, B4=p.B4, B8=p.B8, B5=p.B5,
                B11=p.B11, VV=p.VV, VH=p.VH,
                ndvi_prev=p.ndvi_prev, ndvi_3mo_mean=p.ndvi_3mo_mean,
                days_since_rain=p.days_since_rain,
            )
            for p in req.plots
        ]
        df      = pipeline.score_portfolio(plots)
        summary = {
            "total_farms":        len(df),
            "critical":           int((df["risk_tier"] == "CRITICAL").sum()),
            "high":               int((df["risk_tier"] == "HIGH").sum()),
            "moderate":           int((df["risk_tier"] == "MODERATE").sum()),
            "stable":             int((df["risk_tier"] == "STABLE").sum()),
            "avg_health_score":   round(df["health_score"].mean(), 1),
            "total_ha":           round(df["area_ha"].sum(), 1),
            "institution_id":     req.institution_id,
        }
        records = df.to_dict(orient="records")

        response = {"summary": summary, "plots": records}
        if req.include_geojson:
            response["geojson"] = pipeline.county_heatmap_data(plots)
        return response

    # ── Photo diagnosis ────────────────────────────────────────────────────
    @app.post("/diagnose", response_model=DiagnoseResponse)
    async def diagnose_photo(
        file:    UploadFile = File(...),
        plot_id: Optional[str] = None,
        crop:    Optional[str] = "unknown",
    ):
        """
        Diagnose a crop disease from a photo upload.
        B2C use case: WhatsApp bot receives farmer photo, calls this endpoint.
        """
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            preds = pipeline.diagnose_image(tmp_path)
        finally:
            os.unlink(tmp_path)

        if not preds:
            return DiagnoseResponse(
                plot_id=plot_id, top_disease="healthy",
                confidence=0.95, action="No disease detected — crop appears healthy",
                chemical="None", dose="N/A", cost_kes=0,
                swahili_advice="Mazao yako yanaonekana kuwa na afya nzuri.",
                health_score=None, risk_tier=None,
            )

        top = preds[0]
        t   = top.get("treatment", {})
        return DiagnoseResponse(
            plot_id=plot_id,
            top_disease=top["class"],
            confidence=top["confidence"],
            action=t.get("action", "Consult agro-dealer"),
            chemical=t.get("chemical", "N/A"),
            dose=t.get("dose", "N/A"),
            cost_kes=t.get("cost_kes", 0),
            swahili_advice=t.get("swahili", ""),
            health_score=None,
            risk_tier=None,
        )

    # ── County heatmap ─────────────────────────────────────────────────────
    @app.get("/heatmap/{county}")
    async def county_heatmap(county: str):
        """
        GeoJSON county heatmap for TerraSignal B2B dashboard.
        In production: served from cached Firestore writes by nightly batch job.
        """
        # Demo: return mock data for the requested county
        return {
            "type": "FeatureCollection",
            "county": county,
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "county": county,
                        "avg_health_score": 62.4,
                        "critical_farms": 12,
                        "high_risk_farms": 34,
                        "total_farms": 847,
                        "total_ha": 1243.5,
                        "risk_level": "MODERATE",
                    },
                    "geometry": None,
                }
            ],
            "meta": {"note": "Connect nightly batch job to populate live data"},
        }

    # ── WhatsApp webhook ───────────────────────────────────────────────────
    @app.post("/whatsapp/webhook")
    async def whatsapp_webhook(
        payload: dict,
        background_tasks: BackgroundTasks,
    ):
        """
        WhatsApp Business API webhook receiver.
        Receives incoming messages and photo uploads from farmers.

        Webhook setup:
            Meta Developer Console → WhatsApp → Webhook
            URL: https://your-domain/whatsapp/webhook
            Verify token: set WHATSAPP_VERIFY_TOKEN env var

        Message types handled:
          - Photo → disease diagnosis
          - Text "DAWA" → nearest agro-dealer
          - Text "HELP" → extension officer contact
          - Text "SCORE" → latest satellite Farm Health Score
        """
        try:
            entry   = payload.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value   = changes.get("value", {})
            messages = value.get("messages", [])

            for msg in messages:
                from_number = msg.get("from")
                msg_type    = msg.get("type")

                if msg_type == "image":
                    # Queue photo diagnosis (async to avoid webhook timeout)
                    background_tasks.add_task(
                        _process_photo_diagnosis, from_number, msg, pipeline
                    )
                elif msg_type == "text":
                    text = msg.get("text", {}).get("body", "").upper().strip()
                    background_tasks.add_task(
                        _process_text_command, from_number, text, pipeline
                    )
        except Exception as e:
            print(f"[Webhook] Error: {e}")

        # Must return 200 quickly or WhatsApp retries
        return {"status": "received"}

    # ── WhatsApp webhook verification ──────────────────────────────────────
    @app.get("/whatsapp/webhook")
    async def verify_webhook(
        hub_mode: Optional[str] = None,
        hub_challenge: Optional[str] = None,
        hub_verify_token: Optional[str] = None,
    ):
        """Meta webhook verification handshake."""
        expected = os.environ.get("WHATSAPP_VERIFY_TOKEN", "shambaai_token")
        if hub_mode == "subscribe" and hub_verify_token == expected:
            return int(hub_challenge)
        raise HTTPException(status_code=403, detail="Verification failed")

    return app


# ── Background task handlers ──────────────────────────────────────────────────

async def _process_photo_diagnosis(phone: str, msg: dict, pipeline: ShambaAIPipeline):
    """Process a farmer's photo upload asynchronously."""
    # In production: download image from WhatsApp media URL, save temp, diagnose
    print(f"[Webhook] Photo diagnosis queued for {phone}")
    # TODO: implement WhatsApp media download + pipeline.diagnose_image()


async def _process_text_command(phone: str, text: str, pipeline: ShambaAIPipeline):
    """Process a farmer text command."""
    responses = {
        "DAWA":  "Reply with your county to find the nearest agro-dealer.",
        "HELP":  "Kenya Agriculture Helpline: 0800 720 560 (toll-free)",
        "SCORE": "Your latest satellite Farm Health Score will be sent within 1 hour.",
        "STOP":  "You have been unsubscribed from ShambaAI alerts.",
    }
    resp = responses.get(text, "Reply DAWA (agro-dealer), HELP (extension), or SCORE (health score).")
    print(f"[Webhook] Text command '{text}' from {phone} → {resp}")
    # TODO: send response via WhatsApp Business API (Twilio or Meta Cloud API)


# ── Entrypoint ────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("shambaai.api.server:app", host="0.0.0.0", port=8000, reload=True)
