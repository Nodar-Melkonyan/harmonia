"""
Harmonia analysis API.

POST /analyze  — upload a MusicXML file, get back the analysis JSON
                 (patterns, prediction, pattern plot, cluster plot).
GET  /health   — liveness check.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Your analyzer module
from analyze_single_song import SingleSongAnalyzer, encode_image_base64

logger = logging.getLogger("harmonia.api")
logging.basicConfig(level=logging.INFO)

ALLOWED_SUFFIXES = {".xml", ".musicxml", ".mxl"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# Default artifact paths — override via env vars in production.
PROCESSED_DIR = Path(os.getenv(
    "HARMONIA_PROCESSED_DIR",
    r"C:\Users\Nod-102\Desktop\Music Project\data\processed",
))
MODEL_PATH         = os.getenv("HARMONIA_MODEL_PATH",
                               str(PROCESSED_DIR / "rf_model.pkl"))
FEATURE_NAMES_PATH = os.getenv("HARMONIA_FEATURE_NAMES_PATH",
                               str(PROCESSED_DIR / "feature_extraction_results.json"))
CLUSTER_INFO_PATH  = os.getenv("HARMONIA_CLUSTER_INFO_PATH",
                               str(PROCESSED_DIR / "cluster_info.json"))

app = FastAPI(
    title="Harmonia Analysis API",
    version="0.1.0",
    description="Analyzes a Georgian polyphonic song (MusicXML) and returns "
                "detected patterns, cluster prediction, and visualizations.",
)

# Adjust origins for your frontend deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Build analyzer once at startup (loads model + metadata once, not per request)
analyzer: SingleSongAnalyzer | None = None


@app.on_event("startup")
def _load_analyzer() -> None:
    global analyzer
    logger.info("Loading SingleSongAnalyzer...")
    logger.info("  model:         %s", MODEL_PATH)
    logger.info("  feature_names: %s", FEATURE_NAMES_PATH)
    logger.info("  cluster_info:  %s", CLUSTER_INFO_PATH)
    analyzer = SingleSongAnalyzer(
        model_path=MODEL_PATH,
        feature_names_path=FEATURE_NAMES_PATH,
        cluster_info_path=CLUSTER_INFO_PATH,
    )
    logger.info("Analyzer ready.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "analyzer_loaded": str(analyzer is not None)}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Any:
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not loaded yet.")

    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    # Read with size cap
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} bytes). "
                   f"Limit: {MAX_UPLOAD_BYTES} bytes.",
        )

    # Persist to a temp file so the analyzer can open it by path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        logger.info("Analyzing %s (%d bytes)", file.filename, len(data))

        # 1. Run analysis
        analysis = analyzer.analyze_song(str(tmp_path))
        if "error" in analysis:
            raise HTTPException(status_code=422, detail=analysis["error"])

        # 2. Create pattern plot + 3. cluster similarity plot
        pattern_plot_path = analyzer.create_visualization(analysis)
        cluster_plot_path = analyzer.create_cluster_similarity_plot(analysis)

        # 4. Embed paths + base64
        if pattern_plot_path:
            analysis["pattern_plot_path"]   = str(pattern_plot_path)
            analysis["pattern_plot_base64"] = encode_image_base64(pattern_plot_path)
        if cluster_plot_path:
            analysis["cluster_plot_path"]   = str(cluster_plot_path)
            analysis["cluster_plot_base64"] = encode_image_base64(cluster_plot_path)

        # Use the original upload filename in the response
        analysis["file_name"] = file.filename or tmp_path.name

        return analysis

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.exception("Missing artifact")
        raise HTTPException(status_code=500, detail=f"Missing artifact: {e}")
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete temp file %s", tmp_path)


# Run with:  uvicorn api_new:app --reload --port 8000