
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import existing flood detection modules
from .process_flood import run_flood_detection
from .visualize_flood import (
    create_comparison_map,
    create_interactive_map,
)

# Import ML prediction function
from .ml.predict import predict_flood


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent

FRONTEND_DIR = PROJECT_DIR / "frontend"
REPORTS_DIR = PROJECT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="AquaWatch API",
    description="Satellite Based Flood Detection System",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# STATIC REPORTS
# --------------------------------------------------

app.mount(
    "/reports",
    StaticFiles(directory=str(REPORTS_DIR)),
    name="reports",
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "AquaWatch API is running",
    }


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/api")
def home():
    return {
        "message": "Welcome to AquaWatch API",
        "version": "1.0",
        "endpoints": [
            "/health",
            "/detect-flood",
            "/predict-flood",
            "/docs",
        ],
    }


# --------------------------------------------------
# EXISTING FLOOD DETECTION
# --------------------------------------------------

@app.post("/detect-flood")
def detect_flood():
    try:
        result = run_flood_detection()

        comparison_map = create_comparison_map()
        interactive_map = create_interactive_map()

        return {
            "status": "success",
            "message": "Flood detection completed",
            "data": result,
            "comparison_map": comparison_map,
            "interactive_map": interactive_map,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Flood detection failed: {str(error)}",
        )


# --------------------------------------------------
# ML FLOOD PREDICTION
# --------------------------------------------------

@app.post("/predict-flood")
async def predict_flood_endpoint(
    before_image: UploadFile = File(...),
    after_image: UploadFile = File(...),
):
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    }

    if before_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Before image must be a JPG, PNG, or WEBP image.",
        )

    if after_image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="After image must be a JPG, PNG, or WEBP image.",
        )

    before_path = None
    after_path = None

    try:
        before_data = await before_image.read()
        after_data = await after_image.read()

        max_size = 15 * 1024 * 1024

        if len(before_data) > max_size or len(after_data) > max_size:
            raise HTTPException(
                status_code=413,
                detail="Each image must be smaller than 15 MB.",
            )

        if not before_data or not after_data:
            raise HTTPException(
                status_code=400,
                detail="Both images are required.",
            )

        # Use temporary files so uploaded images are not
        # permanently stored in the project.
        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as before_temp:
            before_temp.write(before_data)
            before_path = before_temp.name

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as after_temp:
            after_temp.write(after_data)
            after_path = after_temp.name

        prediction = predict_flood(
            before_path,
            after_path,
        )

        return {
            "status": "success",
            "message": "ML flood prediction completed",
            "data": prediction,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"ML prediction failed: {str(error)}",
        )

    finally:
        if before_path and os.path.exists(before_path):
            os.remove(before_path)

        if after_path and os.path.exists(after_path):
            os.remove(after_path)

        await before_image.close()
        await after_image.close()


# --------------------------------------------------
# FRONTEND
# Keep this mount LAST because it catches other paths.
# --------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount(
        "/",
        StaticFiles(
            directory=str(FRONTEND_DIR),
            html=True,
        ),
        name="frontend",
    )
else:
    print(f"Warning: Frontend directory not found: {FRONTEND_DIR}")


# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
