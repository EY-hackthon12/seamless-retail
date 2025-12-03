from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Cognitive Retail Brain API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "retail-brain"}

# Import and include routers here later
from app.api import mobile, kiosk, models

app.include_router(mobile.router, prefix="/api/mobile", tags=["Mobile"])
app.include_router(kiosk.router, prefix="/api/kiosk", tags=["Kiosk"])
app.include_router(models.router, prefix="/api/models", tags=["Model Hosting"])

app.include_router(models.router, prefix="/api/models", tags=["Model Hosting"])

# Serve Frontend in Docker (Production)
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Docker path
frontend_dist = "/app/frontend/dist"

if os.path.exists(frontend_dist):
    # Mount assets (JS/CSS)
    app.mount("/assets", StaticFiles(directory=f"{frontend_dist}/assets"), name="assets")

    # Catch-all for SPA (React Router)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # If API route wasn't matched above, serve index.html
        return FileResponse(f"{frontend_dist}/index.html")
