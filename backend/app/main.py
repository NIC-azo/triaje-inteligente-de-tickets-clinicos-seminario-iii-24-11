"""
API REST Principal de Triaje Multimodal Clínico
Fase 3: Endpoints REST, CORS, Lifespan y Evaluación Multimodal
"""

import time
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    HealthResponse,
    SystemInfoResponse,
    TicketTriageResponse,
    NLPResult,
    VisionResult,
    TriageDecision
)
from app.services.nlp_service import nlp_service
from app.services.vision_service import vision_service
from app.services.triage_engine import evaluar_triaje_multimodal


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestión del ciclo de vida de la aplicación.
    Inicializa los modelos en memoria una sola vez al arrancar el servidor.
    """
    print("[Lifespan] Inicializando modelos de Triaje Multimodal...")
    try:
        nlp_service.initialize()
    except Exception as e:
        print(f"[Lifespan WARN] No se pudo inicializar NLPService al arranque: {e}")

    try:
        vision_service.initialize()
    except Exception as e:
        print(f"[Lifespan WARN] No se pudo inicializar VisionService al arranque: {e}")

    yield

    print("[Lifespan] Apagando servicios de Triaje Multimodal...")


app = FastAPI(
    title="Triaje Multimodal Clínico API",
    description="API REST de triaje clínico inteligente con procesamiento multimodal (NLP + Visión por Computadora).",
    version="1.0.0",
    lifespan=lifespan
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=SystemInfoResponse, summary="Información del Sistema")
def get_system_info():
    return {
        "sistema": "Triaje Multimodal Clínico",
        "version": "1.0.0",
        "fase": "Fase 3 - API REST Multimodal y Motor de Reglas",
        "estado": "operativo",
        "entorno": settings.APP_ENV
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    return {
        "status": "healthy",
        "app_env": settings.APP_ENV,
        "nlp_threshold": settings.NLP_CONFIDENCE_THRESHOLD,
        "vision_threshold": settings.VISION_CONFIDENCE_THRESHOLD,
        "nlp_loaded": nlp_service.is_loaded(),
        "vision_loaded": vision_service.is_loaded(),
        "models_configured": {
            "yolo_model_path": settings.YOLO_MODEL_PATH,
            "nlp_model_path": settings.NLP_MODEL_PATH
        }
    }


@app.post(
    "/triaje/evaluar-ticket",
    response_model=TicketTriageResponse,
    summary="Evaluar Ticket Multimodal",
    description="Procesa texto e imagen simultáneamente para determinar prioridad, cola asignada y posibles contradicciones."
)
async def evaluar_ticket(
    texto: str = Form(..., description="Texto del ticket, mensaje o reporte clínico"),
    imagen: UploadFile = File(..., description="Evidencia fotográfica adjunta (JPG, PNG, WebP)")
):
    if not texto.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El campo 'texto' no puede estar vacío."
        )

    # Validar tipo de contenido de la imagen
    content_type = imagen.content_type or ""
    if not (content_type.startswith("image/") or imagen.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"El archivo '{imagen.filename}' no tiene un formato de imagen permitido (debe ser JPG, PNG, WEBP o BMP)."
        )

    start_time = time.perf_counter()

    try:
        image_bytes = await imagen.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El archivo de imagen está vacío."
            )

        # 1. Análisis NLP
        nlp_res = nlp_service.analizar_texto(texto)

        # 2. Análisis de Visión
        vision_res = vision_service.analizar_imagen(image_bytes)

        # 3. Motor de Reglas Multimodal
        triaje_res = evaluar_triaje_multimodal(nlp_res, vision_res)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
        timestamp_now = datetime.now(timezone.utc).isoformat()

        return {
            "ticket_id": ticket_id,
            "fecha_procesamiento": timestamp_now,
            "tiempo_procesamiento_ms": elapsed_ms,
            "nlp": nlp_res,
            "vision": vision_res,
            "triaje": triaje_res
        }

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante la evaluación multimodal: {str(e)}"
        )
