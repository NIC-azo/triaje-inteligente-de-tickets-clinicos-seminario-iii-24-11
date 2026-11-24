"""
Servicio de Visión por Computadora para el Backend FastAPI (Singleton / YOLOv8-cls)
Fase 3: API REST Multimodal
"""

import os
import io
from PIL import Image
from ultralytics import YOLO
from app.config import settings


class VisionService:
    """Servicio Singleton para clasificación visual con YOLOv8-cls."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return

        model_path = settings.YOLO_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo YOLO en: {model_path}")

        print(f"[VisionService] Cargando modelo YOLOv8-cls desde: {model_path}")
        self.model = YOLO(model_path)
        self.initialized = True
        print("[VisionService] Modelo YOLOv8-cls cargado exitosamente en memoria.")

    def is_loaded(self) -> bool:
        return getattr(self, "initialized", False)

    def analizar_imagen(self, image_bytes: bytes) -> dict:
        if not self.initialized:
            self.initialize()

        # Abrir imagen con PIL y convertir a RGB
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"El archivo proporcionado no es una imagen válida: {str(e)}")

        # Optimización de resolución para fotos móviles tomadas en red local
        image.thumbnail((640, 640), Image.Resampling.LANCZOS)

        # Inferencia con YOLOv8n-cls
        results = self.model.predict(source=image, imgsz=224, verbose=False)[0]

        top1_idx = results.probs.top1
        top1_name = results.names[top1_idx]
        top1_conf = float(results.probs.top1conf)

        # Determinar si la predicción supera el umbral de confianza
        es_confiable = top1_conf >= settings.VISION_CONFIDENCE_THRESHOLD

        # Extraer Top-3
        top5_indices = results.probs.top5[:3]
        top3_list = []
        for idx in top5_indices:
            top3_list.append({
                "clase": results.names[idx],
                "confianza": round(float(results.probs.data[idx]), 4)
            })

        return {
            "clase_imagen": top1_name,
            "confianza_imagen": round(top1_conf, 4),
            "es_confiable": es_confiable,
            "top3": top3_list
        }


# Instancia Singleton
vision_service = VisionService()
