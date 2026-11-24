from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    """
    Configuración centralizada y tipada del Backend.
    Utiliza pydantic-settings para fallar de inmediato (fail-fast)
    si alguna variable de entorno requerida no se encuentra presente o es inválida.
    """
    APP_ENV: str = Field(..., description="Entorno de ejecución (development, production)")
    NLP_CONFIDENCE_THRESHOLD: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Umbral mínimo de probabilidad de intención (< 0.65 -> INTENCION_AMBIGUA)"
    )
    VISION_CONFIDENCE_THRESHOLD: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Umbral mínimo de confianza visual (< 0.70 -> Baja confianza)"
    )
    YOLO_MODEL_PATH: str = Field(..., description="Ruta al archivo de pesos YOLOv8-cls")
    NLP_MODEL_PATH: str = Field(..., description="Ruta al bundle de modelos NLP")
    CORS_ORIGINS: str = Field(..., description="Orígenes CORS permitidos")

    model_config = SettingsConfigDict(
        # Lee archivo .env relativo a donde se ejecuta o configurado expresamente
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ORIGINS.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


# Falla rápido al inicializar si falta una variable de entorno requerida
settings = Settings()
