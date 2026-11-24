from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class HealthResponse(BaseModel):
    status: str
    app_env: str
    nlp_threshold: float
    vision_threshold: float
    nlp_loaded: bool
    vision_loaded: bool
    models_configured: Dict[str, str]


class SystemInfoResponse(BaseModel):
    sistema: str
    version: str
    fase: str
    estado: str
    entorno: str


class EntidadesNER(BaseModel):
    sede: str
    equipo_insumo: str
    tipo_documento: str
    urgencia: str


class NLPResult(BaseModel):
    texto_original: str
    intencion: str
    intencion_base: str
    confianza_intencion: float
    es_ambigua: bool
    sentimiento: str
    confianza_sentimiento: float
    entidades: EntidadesNER
    distribucion_probabilidades: Optional[Dict[str, float]] = None


class TopClassItem(BaseModel):
    clase: str
    confianza: float


class VisionResult(BaseModel):
    clase_imagen: str
    confianza_imagen: float
    es_confiable: bool
    top3: List[TopClassItem]


class TriageDecision(BaseModel):
    prioridad: str = Field(..., description="Nivel de prioridad: CRITICA, MEDIA o BAJA")
    cola_asignada: str = Field(..., description="Área o servicio clínico al que se deriva el ticket")
    accion_sugerida: str = Field(..., description="Código de acción operativa recomendada")
    alerta_contradiccion: bool = Field(..., description="Indica si existe discrepancia entre texto e imagen")
    motivo_decision: str = Field(..., description="Explicación clínica y técnica de la resolución")


class TicketTriageResponse(BaseModel):
    ticket_id: str
    fecha_procesamiento: str
    tiempo_procesamiento_ms: float
    nlp: NLPResult
    vision: VisionResult
    triaje: TriageDecision
