# Documento Técnico: Sistema de Triaje Clínico Multimodal
## Clasificación Inteligente de Tickets con Procesamiento NLP y Visión por Computadora

* **Autores:** Equipo de Investigación e Integración — Seminario III
* **Repositorio:** [NIC-azo/triaje-inteligente-de-tickets-clinicos-seminario-iii-24-11](https://github.com/NIC-azo/triaje-inteligente-de-tickets-clinicos-seminario-iii-24-11)
* **Stack Tecnológico:** FastAPI · React 18 + TypeScript · Scikit-Learn (`Calibrated LinearSVC`) · Ultralytics (`YOLOv8n-cls`) · Docker Compose
* **Fecha:** 24 de noviembre de 2026

---

## 1. Flujo de Arquitectura del Ticket Multimodal

El sistema implementa una arquitectura de **fusión tardía (*late fusion*)** en dos canales paralelos que alimentan un motor de reglas determinista:

```text
  [ENTRADA MULTIMODAL: Cliente Web / Móvil (React + Vite)]
     ├── Texto libre del ticket + Fotografía adjunta (cámara o archivo)
     └── Petición HTTP POST /triaje/evaluar-ticket (multipart/form-data)
                                    │
                                    ▼
  [ORQUESTADOR API REST: FastAPI + Uvicorn (Singleton Lifespan)]
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
  ┌───────────────────────────────┐         ┌─────────────────────────────────┐
  │ CANAL 1: NLP (scikit-learn)   │         │ CANAL 2: VISIÓN (YOLOv8n-cls)   │
  │ • TF-IDF (n-gramas 1-2)       │         │ • PIL RGB + thumbnail(640, 640) │
  │ • CalibratedClassifierCV(SVC) │         │ • Inferencia best.pt (224x224)  │
  │ • Sentimiento (LogisticReg)   │         │ • Top-1 / Top-3 probabilidades  │
  │ • HybridNER (Regex + Dicc.)   │         │ • Umbral confianza >= 0.70      │
  │ • Umbral ambigüedad < 0.65    │         └────────────────┬────────────────┘
  └───────────────┬───────────────┘                          │
                  └─────────────────────┬────────────────────┘
                                        ▼
  [MOTOR DE REGLAS CLÍNICAS: triage_engine.py (9 Directivas Deterministas)]
     ├── Correlación Intención + Sentimiento/Urgencia × Clase Visual
     ├── Detección de Contradicciones (Texto vs. Evidencia Fotográfica)
     └── Mecanismo Fail-Safe ante combinaciones anómalas o baja confianza
                                        │
                                        ▼
  [SALIDA ESTRUCTURADA JSON (~11 - 69 ms)]
     └── { ticket_id, tiempo_procesamiento_ms, nlp, vision, triaje }
```

---

## 2. Tabla de Reglas de Cruce Texto–Imagen Implementadas (`triage_engine.py`)

| Regla | Condición NLP (Texto) | Condición Visión (Imagen) | Prioridad | Cola Asignada | Acción Sugerida | Contradicción |
| :---: | :--- | :--- | :---: | :--- | :--- | :---: |
| **R1** | `REPORTE_FALLA_EQUIPO` + (`NEGATIVO` o `ALTA`) | `EQUIPO_O_PANTALLA_CON_FALLA` ($\ge 0.70$) | **`CRITICA`** | Soporte Técnico de Equipos | `ATENCION_INMEDIATA_TECNICA` | `False` |
| **R2** | `REPORTE_FALLA_EQUIPO` (moderada) | `EQUIPO_O_PANTALLA_CON_FALLA` ($\ge 0.70$) | **`MEDIA`** | Soporte Técnico de Equipos | `PROGRAMAR_REVISION_EQUIPO` | `False` |
| **R3** | `REPORTE_FALLA_EQUIPO` | `DOCUMENTO_LEGIBLE` o `INSUMO_O_ETIQUETA_OK` ($\ge 0.70$) | **`MEDIA`** | Revisión Asistida / Mesa de Ayuda | `VERIFICAR_INCOHERENCIA_EVIDENCIA` | **`True`** |
| **R4** | `OBSERVACION_DOCUMENTO_O_MUESTRA` o `GESTION_CITA_O_TURNO` | `DOCUMENTO_BORROSO_O_INCOMPLETO` ($\ge 0.70$) | **`MEDIA`** | Validación Documental / Recepción | `SOLICITAR_NUEVA_IMAGEN` | **`True`** *(si pide cita)* |
| **R5** | `INTENCION_AMBIGUA` ($< 0.65$) | `EQUIPO_O_PANTALLA_CON_FALLA` ($\ge 0.70$) | **`CRITICA`** | Soporte Técnico de Equipos | `DERIVACION_POR_EVIDENCIA_VISUAL` | `False` |
| **R6** | `INTENCION_AMBIGUA` ($< 0.65$) | `DOCUMENTO_BORROSO_O_INCOMPLETO` ($\ge 0.70$) | **`MEDIA`** | Validación Documental / Recepción | `SOLICITAR_NUEVA_IMAGEN` | `False` |
| **R7** | `GESTION_CITA_O_TURNO` u `OBSERVACION_DOCUMENTO_O_MUESTRA` | `DOCUMENTO_LEGIBLE` o `INSUMO_O_ETIQUETA_OK` ($\ge 0.70$) | **`BAJA`** | Admisión y Gestión de Citas | `PROCESAR_SOLICITUD_ESTANDAR` | `False` |
| **R8** | `CONSULTA_GENERAL_SERVICIO` | `DOCUMENTO_LEGIBLE` o `INSUMO_O_ETIQUETA_OK` ($\ge 0.70$) | **`BAJA`** | Atención al Paciente / Informes | `RESPONDER_CONSULTA_INFORMATIVA` | `False` |
| **R9** | Combinación no concluyente *(Fallback)* | Confianza $< 0.70$ o discrepancia estructural | **`MEDIA`** | Triaje Manual Nivel 1 | `REVISION_MANUAL_OPERADOR` | `False` |

---

## 3. Evidencia JSON de Casos Reales Evaluados

### Caso A: Coherente Crítico (`ESCENARIO_1` — Regla 1)
Reporte textual de falla crítica con confirmación visual de impresora térmica dañada:

```json
{
  "ticket_id": "TCK-BD2AEA8E",
  "fecha_procesamiento": "2026-11-24T17:05:09.657238+00:00",
  "tiempo_procesamiento_ms": 56.25,
  "nlp": {
    "texto_original": "Urgente: La impresora térmica de Sede Sur tiene atasco de papel severo y no imprime los tickets, cabezal quemado",
    "intencion": "REPORTE_FALLA_EQUIPO",
    "confianza_intencion": 0.8952,
    "es_ambigua": false,
    "sentimiento": "NEGATIVO",
    "entidades": { "sede": "Sede Sur", "equipo_insumo": "impresora térmica", "tipo_documento": "NONE", "urgencia": "ALTA" }
  },
  "vision": {
    "clase_imagen": "EQUIPO_O_PANTALLA_CON_FALLA",
    "confianza_imagen": 1.0,
    "es_confiable": true
  },
  "triaje": {
    "prioridad": "CRITICA",
    "cola_asignada": "Soporte Técnico de Equipos",
    "accion_sugerida": "ATENCION_INMEDIATA_TECNICA",
    "alerta_contradiccion": false,
    "motivo_decision": "Falla física u operativa confirmada visualmente en equipo con impacto crítico en la atención clínica."
  }
}
```

### Caso B: Contradictorio con Imagen Ilegible (`ESCENARIO_5` — Regla 4)
El paciente afirma adjuntar una orden médica clara para adelantar cita, pero la imagen está borrosa/incompleta:

```json
{
  "ticket_id": "TCK-982D0CB0",
  "fecha_procesamiento": "2026-11-24T17:05:09.770032+00:00",
  "tiempo_procesamiento_ms": 11.2,
  "nlp": {
    "texto_original": "Adjunto mi orden médica clara para adelantar cita urgente en Sede Central",
    "intencion": "GESTION_CITA_O_TURNO",
    "confianza_intencion": 0.7085,
    "es_ambigua": false,
    "sentimiento": "NEUTRAL",
    "entidades": { "sede": "Sede Central", "equipo_insumo": "NONE", "tipo_documento": "orden médica", "urgencia": "ALTA" }
  },
  "vision": {
    "clase_imagen": "DOCUMENTO_BORROSO_O_INCOMPLETO",
    "confianza_imagen": 1.0,
    "es_confiable": true
  },
  "triaje": {
    "prioridad": "MEDIA",
    "cola_asignada": "Validación Documental / Recepción",
    "accion_sugerida": "SOLICITAR_NUEVA_IMAGEN",
    "alerta_contradiccion": true,
    "motivo_decision": "Evidencia documental ilegible, recortada o con reflejos. Contradicción: Se solicitó gestión de cita pero el soporte adjunto no es válido. Se solicita reenvío de nueva fotografía nítida para continuar el trámite."
  }
}
```

### Caso C: Prueba en Vivo con Imagen Externa de Cita (`TCK-58C7BA82` — Salvaguarda Fail-Safe Regla 9)
Prueba en vivo combinando un texto claro de cita médica con un escaneo digital monocromático externo (`thumb_1200_1706.png`):

```json
{
  "ticket_id": "TCK-58C7BA82",
  "fecha_procesamiento": "2026-11-24T18:12:24.695762+00:00",
  "tiempo_procesamiento_ms": 69.19,
  "nlp": {
    "texto_original": "Buenas tardes, adjunto este ticket para confirmar mi cita en Gastroenterología para el paciente Ricardo Rojas.",
    "intencion": "GESTION_CITA_O_TURNO",
    "confianza_intencion": 0.8946,
    "es_ambigua": false,
    "sentimiento": "NEUTRAL",
    "entidades": { "sede": "NONE", "equipo_insumo": "NONE", "tipo_documento": "NONE", "urgencia": "BAJA" }
  },
  "vision": {
    "clase_imagen": "EQUIPO_O_PANTALLA_CON_FALLA",
    "confianza_imagen": 0.9576,
    "es_confiable": true
  },
  "triaje": {
    "prioridad": "MEDIA",
    "cola_asignada": "Triaje Manual Nivel 1",
    "accion_sugerida": "REVISION_MANUAL_OPERADOR",
    "alerta_contradiccion": false,
    "motivo_decision": "Baja confianza en la evidencia visual o combinación textual-visual no concluyente; derivar a revisión de operador humano."
  }
}
```

---

## 4. Análisis de Fallos Durante Entrenamiento y Pruebas, y su Resolución

1. **Calibración de Ambigüedad en NLP (`LinearSVC` sin probabilidades):**
   * **Qué falló:** Un `LinearSVC` estándar entrega distancias al hiperplano en lugar de probabilidades reales, impidiendo fijar un umbral confiable (`< 0.65`) para textos vagos de celulares (ej. *"hola ayuda con esto porfa no pasa"*).
   * **Cómo se resolvió:** Se envolvió el clasificador con `CalibratedClassifierCV(cv=5)` para obtener probabilidades calibradas (`0.6069` en el caso ambiguo, activando `INTENCION_AMBIGUA`) y se implementó un extractor `HybridNER` (Regex + diccionario normalizado) resistente a abreviaturas y errores ortográficos.

2. **Brecha de Dominio (*Domain Shift*) en Visión durante Pruebas en Vivo:**
   * **Qué falló:** Al evaluar en el cliente web un escaneo digital vertical en escala de grises y fondo blanco plano (`thumb_1200_1706.png`, Modo `L`, 1200×1706 px), `YOLOv8n-cls` lo clasificó erróneamente como `EQUIPO_O_PANTALLA_CON_FALLA` (`95.76%`). Al haberse entrenado con fotografías RGB horizontales (1408×768 px) en entorno clínico físico, la compresión a 224×224 de un ticket monocromático replicó visualmente las franjas de impresora térmica dañada. Con texto vago (*"este es mi ticket"*), esto provocó un falso escalamiento a `CRITICA`.
   * **Cómo se resolvió:** Se comprobó primero la resiliencia del diseño multimodal: al enviar un texto explícito de cita (`TCK-58C7BA82`), la **Regla 9 (*Fail-Safe*)** bloqueó el envío erróneo a Soporte Técnico y derivó el caso a `Triaje Manual Nivel 1` (`MEDIA`). Posteriormente, al validar con una **fotografía real tomada en entorno físico** de un ticket de EsSalud (`images (2).jpeg`, ticket en vivo `TCK-391A37B4`, *"buenas, adjunto este ticket para confirmar mi cita en medicina general para el paciente cueto"*), el modelo de visión clasificó correctamente **`DOCUMENTO_LEGIBLE` con 99.5% de confianza**, el NLP detectó `GESTION_CITA_O_TURNO` (`89.5%`) y el motor resolvió en **23.38 ms** como prioridad **`BAJA`** hacia **`Admisión y Gestión de Citas`**.

3. **Peso de Contenedor Docker y Latencia de Carga Móvil:**
   * **Qué falló:** En las primeras compilaciones del backend, `ultralytics` intentaba descargar dependencias CUDA de >2.5 GB y fallaba OpenCV por falta de librerías gráficas del sistema (`libgl1`, `libglib2.0-0`).
   * **Cómo se resolvió:** Se actualizó `backend/Dockerfile` preinstalando `torch` y `torchvision` en su versión CPU (`--index-url https://download.pytorch.org/whl/cpu`), añadiendo las dependencias de sistema mínimas y aplicando `PIL.Image.thumbnail((640, 640))` en `vision_service.py`, logrando tiempos de respuesta estables entre **11 y 69 ms**.
