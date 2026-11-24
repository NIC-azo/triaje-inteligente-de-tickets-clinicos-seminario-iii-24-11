# Documento Técnico: Sistema de Triaje Clínico Multimodal
## Clasificación Inteligente de Tickets con Procesamiento NLP y Visión por Computadora

* **Autores:** Equipo de Investigación e Integración — Seminario III
* **Repositorio:** `triaje-multimodal-clinico`
* **Entorno:** Docker Desktop / FastAPI / React 18 + TypeScript / YOLOv8-cls / Scikit-Learn
* **Fecha:** Noviembre 2026

---

## 1. Arquitectura Multimodal del Sistema

El sistema implementa una arquitectura desacoplada de doble canal (*late fusion*) donde el análisis textual y la evidencia visual se procesan de forma independiente para confluir en un motor determinista de reglas clínicas:

```text
+---------------------------------------------------------------------------------------------------+
|                                 CLIENTE WEB / MÓVIL (React + Vite)                                |
|  - Carga de texto libre y captura fotográfica directa (accept="image/*" capture="environment")    |
|  - Fallback dinámico de IP local (http://${window.location.hostname}:8000)                        |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  | HTTP POST /triaje/evaluar-ticket (multipart/form-data)
                                                  v
+-------------------------------------------------+-------------------------------------------------+
|                                BACKEND ASÍNCRONO (FastAPI + Uvicorn)                              |
+-------------------------------------------------+-------------------------------------------------+
                         |                                                 |
         [Canal 1: Texto en Lenguaje Natural]              [Canal 2: Evidencia Visual de Cámara]
                         |                                                 |
                         v                                                 v
+-------------------------------------------------+ +-----------------------------------------------+
|         PIPELINE NLP (scikit-learn)             | |       PIPELINE DE VISIÓN (Ultralytics)        |
|  - TF-IDF Vectorizer (1-2 n-gramas)             | |  - Normalización PIL + thumbnail(640, 640)    |
|  - LinearSVC con CalibratedClassifierCV (cv=5)  | |  - YOLOv8n-cls (Pesos: best.pt, 2.83 MB)      |
|  - Clasificador de Sentimiento (LogisticReg)   | |  - Inferencia Top-1 y Top-3 de clases         |
|  - Extractor Híbrido NER (Regex + Diccionario)  | |  - Validación de Umbral (Confianza >= 0.70)   |
|  - Filtro de Ambigüedad (Confianza < 0.65)      | |                                               |
+-------------------------------------------------+ +-----------------------------------------------+
                         \                                                 /
                          \                                               /
                           v                                             v
+---------------------------------------------------------------------------------------------------+
|                        MOTOR DE CORRELACIÓN CLÍNICA (triage_engine.py)                            |
|  - Cruce de Intención y Entidades con Hallazgo Visual de Imagen                                    |
|  - Detección de Contradicciones (Texto Falla vs. Imagen Normal / Trámite vs. Documento Ilegible)  |
|  - Asignación de Prioridad (CRITICA, MEDIA, BAJA) y Cola Operativa                                |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v JSON Consolidado (Latencia ~12-50 ms)
+---------------------------------------------------------------------------------------------------+
|                                     RESPUESTA ESTRUCTURADA                                        |
|  { ticket_id, tiempo_ms, nlp: {...}, vision: {...}, triaje: { prioridad, cola, contradiccion } } |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Matriz de Reglas de Triaje Multimodal

| # | Condición NLP (Texto) | Condición Visión (Imagen) | Prioridad | Cola Asignada | Acción Sugerida | Contradicción |
| :-: | :--- | :--- | :---: | :--- | :--- | :-: |
| **1** | `REPORTE_FALLA_EQUIPO` + (`NEGATIVO` o `ALTA`) | `EQUIPO_O_PANTALLA_CON_FALLA` ($\ge 0.70$) | **`CRITICA`** | Soporte Técnico de Equipos | `ATENCION_INMEDIATA_TECNICA` | No |
| **2** | `REPORTE_FALLA_EQUIPO` (moderada) | `EQUIPO_O_PANTALLA_CON_FALLA` ($\ge 0.70$) | **`MEDIA`** | Soporte Técnico de Equipos | `PROGRAMAR_REVISION_EQUIPO` | No |
| **3** | `OBSERVACION_DOC` o `GESTION_CITA` | `DOCUMENTO_BORROSO_O_INCOMPLETO` ($\ge 0.70$) | **`MEDIA`** | Validación Documental / Recepción | `SOLICITAR_NUEVA_IMAGEN` | Sí (si pide cita) |
| **4** | `REPORTE_FALLA_EQUIPO` | `DOCUMENTO_LEGIBLE` o `INSUMO_O_ETIQUETA_OK` | **`MEDIA`** | Revisión Asistida / Mesa de Ayuda | `VERIFICAR_INCOHERENCIA_EVIDENCIA` | **Sí (Falla vs OK)** |
| **5** | `INTENCION_AMBIGUA` ($< 0.65$) | `EQUIPO_O_PANTALLA_CON_FALLA` ($\ge 0.70$) | **`CRITICA`** | Soporte Técnico de Equipos | `DERIVACION_POR_EVIDENCIA_VISUAL` | No |
| **6** | `INTENCION_AMBIGUA` ($< 0.65$) | `DOCUMENTO_BORROSO_O_INCOMPLETO` ($\ge 0.70$) | **`MEDIA`** | Validación Documental / Recepción | `SOLICITAR_NUEVA_IMAGEN` | No |
| **7** | `GESTION_CITA` u `OBSERVACION_DOC` | `DOCUMENTO_LEGIBLE` o `INSUMO_O_ETIQUETA_OK` | **`BAJA`** | Admisión y Gestión de Citas | `PROCESAR_SOLICITUD_ESTANDAR` | No |
| **8** | `CONSULTA_GENERAL_SERVICIO` | `DOCUMENTO_LEGIBLE` o `INSUMO_O_ETIQUETA_OK` | **`BAJA`** | Atención al Paciente / Informes | `RESPONDER_CONSULTA_INFORMATIVA` | No |
| **9** | Cualquier canal no concluyente | Confianza visual $< 0.70$ o discrepancia | **`MEDIA`** | Triaje Manual Nivel 1 | `REVISION_MANUAL_OPERADOR` | No |

---

## 3. Evidencia Real de Inferencia (API REST en Producción)

Las siguientes respuestas fueron obtenidas directamente del endpoint `POST /triaje/evaluar-ticket`:

### Caso A: Escenario 1 — Falla Crítica Coherente (Prioridad CRÍTICA)
```json
{
  "ticket_id": "TCK-BD2AEA8E",
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
    "es_confiable": true,
    "top3": [
      { "clase": "EQUIPO_O_PANTALLA_CON_FALLA", "confianza": 1.0 },
      { "clase": "INSUMO_O_ETIQUETA_OK", "confianza": 0.0 }
    ]
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

### Caso B: Escenario 4 — Incoherencia Texto vs. Imagen (Alerta Activada)
```json
{
  "ticket_id": "TCK-9187BC61",
  "tiempo_procesamiento_ms": 12.17,
  "nlp": {
    "texto_original": "El escáner de código de barras está totalmente roto y no lee ninguna muestra en Consultorio 1",
    "intencion": "REPORTE_FALLA_EQUIPO",
    "confianza_intencion": 0.6717,
    "sentimiento": "NEGATIVO",
    "entidades": { "sede": "Consultorio 1", "equipo_insumo": "lector de código de barras", "tipo_documento": "NONE", "urgencia": "MEDIA" }
  },
  "vision": {
    "clase_imagen": "INSUMO_O_ETIQUETA_OK",
    "confianza_imagen": 1.0,
    "es_confiable": true
  },
  "triaje": {
    "prioridad": "MEDIA",
    "cola_asignada": "Revisión Asistida / Mesa de Ayuda",
    "accion_sugerida": "VERIFICAR_INCOHERENCIA_EVIDENCIA",
    "alerta_contradiccion": true,
    "motivo_decision": "Contradicción detectada: El texto reporta falla de equipo pero la imagen adjunta muestra un elemento en estado normal (INSUMO_O_ETIQUETA_OK). Requiere validación por operador."
  }
}
```

---

## 4. Retos Técnicos Resueltos e Innovación

1. **Calibración de Probabilidades con `CalibratedClassifierCV`:**
   Los clasificadores `LinearSVC` estándar maximizan el margen geométrico pero no entregan probabilidades posteriores verdaderas. Se integró una calibración sigmoidea mediante validación cruzada (`cv=5`), permitiendo que el umbral estricto de $< 0.65$ identifique de forma determinista textos vagos o informales (*"hola ayuda con esto porfa no pasa"*, confianza: `0.6069`), derivándolos a `INTENCION_AMBIGUA` para ser rescatados por la visión.

2. **Mitigación de la Brecha de Dominio (*Domain Gap*) con Data Augmentation OpenCV:**
   Las fotografías reales tomadas con celular presentan reflejos, desenfoque por movimiento y recortes. Mediante `01_generador_dataset_img.py` se inyectaron deformaciones de perspectiva, desenfoque gaussiano/motion blur, manchas radiales simulando flash y líneas de cabezal térmico sobre el conjunto base. Esto garantizó que `YOLOv8n-cls` aprendiera a discriminar por atributos de nitidez y fallas operativas reales con un **100% de Top-1 Accuracy** en validación.

3. **Inferencia de Ultra-Baja Latencia (~12 ms) en Red Local Wi-Fi:**
   Las imágenes de celular pesan típicamente entre 3 y 8 MB. Para evitar cuellos de botella en la red local y saturación de memoria en Docker, el servicio de visión aplica redimensionamiento previo en memoria con `PIL.Image.thumbnail((640, 640))` junto con la versión optimizada de **PyTorch CPU**. Esto redujo los tiempos de inferencia total a **12–15 ms** por petición, permitiendo un uso fluido desde teléfonos móviles conectados a la Zona Wi-Fi de la PC.
