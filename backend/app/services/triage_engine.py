"""
Motor de Reglas de Triaje Multimodal Clínico (Triage Engine)
Fase 3: Correlación cruzada entre NLP y Visión por Computadora.
"""

from typing import Dict, Any


def evaluar_triaje_multimodal(nlp_res: dict, vision_res: dict) -> dict:
    """
    Evalúa la correlación entre el análisis textual (NLP) y el análisis visual (Visión)
    siguiendo la matriz de reglas clínicas deterministas.

    Parámetros:
    - nlp_res: Diccionario con la salida de nlp_service.analizar_texto
    - vision_res: Diccionario con la salida de vision_service.analizar_imagen

    Retorna:
    - Diccionario con prioridad (CRITICA, MEDIA, BAJA), cola_asignada,
      accion_sugerida, alerta_contradiccion y motivo_decision.
    """
    intencion = nlp_res.get("intencion", "")
    sentimiento = nlp_res.get("sentimiento", "")
    urgencia = nlp_res.get("entidades", {}).get("urgencia", "MEDIA")

    clase_imagen = vision_res.get("clase_imagen", "")
    es_confiable = vision_res.get("es_confiable", False)

    # ------------------------------------------------------------------
    # Regla 1: Falla Crítica Coherente
    # Texto reporta falla de equipo (con tono negativo o urgencia alta)
    # y la imagen confirma falla de equipo o pantalla.
    # ------------------------------------------------------------------
    if (
        intencion == "REPORTE_FALLA_EQUIPO"
        and (sentimiento == "NEGATIVO" or urgencia == "ALTA")
        and clase_imagen == "EQUIPO_O_PANTALLA_CON_FALLA"
        and es_confiable
    ):
        return {
            "prioridad": "CRITICA",
            "cola_asignada": "Soporte Técnico de Equipos",
            "accion_sugerida": "ATENCION_INMEDIATA_TECNICA",
            "alerta_contradiccion": False,
            "motivo_decision": (
                "Falla física u operativa confirmada visualmente en equipo con impacto crítico en la atención clínica."
            )
        }

    # ------------------------------------------------------------------
    # Regla 2: Falla de Equipo Moderada
    # Texto reporta falla y la imagen confirma equipo con falla,
    # pero sin indicador crítico ni urgencia extrema.
    # ------------------------------------------------------------------
    if (
        intencion == "REPORTE_FALLA_EQUIPO"
        and clase_imagen == "EQUIPO_O_PANTALLA_CON_FALLA"
        and es_confiable
    ):
        return {
            "prioridad": "MEDIA",
            "cola_asignada": "Soporte Técnico de Equipos",
            "accion_sugerida": "PROGRAMAR_REVISION_EQUIPO",
            "alerta_contradiccion": False,
            "motivo_decision": (
                "Incidencia técnica reportada en equipo confirmada visualmente; programar revisión técnica estándar."
            )
        }

    # ------------------------------------------------------------------
    # Regla 3: Contradicción (Texto reporta falla pero imagen muestra todo OK)
    # ------------------------------------------------------------------
    if (
        intencion == "REPORTE_FALLA_EQUIPO"
        and clase_imagen in ["DOCUMENTO_LEGIBLE", "INSUMO_O_ETIQUETA_OK"]
        and es_confiable
    ):
        return {
            "prioridad": "MEDIA",
            "cola_asignada": "Revisión Asistida / Mesa de Ayuda",
            "accion_sugerida": "VERIFICAR_INCOHERENCIA_EVIDENCIA",
            "alerta_contradiccion": True,
            "motivo_decision": (
                "Contradicción detectada: El texto reporta falla de equipo pero la imagen adjunta muestra "
                f"un elemento en estado normal ({clase_imagen}). Requiere validación por operador."
            )
        }

    # ------------------------------------------------------------------
    # Regla 4: Documento Ilegible o Incompleto
    # Usuario envía orden/boleta o gestiona cita pero la imagen está rota/borrosa
    # ------------------------------------------------------------------
    if (
        intencion in ["OBSERVACION_DOCUMENTO_O_MUESTRA", "GESTION_CITA_O_TURNO"]
        and clase_imagen == "DOCUMENTO_BORROSO_O_INCOMPLETO"
        and es_confiable
    ):
        # Si el usuario solicitaba trámite/cita dando por hecho que la orden era válida, hay contradicción
        es_contradictorio = (intencion == "GESTION_CITA_O_TURNO")
        motivo = (
            "Evidencia documental ilegible, recortada o con reflejos. "
            + ("Contradicción: Se solicitó gestión de cita pero el soporte adjunto no es válido. " if es_contradictorio else "")
            + "Se solicita reenvío de nueva fotografía nítida para continuar el trámite."
        )
        return {
            "prioridad": "MEDIA",
            "cola_asignada": "Validación Documental / Recepción",
            "accion_sugerida": "SOLICITAR_NUEVA_IMAGEN",
            "alerta_contradiccion": es_contradictorio,
            "motivo_decision": motivo
        }

    # ------------------------------------------------------------------
    # Regla 5: Rescate Visual de Texto Ambiguo hacia Falla de Equipo
    # Texto vago (< 0.65) pero rescatado por foto clara de pantalla/equipo roto
    # ------------------------------------------------------------------
    if (
        intencion == "INTENCION_AMBIGUA"
        and clase_imagen == "EQUIPO_O_PANTALLA_CON_FALLA"
        and es_confiable
    ):
        return {
            "prioridad": "CRITICA",
            "cola_asignada": "Soporte Técnico de Equipos",
            "accion_sugerida": "DERIVACION_POR_EVIDENCIA_VISUAL",
            "alerta_contradiccion": False,
            "motivo_decision": (
                "Texto del ticket vago o ambiguo, pero rescatado determinantemente por evidencia visual concluyente "
                "de equipo o pantalla con falla crítica."
            )
        }

    # ------------------------------------------------------------------
    # Regla 6: Rescate Visual de Texto Ambiguo hacia Documento Ilegible
    # Texto poco claro y además la foto adjunta está borrosa
    # ------------------------------------------------------------------
    if (
        intencion == "INTENCION_AMBIGUA"
        and clase_imagen == "DOCUMENTO_BORROSO_O_INCOMPLETO"
        and es_confiable
    ):
        return {
            "prioridad": "MEDIA",
            "cola_asignada": "Validación Documental / Recepción",
            "accion_sugerida": "SOLICITAR_NUEVA_IMAGEN",
            "alerta_contradiccion": False,
            "motivo_decision": (
                "Texto del ticket ambiguo y documento adjunto ilegible o incompleto; se requiere asistencia y reenvío guiado."
            )
        }

    # ------------------------------------------------------------------
    # Regla 7: Trámite Documental o Gestión de Cita Válido
    # Intención de cita u observación con documento nítido o insumo OK
    # ------------------------------------------------------------------
    if (
        intencion in ["GESTION_CITA_O_TURNO", "OBSERVACION_DOCUMENTO_O_MUESTRA"]
        and clase_imagen in ["DOCUMENTO_LEGIBLE", "INSUMO_O_ETIQUETA_OK"]
        and es_confiable
    ):
        return {
            "prioridad": "BAJA",
            "cola_asignada": "Admisión y Gestión de Citas",
            "accion_sugerida": "PROCESAR_SOLICITUD_ESTANDAR",
            "alerta_contradiccion": False,
            "motivo_decision": (
                "Solicitud de trámite o gestión clínica con soporte documental e insumos válidos y legibles listos para atención."
            )
        }

    # ------------------------------------------------------------------
    # Regla 8: Consulta General con Evidencia Limpia
    # ------------------------------------------------------------------
    if (
        intencion == "CONSULTA_GENERAL_SERVICIO"
        and clase_imagen in ["DOCUMENTO_LEGIBLE", "INSUMO_O_ETIQUETA_OK"]
        and es_confiable
    ):
        return {
            "prioridad": "BAJA",
            "cola_asignada": "Atención al Paciente / Informes",
            "accion_sugerida": "RESPONDER_CONSULTA_INFORMATIVA",
            "alerta_contradiccion": False,
            "motivo_decision": (
                "Consulta informativa de servicios o requisitos sin anomalías operativas ni incidencias en la evidencia adjunta."
            )
        }

    # ------------------------------------------------------------------
    # Regla 9: Fallback / Baja Confianza Multimodal / Inconcluyente
    # ------------------------------------------------------------------
    return {
        "prioridad": "MEDIA",
        "cola_asignada": "Triaje Manual Nivel 1",
        "accion_sugerida": "REVISION_MANUAL_OPERADOR",
        "alerta_contradiccion": False,
        "motivo_decision": (
            "Baja confianza en la evidencia visual o combinación textual-visual no concluyente; derivar a revisión de operador humano."
        )
    }
