"""
Script de Verificación Automatizada de la API REST Multimodal
Fase 3: Batería de Pruebas de los 6 Escenarios Obligatorios

Ejecuta peticiones multipart/form-data contra:
POST http://localhost:8000/triaje/evaluar-ticket
usando imágenes reales de training/dataset_clinico/val/
y guarda las respuestas consolidadas en training/datasets/evidencia_api_fase3.json.
"""

import os
import glob
import json
import time
import requests

API_URL = "http://localhost:8000/triaje/evaluar-ticket"
HEALTH_URL = "http://localhost:8000/health"


def wait_for_api(max_attempts=15, delay_s=2):
    """Espera hasta que la API esté lista y con los modelos cargados."""
    print("[Test Suite] Verificando disponibilidad de la API...")
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(HEALTH_URL, timeout=3)
            if r.status_code == 200:
                data = r.json()
                if data.get("nlp_loaded") and data.get("vision_loaded"):
                    print(f"[Test Suite] API y modelos listos en intento {attempt}.")
                    return True
                else:
                    print(f"[Test Suite] API viva pero modelos aún cargando (nlp: {data.get('nlp_loaded')}, vision: {data.get('vision_loaded')})...")
        except requests.exceptions.RequestException:
            pass
        time.sleep(delay_s)
    return False


def main():
    print("=" * 75)
    print(" BATERÍA DE PRUEBAS DE INTEGRACIÓN: API REST MULTIMODAL (6 ESCENARIOS)")
    print("=" * 75)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    val_dir = os.path.join(base_dir, "training", "dataset_clinico", "val")

    # Localizar imágenes de prueba en val/
    imgs_falla = glob.glob(os.path.join(val_dir, "EQUIPO_O_PANTALLA_CON_FALLA", "*.jpg"))
    imgs_borroso = glob.glob(os.path.join(val_dir, "DOCUMENTO_BORROSO_O_INCOMPLETO", "*.jpg"))
    imgs_legible = glob.glob(os.path.join(val_dir, "DOCUMENTO_LEGIBLE", "*.jpg"))
    imgs_insumo = glob.glob(os.path.join(val_dir, "INSUMO_O_ETIQUETA_OK", "*.jpg"))

    if not (imgs_falla and imgs_borroso and imgs_legible and imgs_insumo):
        raise FileNotFoundError(f"No se encontraron imágenes de prueba en {val_dir}")

    # Definición de los 6 Escenarios Obligatorios
    escenarios = [
        {
            "id": "ESCENARIO_1_CRITICO_COHERENTE",
            "titulo": "1. Crítico Coherente: Falla de Impresora + Pantalla/Equipo con Falla",
            "texto": "Urgente: La impresora térmica de Sede Sur tiene atasco de papel severo y no imprime los tickets, cabezal quemado",
            "imagen_path": imgs_falla[0],
            "esperado": {
                "prioridad": "CRITICA",
                "cola_asignada": "Soporte Técnico de Equipos",
                "alerta_contradiccion": False
            }
        },
        {
            "id": "ESCENARIO_2_AMBIGUO_DOC_BORROSO",
            "titulo": "2. Texto Ambiguo + Documento Ilegible / Recortado",
            "texto": "hola ayuda con esto porfa no pasa",
            "imagen_path": imgs_borroso[0],
            "esperado": {
                "prioridad": "MEDIA",
                "cola_asignada": "Validación Documental / Recepción",
                "alerta_contradiccion": False
            }
        },
        {
            "id": "ESCENARIO_3_CONSULTA_GENERAL_LIMPIA",
            "titulo": "3. Consulta General con Documento Nítido / Limpio",
            "texto": "Buenos días, quisiera consultar cuáles son los requisitos de ayuno y horarios para la toma de muestras",
            "imagen_path": imgs_legible[0],
            "esperado": {
                "prioridad": "BAJA",
                "cola_asignada": "Atención al Paciente / Informes",
                "alerta_contradiccion": False
            }
        },
        {
            "id": "ESCENARIO_4_CONTRADICCION_FALLA_VS_OK",
            "titulo": "4. Contradicción: Texto reporta Falla pero Imagen muestra Insumo OK",
            "texto": "El escáner de código de barras está totalmente roto y no lee ninguna muestra en Consultorio 1",
            "imagen_path": imgs_insumo[0],
            "esperado": {
                "prioridad": "MEDIA",
                "cola_asignada": "Revisión Asistida / Mesa de Ayuda",
                "alerta_contradiccion": True
            }
        },
        {
            "id": "ESCENARIO_5_CONTRADICCION_TRAMITE_VS_ROTO",
            "titulo": "5. Contradicción: Solicitud de Trámite/Cita pero Documento Ilegible/Roto",
            "texto": "Adjunto mi orden médica clara para adelantar cita urgente en Sede Central",
            "imagen_path": imgs_borroso[min(1, len(imgs_borroso) - 1)],
            "esperado": {
                "prioridad": "MEDIA",
                "cola_asignada": "Validación Documental / Recepción",
                "alerta_contradiccion": True
            }
        },
        {
            "id": "ESCENARIO_6_GESTION_CITA_VALIDA",
            "titulo": "6. Gestión de Cita Válida con Documento Legible",
            "texto": "Buenas tardes, solicito por favor reprogramar mi cita de control en Sede Sur para la próxima semana",
            "imagen_path": imgs_legible[min(1, len(imgs_legible) - 1)],
            "esperado": {
                "prioridad": "BAJA",
                "cola_asignada": "Admisión y Gestión de Citas",
                "alerta_contradiccion": False
            }
        }
    ]

    if not wait_for_api():
        raise RuntimeError(f"La API en {API_URL} no respondió en el tiempo límite.")

    resultados_consolidado = []

    for idx, esc in enumerate(escenarios, 1):
        print(f"\n[{idx}/6] Ejecutando: {esc['titulo']}")
        print(f"  - Texto:  \"{esc['texto']}\"")
        print(f"  - Imagen: {os.path.basename(esc['imagen_path'])}")

        with open(esc["imagen_path"], "rb") as f_img:
            files = {"imagen": (os.path.basename(esc["imagen_path"]), f_img, "image/jpeg")}
            data = {"texto": esc["texto"]}
            resp = requests.post(API_URL, files=files, data=data, timeout=10)

        if resp.status_code != 200:
            print(f"  [ERROR] Status Code {resp.status_code}: {resp.text}")
            continue

        resp_json = resp.json()
        triaje = resp_json["triaje"]
        nlp = resp_json["nlp"]
        vision = resp_json["vision"]

        prio_ok = triaje["prioridad"] == esc["esperado"]["prioridad"]
        cola_ok = triaje["cola_asignada"] == esc["esperado"]["cola_asignada"]
        alerta_ok = triaje["alerta_contradiccion"] == esc["esperado"]["alerta_contradiccion"]
        caso_exitoso = prio_ok and cola_ok and alerta_ok

        print(f"  -> Resultado: Prioridad={triaje['prioridad']} | Cola={triaje['cola_asignada']} | Contradicción={triaje['alerta_contradiccion']}")
        print(f"  -> Acción:    {triaje['accion_sugerida']}")
        print(f"  -> NLP:       Intención={nlp['intencion']} ({nlp['confianza_intencion']:.2f}) | Sentimiento={nlp['sentimiento']}")
        print(f"  -> Visión:    Clase={vision['clase_imagen']} ({vision['confianza_imagen']:.2f})")
        print(f"  -> Latencia:  {resp_json['tiempo_procesamiento_ms']} ms")
        print(f"  -> Validación: {'APROBADO' if caso_exitoso else 'FALLIDO'}")

        resultados_consolidado.append({
            "escenario_id": esc["id"],
            "titulo": esc["titulo"],
            "cumplio_criterios": caso_exitoso,
            "respuesta_api": resp_json
        })

    # Guardar reporte JSON
    output_json_path = os.path.join(base_dir, "training", "datasets", "evidencia_api_fase3.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(resultados_consolidado, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"Batería de pruebas completada. Reporte guardado en:\n -> {output_json_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
