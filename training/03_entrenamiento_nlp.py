"""
Módulo de Entrenamiento NLP - Triaje Multimodal Clínico
Fase 1: Módulo NLP y Lógica de Priorización

Este script entrena:
1. Clasificador de Intención: TF-IDF + CalibratedClassifierCV(LinearSVC) con probabilidades calibradas.
2. Clasificador de Sentimiento: TF-IDF + LogisticRegression.
3. Extractor de Entidades Híbrido (NER): Regex + Diccionarios normalizados.
4. Exporta el artefacto serializado a backend/models/nlp_bundle.joblib.
5. Ejecuta 5 casos de prueba de inferencia clínica.
"""

import os
import re
import json
import unicodedata
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =====================================================================
# 1. Extractor de Entidades Híbrido (Regex + Diccionario Normalizado)
# =====================================================================

class HybridNER:
    """
    Extractor de entidades clínicas mediante reglas híbridas
    (expresiones regulares y diccionarios de sinónimos normalizados).
    """

    @staticmethod
    def normalize(text: str) -> str:
        """Convierte a minúsculas y elimina acentos/tildes."""
        if not text:
            return ""
        text = text.lower()
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def __init__(self):
        # Diccionario de Sedes
        self.sedes_patterns = [
            ("Sede Central", [r"\bsede central\b", r"\bcentral\b"]),
            ("Sede Norte", [r"\bsede norte\b", r"\bnorte\b"]),
            ("Sede Sur", [r"\bsede sur\b", r"\bsur\b"]),
            ("Laboratorio Principal", [r"\blaboratorio principal\b", r"\blab principal\b", r"\blaboratorio\b", r"\blab\b"]),
            ("Consultorio 1", [r"\bconsultorio 1\b", r"\bcons 1\b", r"\bconsultorio uno\b"]),
            ("Consultorio 2", [r"\bconsultorio 2\b", r"\bcons 2\b", r"\bconsultorio dos\b"]),
            ("Consultorio 3", [r"\bconsultorio 3\b", r"\bcons 3\b", r"\bconsultorio tres\b"]),
        ]

        # Diccionario de Equipos e Insumos
        self.equipos_patterns = [
            ("impresora térmica", [
                r"\bimpresora termica\b", r"\bimp\.?\s*termica\b", r"\bimpresora\b",
                r"\bcabezal termico\b", r"\bcabezal\b", r"\brodillo\b", r"\btermica\b"
            ]),
            ("lector de código de barras", [
                r"\blector de codigo de barras\b", r"\blector de barras\b", r"\blector de codigo\b",
                r"\blector\b", r"\bescaner\b", r"\bscanner\b", r"\bpistola de barras\b"
            ]),
            ("analizador hematológico", [
                r"\banalizador hematologico\b", r"\banalizador hematologia\b", r"\banalizador\b",
                r"\bequipo hematologico\b"
            ]),
            ("centrífuga", [r"\bcentrifuga\b", r"\bcentrifugadora\b"]),
            ("etiqueta", [r"\betiqueta\b", r"\betiquetas\b", r"\bsticker\b", r"\bstickers\b"]),
            ("tubo de muestra", [
                r"\btubo de muestra\b", r"\btubo\b", r"\btubos\b", r"\bfrasco\b",
                r"\bfrasco de muestra\b", r"\bmuestra de sangre\b"
            ]),
            ("tóner", [r"\btoner\b", r"\btinta\b", r"\bcartucho\b"]),
            ("monitor", [r"\bmonitor\b", r"\bpantalla\b", r"\bdisplay\b"]),
        ]

        # Diccionario de Tipos de Documento
        self.documentos_patterns = [
            ("orden médica", [
                r"\borden medica\b", r"\bord\.?\s*med\.?\b", r"\borden med\b",
                r"\borden\b", r"\breceta medica\b"
            ]),
            ("comprobante de pago", [
                r"\bcomprobante de pago\b", r"\bcomprobante\b", r"\bcomprovante\b",
                r"\bvoucher\b", r"\bvaucher\b"
            ]),
            ("boleta", [r"\bboleta electronica\b", r"\bboleta de pago\b", r"\bboleta\b", r"\bfactura\b"]),
            ("DNI", [r"\bdni\b", r"\bdocumento de identidad\b", r"\bcarnet\b"]),
            ("resultado de laboratorio", [
                r"\bresultado de laboratorio\b", r"\bresultados de laboratorio\b",
                r"\bresultado lab\b", r"\bresultados lab\b", r"\banalisis de sangre\b"
            ]),
            ("consentimiento informado", [r"\bconsentimiento informado\b", r"\bconsentimiento\b"]),
        ]

        # Patrones de Urgencia
        self.urgencia_alta_patterns = [
            r"\burgente\b", r"\burgnte\b", r"\binmediato\b", r"\bemergencia\b",
            r"\bcritico\b", r"\bhoy\b", r"\bbloqueado\b", r"\bpaciente esperando\b",
            r"\bdolor agudo\b", r"\bdolor fuerte\b", r"\bhumo\b", r"\bquemado\b"
        ]
        self.urgencia_media_patterns = [
            r"\bpronto\b", r"\bayuda\b", r"\brevision\b", r"\bproblema\b",
            r"\bfalla\b", r"\breprogramar\b", r"\bdemora\b", r"\bincompleto\b",
            r"\bborroso\b", r"\bboroso\b", r"\bconfuso\b", r"\brecortado\b"
        ]
        self.urgencia_baja_patterns = [
            r"\bconsulta\b", r"\binformes\b", r"\bhorario\b", r"\bprecio\b",
            r"\bgracias\b", r"\bconfirmar\b", r"\bduda\b", r"\bcuando\b",
            r"\bcual es\b", r"\bdonde queda\b"
        ]

    def extract_entities(self, text: str) -> dict:
        """Extrae entidades normalizadas desde el texto en lenguaje natural."""
        norm_text = self.normalize(text)

        # 1. Extraer Sede
        sede_detectada = "NONE"
        for canon, patterns in self.sedes_patterns:
            for pat in patterns:
                if re.search(pat, norm_text):
                    sede_detectada = canon
                    break
            if sede_detectada != "NONE":
                break

        # 2. Extraer Equipo / Insumo
        equipo_detectado = "NONE"
        for canon, patterns in self.equipos_patterns:
            for pat in patterns:
                if re.search(pat, norm_text):
                    equipo_detectado = canon
                    break
            if equipo_detectado != "NONE":
                break

        # 3. Extraer Tipo de Documento
        documento_detectado = "NONE"
        for canon, patterns in self.documentos_patterns:
            for pat in patterns:
                if re.search(pat, norm_text):
                    documento_detectado = canon
                    break
            if documento_detectado != "NONE":
                break

        # 4. Extraer Nivel de Urgencia
        urgencia_detectada = "MEDIA"  # Default equilibrado
        if any(re.search(pat, norm_text) for pat in self.urgencia_alta_patterns):
            urgencia_detectada = "ALTA"
        elif any(re.search(pat, norm_text) for pat in self.urgencia_baja_patterns):
            urgencia_detectada = "BAJA"
        elif any(re.search(pat, norm_text) for pat in self.urgencia_media_patterns):
            urgencia_detectada = "MEDIA"

        return {
            "sede": sede_detectada,
            "equipo_insumo": equipo_detectado,
            "tipo_documento": documento_detectado,
            "urgencia": urgencia_detectada
        }


# =====================================================================
# 2. Clase de Servicio NLP para Inferencia
# =====================================================================

class ClinicalNLPInference:
    """Clase envoltorio para ejecutar predicciones a partir del bundle serializado."""

    def __init__(self, bundle: dict):
        self.intent_vectorizer = bundle["intent_vectorizer"]
        self.intent_classifier = bundle["intent_classifier"]
        self.sentiment_vectorizer = bundle["sentiment_vectorizer"]
        self.sentiment_classifier = bundle["sentiment_classifier"]
        self.ner = HybridNER()
        self.default_threshold = bundle.get("nlp_confidence_threshold", 0.65)

    def predict(self, text: str, threshold: float = None) -> dict:
        if threshold is None:
            threshold = self.default_threshold

        # Predicción de Intención con CalibratedClassifierCV
        X_intent = self.intent_vectorizer.transform([text])
        probs = self.intent_classifier.predict_proba(X_intent)[0]
        classes = self.intent_classifier.classes_

        top_idx = probs.argmax()
        top_prob = float(probs[top_idx])
        top_intent = classes[top_idx]

        # Lógica de Umbral para INTENCION_AMBIGUA
        if top_prob < threshold:
            final_intent = "INTENCION_AMBIGUA"
            es_ambigua = True
        else:
            final_intent = top_intent
            es_ambigua = False

        # Predicción de Sentimiento
        X_sent = self.sentiment_vectorizer.transform([text])
        sentiment_pred = self.sentiment_classifier.predict(X_sent)[0]
        sentiment_probs = self.sentiment_classifier.predict_proba(X_sent)[0]
        sentiment_conf = float(sentiment_probs.max())

        # Extracción NER
        entidades = self.ner.extract_entities(text)

        distribucion = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(classes, probs)
        }

        return {
            "texto": text,
            "intencion": final_intent,
            "intencion_base": top_intent,
            "confianza_intencion": round(top_prob, 4),
            "es_ambigua": es_ambigua,
            "umbral_aplicado": threshold,
            "sentimiento": sentiment_pred,
            "confianza_sentimiento": round(sentiment_conf, 4),
            "entidades": entidades,
            "probabilidades_detalle": distribucion
        }


# =====================================================================
# 3. Flujo Principal de Entrenamiento
# =====================================================================

def main():
    print("=" * 70)
    print(" FASE 1: ENTRENAMIENTO NLP Y LÓGICA DE PRIORIZACIÓN")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "training", "datasets", "tickets_nlp.csv")
    bundle_output_path = os.path.join(base_dir, "backend", "models", "nlp_bundle.joblib")

    print(f"\n[1] Cargando dataset desde: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"No se encontró el dataset en {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"    - Registros cargados: {len(df)}")
    print(f"    - Distribución de intenciones:\n{df['intencion'].value_counts().to_string()}")

    X = df["texto"].values
    y_intent = df["intencion"].values
    y_sentiment = df["sentimiento"].values

    # Split estratificado 80/20
    X_train, X_test, y_intent_train, y_intent_test, y_sent_train, y_sent_test = train_test_split(
        X, y_intent, y_sentiment,
        test_size=0.20,
        random_state=42,
        stratify=y_intent
    )

    print(f"\n[2] División de datos: Train={len(X_train)} (80%), Test={len(X_test)} (20%)")

    # -------------------------------------------------------------
    # 2.1 Entrenar Clasificador de Intención (Calibrated LinearSVC)
    # -------------------------------------------------------------
    print("\n[3] Entrenando Clasificador de Intención...")
    intent_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_intent_vec = intent_vectorizer.fit_transform(X_train)
    X_test_intent_vec = intent_vectorizer.transform(X_test)

    base_svc = LinearSVC(class_weight="balanced", random_state=42, dual="auto")
    # CalibratedClassifierCV para calibración sigmoidea/isotónica de probabilidades
    intent_classifier = CalibratedClassifierCV(
        estimator=base_svc,
        cv=5
    )
    intent_classifier.fit(X_train_intent_vec, y_intent_train)

    y_intent_pred = intent_classifier.predict(X_test_intent_vec)
    intent_acc = accuracy_score(y_intent_test, y_intent_pred)

    print(f"\n=== MÉTRICAS DE CLASIFICACIÓN DE INTENCIÓN ===")
    print(f"Accuracy en Test: {intent_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_intent_test, y_intent_pred, digits=4))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_intent_test, y_intent_pred))

    # -------------------------------------------------------------
    # 2.2 Entrenar Clasificador de Sentimiento (Logistic Regression)
    # -------------------------------------------------------------
    print("\n[4] Entrenando Clasificador de Sentimiento...")
    sentiment_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_sent_vec = sentiment_vectorizer.fit_transform(X_train)
    X_test_sent_vec = sentiment_vectorizer.transform(X_test)

    sentiment_classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    sentiment_classifier.fit(X_train_sent_vec, y_sent_train)

    y_sent_pred = sentiment_classifier.predict(X_test_sent_vec)
    sent_acc = accuracy_score(y_sent_test, y_sent_pred)

    print(f"\n=== MÉTRICAS DE CLASIFICACIÓN DE SENTIMIENTO ===")
    print(f"Accuracy en Test: {sent_acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_sent_test, y_sent_pred, digits=4))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_sent_test, y_sent_pred))

    # -------------------------------------------------------------
    # 2.3 Empaquetar y Serializar Modelo en nlp_bundle.joblib
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(bundle_output_path), exist_ok=True)
    bundle_data = {
        "metadata": {
            "version": "1.0.0",
            "fase": "Fase 1",
            "framework": "scikit-learn",
            "intent_accuracy": round(intent_acc, 4),
            "sentiment_accuracy": round(sent_acc, 4),
            "total_samples": len(df)
        },
        "intent_vectorizer": intent_vectorizer,
        "intent_classifier": intent_classifier,
        "sentiment_vectorizer": sentiment_vectorizer,
        "sentiment_classifier": sentiment_classifier,
        "intent_classes": list(intent_classifier.classes_),
        "sentiment_classes": list(sentiment_classifier.classes_),
        "nlp_confidence_threshold": 0.65
    }

    joblib.dump(bundle_data, bundle_output_path)
    print(f"\n[5] Bundle NLP serializado exitosamente en:\n    -> {bundle_output_path}")
    print(f"    Tamaño del archivo: {os.path.getsize(bundle_output_path) / 1024:.2f} KB")

    # -------------------------------------------------------------
    # 2.4 Batería de 5 Casos de Prueba de Inferencia Clínica
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [6] EJECUTANDO BATERÍA DE 5 CASOS DE PRUEBA DE INFERENCIA CLÍNICA")
    print("=" * 70)

    inferencia = ClinicalNLPInference(bundle_data)

    casos_prueba = [
        {
            "id": "CASO_1_FALLA_CRITICA",
            "descripcion": "Falla crítica de impresora térmica en Sede Sur",
            "texto": "Urgente: La impresora térmica de Sede Sur tiene atasco de papel y no imprime nada, cabezal dañado"
        },
        {
            "id": "CASO_2_DOCUMENTO_BORROSO",
            "descripcion": "Orden médica borrosa y recortada enviada por paciente",
            "texto": "El paciente envió foto de orden médica muy borosa y cortada en los bordes, no se lee"
        },
        {
            "id": "CASO_3_GESTION_CITA",
            "descripcion": "Reprogramación de cita médica para próxima semana",
            "texto": "Buenas tardes, solicito por favor reprogramar mi cita médica para la próxima semana"
        },
        {
            "id": "CASO_4_CONSULTA_AYUNO",
            "descripcion": "Consulta de requisitos de ayuno para examen de laboratorio",
            "texto": "Buenos días, ¿cuáles son los requisitos de ayuno para el análisis de glucosa en el laboratorio?"
        },
        {
            "id": "CASO_5_AMBIGUO_VAGO",
            "descripcion": "Caso ambiguo/vago para evaluar umbral < 0.65",
            "texto": "hola ayuda con esto porfa no pasa"
        }
    ]

    resultados_evaluacion = []

    for caso in casos_prueba:
        res = inferencia.predict(caso["texto"])
        res_consolidado = {
            "caso_id": caso["id"],
            "descripcion": caso["descripcion"],
            "resultado": res
        }
        resultados_evaluacion.append(res_consolidado)

        print(f"\n--- {caso['id']}: {caso['descripcion']} ---")
        print(f"Texto: \"{caso['texto']}\"")
        print(f"-> Intención Final: {res['intencion']} (Confianza: {res['confianza_intencion']:.4f})")
        print(f"-> Intención Base:  {res['intencion_base']} | Es Ambigua (<0.65): {res['es_ambigua']}")
        print(f"-> Sentimiento:     {res['sentimiento']} (Confianza: {res['confianza_sentimiento']:.4f})")
        print(f"-> Entidades NER:   {res['entidades']}")

    # Guardar resultados en json para registro y verificación
    resultados_json_path = os.path.join(base_dir, "training", "datasets", "evaluacion_casos_nlp.json")
    with open(resultados_json_path, "w", encoding="utf-8") as f:
        json.dump(resultados_evaluacion, f, indent=2, ensure_ascii=False)
    print(f"\nResultados de las pruebas guardados en: {resultados_json_path}")
    print("=" * 70)
    print(" FASE 1: ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 70)


if __name__ == "__main__":
    main()
