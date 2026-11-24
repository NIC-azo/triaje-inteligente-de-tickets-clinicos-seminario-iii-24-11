"""
Servicio NLP para el Backend FastAPI (Singleton / Inferencia en memoria)
Fase 3: API REST Multimodal
"""

import os
import re
import unicodedata
import joblib
from app.config import settings


class HybridNER:
    """Extractor de entidades clínicas mediante reglas híbridas."""

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    def __init__(self):
        self.sedes_patterns = [
            ("Sede Central", [r"\bsede central\b", r"\bcentral\b"]),
            ("Sede Norte", [r"\bsede norte\b", r"\bnorte\b"]),
            ("Sede Sur", [r"\bsede sur\b", r"\bsur\b"]),
            ("Laboratorio Principal", [r"\blaboratorio principal\b", r"\blab principal\b", r"\blaboratorio\b", r"\blab\b"]),
            ("Consultorio 1", [r"\bconsultorio 1\b", r"\bcons 1\b", r"\bconsultorio uno\b"]),
            ("Consultorio 2", [r"\bconsultorio 2\b", r"\bcons 2\b", r"\bconsultorio dos\b"]),
            ("Consultorio 3", [r"\bconsultorio 3\b", r"\bcons 3\b", r"\bconsultorio tres\b"]),
        ]

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

        self.urgencia_alta = [
            r"\burgente\b", r"\burgnte\b", r"\binmediato\b", r"\bemergencia\b",
            r"\bcritico\b", r"\bhoy\b", r"\bbloqueado\b", r"\bpaciente esperando\b",
            r"\bdolor agudo\b", r"\bdolor fuerte\b", r"\bhumo\b", r"\bquemado\b"
        ]
        self.urgencia_media = [
            r"\bpronto\b", r"\bayuda\b", r"\brevision\b", r"\bproblema\b",
            r"\bfalla\b", r"\breprogramar\b", r"\bdemora\b", r"\bincompleto\b",
            r"\bborroso\b", r"\bboroso\b", r"\bconfuso\b", r"\brecortado\b"
        ]
        self.urgencia_baja = [
            r"\bconsulta\b", r"\binformes\b", r"\bhorario\b", r"\bprecio\b",
            r"\bgracias\b", r"\bconfirmar\b", r"\bduda\b", r"\bcuando\b"
        ]

    def extract(self, text: str) -> dict:
        nt = self.normalize(text)

        sede = "NONE"
        for canon, pats in self.sedes_patterns:
            if any(re.search(p, nt) for p in pats):
                sede = canon
                break

        equipo = "NONE"
        for canon, pats in self.equipos_patterns:
            if any(re.search(p, nt) for p in pats):
                equipo = canon
                break

        doc = "NONE"
        for canon, pats in self.documentos_patterns:
            if any(re.search(p, nt) for p in pats):
                doc = canon
                break

        urg = "MEDIA"
        if any(re.search(p, nt) for p in self.urgencia_alta):
            urg = "ALTA"
        elif any(re.search(p, nt) for p in self.urgencia_baja):
            urg = "BAJA"
        elif any(re.search(p, nt) for p in self.urgencia_media):
            urg = "MEDIA"

        return {
            "sede": sede,
            "equipo_insumo": equipo,
            "tipo_documento": doc,
            "urgencia": urg
        }


class NLPService:
    """Servicio Singleton para inferencia NLP."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLPService, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self):
        if self.initialized:
            return

        model_path = settings.NLP_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el modelo NLP en: {model_path}")

        print(f"[NLPService] Cargando modelo NLP desde: {model_path}")
        bundle = joblib.load(model_path)
        self.intent_vectorizer = bundle["intent_vectorizer"]
        self.intent_classifier = bundle["intent_classifier"]
        self.sentiment_vectorizer = bundle["sentiment_vectorizer"]
        self.sentiment_classifier = bundle["sentiment_classifier"]
        self.intent_classes = list(self.intent_classifier.classes_)
        self.ner = HybridNER()
        self.initialized = True
        print("[NLPService] Modelo NLP inicializado correctamente en memoria.")

    def is_loaded(self) -> bool:
        return getattr(self, "initialized", False)

    def analizar_texto(self, texto: str) -> dict:
        if not self.initialized:
            self.initialize()

        # Inferencia de Intención con probabilidades calibradas
        X_intent = self.intent_vectorizer.transform([texto])
        probs = self.intent_classifier.predict_proba(X_intent)[0]
        top_idx = probs.argmax()
        top_prob = float(probs[top_idx])
        top_intent = self.intent_classes[top_idx]

        # Regla de fallo rápido por ambigüedad (< NLP_CONFIDENCE_THRESHOLD)
        es_ambigua = top_prob < settings.NLP_CONFIDENCE_THRESHOLD
        final_intent = "INTENCION_AMBIGUA" if es_ambigua else top_intent

        # Inferencia de Sentimiento
        X_sent = self.sentiment_vectorizer.transform([texto])
        sent_pred = self.sentiment_classifier.predict(X_sent)[0]
        sent_probs = self.sentiment_classifier.predict_proba(X_sent)[0]
        sent_conf = float(sent_probs.max())

        # Extracción de Entidades NER
        entidades = self.ner.extract(texto)

        distribucion = {
            cls_name: round(float(p), 4)
            for cls_name, p in zip(self.intent_classes, probs)
        }

        return {
            "texto_original": texto,
            "intencion": final_intent,
            "intencion_base": top_intent,
            "confianza_intencion": round(top_prob, 4),
            "es_ambigua": es_ambigua,
            "sentimiento": sent_pred,
            "confianza_sentimiento": round(sent_conf, 4),
            "entidades": entidades,
            "distribucion_probabilidades": distribucion
        }


# Instancia Singleton
nlp_service = NLPService()
