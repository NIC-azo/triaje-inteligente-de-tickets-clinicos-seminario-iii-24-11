"""
Entrenamiento y Evaluación de YOLOv8-cls
Fase 2: Módulo de Visión y Data Augmentation

Entrena un modelo de clasificación visual con Ultralytics YOLOv8n-cls:
- Dataset: training/dataset_clinico
- Épocas: 35
- Tamaño de imagen: 224x224
- Batch: 16
- Semilla: 42
- Exportación automática: backend/models/best.pt
- Evaluación de 4 casos de prueba (1 por clase) y guardado en JSON.
"""

import os
import shutil
import json
import glob
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Fijar semilla para reproducibilidad
SEED = 42


def evaluate_val_set(model, val_dir, classes):
    """
    Evalúa manualmente las predicciones sobre el conjunto de validación
    para obtener matriz de confusión, Top-1 accuracy y confianza promedio por clase.
    """
    y_true = []
    y_pred = []
    confidences = {c: [] for c in classes}

    for true_idx, cls_name in enumerate(classes):
        cls_folder = os.path.join(val_dir, cls_name)
        img_paths = glob.glob(os.path.join(cls_folder, "*.jpg"))
        
        for p in img_paths:
            res = model.predict(source=p, imgsz=224, verbose=False)[0]
            pred_idx = res.probs.top1
            pred_conf = float(res.probs.top1conf)
            pred_name = res.names[pred_idx]

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            confidences[cls_name].append(pred_conf)

    # Matriz de confusión
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    top1_acc = np.mean([t == p for t, p in zip(y_true, y_pred)])
    avg_conf = {c: float(np.mean(confidences[c])) if confidences[c] else 0.0 for c in classes}

    return {
        "top1_accuracy": float(top1_acc),
        "confusion_matrix": cm.tolist(),
        "average_confidence_by_class": avg_conf,
        "classes": classes,
        "total_val_samples": len(y_true)
    }


def main():
    print("=" * 70)
    print(" FASE 2 - PASO 2: ENTRENAMIENTO DE YOLOV8-CLS (Ultralytics)")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    dataset_dir = os.path.join(script_dir, "dataset_clinico")
    val_dir = os.path.join(dataset_dir, "val")
    backend_models_dir = os.path.join(base_dir, "backend", "models")
    os.makedirs(backend_models_dir, exist_ok=True)
    target_best_pt = os.path.join(backend_models_dir, "best.pt")

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"No se encontró el dataset en: {dataset_dir}")

    # Lista de clases
    classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    print(f"[1] Clases detectadas ({len(classes)}): {classes}")
    print(f"[2] Directorio del dataset: {dataset_dir}")

    # Inicializar modelo preentrenado yolov8n-cls.pt
    print("\n[3] Inicializando YOLOv8n-cls...")
    model = YOLO("yolov8n-cls.pt")

    # Entrenar modelo
    print("\n[4] Iniciando entrenamiento (epochs=35, imgsz=224, batch=16, seed=42)...")
    train_results = model.train(
        data=dataset_dir,
        epochs=35,
        imgsz=224,
        batch=16,
        seed=SEED,
        project=os.path.join(script_dir, "runs", "classify"),
        name="triaje_yolo_cls",
        exist_ok=True,
        verbose=True
    )

    # Ubicación del mejor peso
    save_dir = model.trainer.save_dir
    best_weights_path = os.path.join(save_dir, "weights", "best.pt")

    if not os.path.exists(best_weights_path):
        # Buscar en caso de ruta alterna
        candidates = glob.glob(os.path.join(script_dir, "runs", "**", "best.pt"), recursive=True)
        if candidates:
            best_weights_path = candidates[-1]
        else:
            raise FileNotFoundError(f"No se encontró best.pt en {save_dir}")

    print(f"\n[5] Entrenamiento finalizado. Mejor peso guardado en:\n    -> {best_weights_path}")

    # Copiar a backend/models/best.pt
    shutil.copy2(best_weights_path, target_best_pt)
    print(f"[6] Copia exitosa del modelo hacia backend:\n    -> {target_best_pt}")
    print(f"    Tamaño del peso exportado: {os.path.getsize(target_best_pt) / (1024*1024):.2f} MB")

    # Evaluar sobre el conjunto val/
    print("\n[7] Evaluando métricas detalladas en conjunto de validación (val/)...")
    eval_model = YOLO(target_best_pt)
    val_metrics = evaluate_val_set(eval_model, val_dir, classes)

    print("\n" + "=" * 70)
    print(" MÉTRICAS DE VALIDACIÓN DE YOLOV8-CLS")
    print("=" * 70)
    print(f"Top-1 Accuracy: {val_metrics['top1_accuracy'] * 100:.2f}% ({val_metrics['total_val_samples']} muestras)")
    print("\nMatriz de Confusión (Filas: Real, Columnas: Predicho):")
    print(f"Clases: {classes}")
    for i, row in enumerate(val_metrics['confusion_matrix']):
        print(f"  {classes[i][:25]:<26}: {row}")

    print("\nConfianza Promedio por Clase:")
    for c, conf in val_metrics['average_confidence_by_class'].items():
        print(f"  - {c:<32}: {conf * 100:.2f}%")

    # -------------------------------------------------------------
    # Batería de 4 Casos de Prueba Visuales (1 por clase)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" [8] EJECUTANDO PRUEBAS DE INFERENCIA VISUAL (4 CASOS)")
    print("=" * 70)

    test_cases_results = []

    for cls_name in classes:
        sample_imgs = glob.glob(os.path.join(val_dir, cls_name, "*.jpg"))
        if not sample_imgs:
            continue
        test_img_path = sample_imgs[0]  # Primer ejemplo de validación

        res = eval_model.predict(source=test_img_path, imgsz=224, verbose=False)[0]
        top1_idx = res.probs.top1
        top1_name = res.names[top1_idx]
        top1_conf = float(res.probs.top1conf)

        # Extraer Top-3
        top5_indices = res.probs.top5[:3]
        top3_list = []
        for idx in top5_indices:
            top3_list.append({
                "clase": res.names[idx],
                "confianza": round(float(res.probs.data[idx]), 4)
            })

        caso_data = {
            "clase_esperada": cls_name,
            "imagen_evaluada": os.path.basename(test_img_path),
            "clase_imagen": top1_name,
            "confianza_imagen": round(top1_conf, 4),
            "es_correcta": (top1_name == cls_name),
            "top3": top3_list
        }
        test_cases_results.append(caso_data)

        print(f"\n--- Clase Esperada: {cls_name} ---")
        print(f"Imagen: {os.path.basename(test_img_path)}")
        print(f"-> Predicción Top-1: {top1_name} (Confianza: {top1_conf * 100:.2f}%)")
        print(f"-> Top-3 Detalle:   {top3_list}")

    # Guardar resultados en JSON
    json_output_path = os.path.join(base_dir, "training", "datasets", "evaluacion_casos_vision.json")
    resultado_global = {
        "metricas_validacion": val_metrics,
        "casos_prueba": test_cases_results
    }
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(resultado_global, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados exitosamente en:\n    -> {json_output_path}")
    print("=" * 70)
    print(" FASE 2: ENTRENAMIENTO Y VALIDACIÓN COMPLETADOS")
    print("=" * 70)


if __name__ == "__main__":
    main()
