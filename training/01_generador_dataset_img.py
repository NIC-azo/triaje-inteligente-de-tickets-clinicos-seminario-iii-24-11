"""
Script de Generación y Data Augmentation Estandarizado para YOLOv8-cls
Fase 2: Módulo de Visión por Computadora

Genera 50 imágenes por clase (200 en total):
- 40 en train/ (80%)
- 10 en val/ (20%)
Redimensionadas a 224x224 con transformaciones específicas por clase.
"""

import os
import glob
import random
import numpy as np
import cv2

# Fijar semilla para reproducibilidad exacta
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

IMG_SIZE = (224, 224)
SAMPLES_PER_CLASS = 50
TRAIN_RATIO = 0.8  # 40 train, 10 val

CLASSES = [
    "DOCUMENTO_LEGIBLE",
    "DOCUMENTO_BORROSO_O_INCOMPLETO",
    "EQUIPO_O_PANTALLA_CON_FALLA",
    "INSUMO_O_ETIQUETA_OK"
]


# =====================================================================
# Transformaciones con OpenCV
# =====================================================================

def rotate_image(img, max_angle=4.0):
    """Rotación leve en torno al centro con réplica de bordes."""
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def adjust_brightness_contrast(img, alpha_range=(0.88, 1.12), beta_range=(-12, 12)):
    """Ajuste suave de contraste (alpha) y brillo (beta)."""
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    beta = np.random.uniform(beta_range[0], beta_range[1])
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def smooth_perspective(img, max_shift=8):
    """Deformación de perspectiva leve."""
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dx1, dy1 = np.random.uniform(-max_shift, max_shift, 2)
    dx2, dy2 = np.random.uniform(-max_shift, max_shift, 2)
    dx3, dy3 = np.random.uniform(-max_shift, max_shift, 2)
    dx4, dy4 = np.random.uniform(-max_shift, max_shift, 2)
    pts2 = np.float32([
        [max(0, dx1), max(0, dy1)],
        [w - max(0, -dx2), max(0, dy2)],
        [max(0, dx3), h - max(0, -dy3)],
        [w - max(0, -dx4), h - max(0, -dy4)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_motion_blur(img, kernel_size=15):
    """Desenfoque de movimiento direccional (horizontal o diagonal)."""
    kernel = np.zeros((kernel_size, kernel_size))
    if np.random.rand() > 0.5:
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    else:
        np.fill_diagonal(kernel, 1)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_gaussian_blur(img, ksize=15):
    """Desenfoque gaussiano para simular pérdida de foco."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_flash_glare(img):
    """Mancha circular blanca semitransparente simulando reflejo de flash."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cx = int(np.random.uniform(w * 0.25, w * 0.75))
    cy = int(np.random.uniform(h * 0.25, h * 0.75))
    radius = int(np.random.uniform(w * 0.15, w * 0.35))
    intensity = np.random.uniform(0.4, 0.75)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    mask = cv2.GaussianBlur(mask, (45, 45), 0)

    white = np.full_like(img, 255)
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    blended = (img * (1 - mask_3d * intensity) + white * (mask_3d * intensity)).astype(np.uint8)
    return blended


def apply_thermal_printer_lines(img):
    """Franjas horizontales blancas simulando cabezal térmico dañado."""
    res = img.copy()
    h, w = res.shape[:2]
    num_lines = np.random.randint(1, 4)
    for _ in range(num_lines):
        y = np.random.randint(int(h * 0.1), int(h * 0.9))
        thickness = np.random.randint(2, 5)
        cv2.line(res, (0, y), (w, y), (255, 255, 255), thickness)
    return res


def apply_aggressive_crop(img):
    """Recorte agresivo del 25-35% de un borde simulando documento incompleto."""
    h, w = img.shape[:2]
    crop_fraction = np.random.uniform(0.25, 0.35)
    side = np.random.choice(["top", "bottom", "left", "right"])

    if side == "top":
        ch = int(h * crop_fraction)
        cropped = img[ch:, :]
    elif side == "bottom":
        ch = int(h * (1 - crop_fraction))
        cropped = img[:ch, :]
    elif side == "left":
        cw = int(w * crop_fraction)
        cropped = img[:, cw:]
    else:
        cw = int(w * (1 - crop_fraction))
        cropped = img[:, :cw]

    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def add_gaussian_noise(img, sigma=18):
    """Ruido gaussiano de sensor de cámara móvil."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def adjust_saturation(img, factor=1.2):
    """Ajuste de saturación en espacio HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# =====================================================================
# Generación Específica por Clase
# =====================================================================

def augment_by_class(img, clase):
    """Aplica el pipeline de transformaciones de acuerdo a la clase objetivo."""
    res = img.copy()

    if clase == "DOCUMENTO_LEGIBLE":
        # 1. Rotación leve (-4 a 4 grados)
        res = rotate_image(res, max_angle=4.0)
        # 2. Leve variación de brillo/contraste
        res = adjust_brightness_contrast(res, alpha_range=(0.92, 1.08), beta_range=(-8, 8))
        # 3. Perspectiva suave
        if np.random.rand() > 0.4:
            res = smooth_perspective(res, max_shift=5)

    elif clase == "DOCUMENTO_BORROSO_O_INCOMPLETO":
        # Selección variada de degradaciones
        tipo_degradacion = np.random.choice([
            "blur_gaussiano", "motion_blur", "crop_agresivo",
            "flash_reflejo", "lineas_cabezal", "combinado"
        ])

        if tipo_degradacion == "blur_gaussiano":
            res = apply_gaussian_blur(res, ksize=np.random.choice([13, 17, 21]))
        elif tipo_degradacion == "motion_blur":
            res = apply_motion_blur(res, kernel_size=np.random.choice([15, 21, 27]))
        elif tipo_degradacion == "crop_agresivo":
            res = apply_aggressive_crop(res)
        elif tipo_degradacion == "flash_reflejo":
            res = apply_flash_glare(res)
            if np.random.rand() > 0.5:
                res = apply_gaussian_blur(res, ksize=7)
        elif tipo_degradacion == "lineas_cabezal":
            res = apply_thermal_printer_lines(res)
        else:  # Combinado
            res = apply_aggressive_crop(res)
            res = apply_gaussian_blur(res, ksize=11)
            if np.random.rand() > 0.5:
                res = apply_thermal_printer_lines(res)

        # Rotación leve para variabilidad adicional
        res = rotate_image(res, max_angle=6.0)

    elif clase == "EQUIPO_O_PANTALLA_CON_FALLA":
        # 1. Cambios de ángulo y perspectiva
        res = smooth_perspective(res, max_shift=12)
        res = rotate_image(res, max_angle=8.0)
        # 2. Ruido gaussiano de cámara móvil
        res = add_gaussian_noise(res, sigma=np.random.uniform(12, 22))
        # 3. Ajuste de contraste y saturación preservando indicadores
        res = adjust_brightness_contrast(res, alpha_range=(0.85, 1.20), beta_range=(-15, 15))
        if np.random.rand() > 0.4:
            res = adjust_saturation(res, factor=np.random.uniform(0.9, 1.35))

    elif clase == "INSUMO_O_ETIQUETA_OK":
        # 1. Variaciones suaves de iluminación
        res = adjust_brightness_contrast(res, alpha_range=(0.90, 1.10), beta_range=(-10, 10))
        # 2. Rotación leve
        res = rotate_image(res, max_angle=5.0)
        # 3. Escala / perspectiva suave manteniendo código legible
        if np.random.rand() > 0.5:
            res = smooth_perspective(res, max_shift=6)

    # Redimensionar a resolución estándar 224x224
    res = cv2.resize(res, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return res


# =====================================================================
# Orquestación del Dataset Clínico
# =====================================================================

def main():
    print("=" * 70)
    print(" FASE 2 - PASO 1: GENERACIÓN Y DATA AUGMENTATION (OpenCV)")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Buscar carpeta de raw_base
    possible_raw_dirs = [
        os.path.join(script_dir, "raw_base"),
        os.path.join(os.path.dirname(script_dir), "training", "raw_base"),
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), "training", "raw_base")
    ]
    raw_base_dir = None
    for p in possible_raw_dirs:
        if os.path.exists(p) and any(os.path.exists(os.path.join(p, c)) for c in CLASSES):
            raw_base_dir = p
            break

    if raw_base_dir is None:
        raise FileNotFoundError(f"No se encontró carpeta raw_base válida en: {possible_raw_dirs}")

    output_dir = os.path.join(script_dir, "dataset_clinico")
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    print(f"[1] Carpeta origen de imágenes base: {raw_base_dir}")
    print(f"[2] Carpeta destino del dataset:     {output_dir}")

    total_generadas = 0

    for clase in CLASSES:
        clase_raw_dir = os.path.join(raw_base_dir, clase)
        clase_train_dir = os.path.join(train_dir, clase)
        clase_val_dir = os.path.join(val_dir, clase)

        os.makedirs(clase_train_dir, exist_ok=True)
        os.makedirs(clase_val_dir, exist_ok=True)

        # Buscar imágenes base
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        base_files = []
        for ext in extensions:
            base_files.extend(glob.glob(os.path.join(clase_raw_dir, ext)))

        if not base_files:
            raise FileNotFoundError(f"No se encontraron imágenes base en {clase_raw_dir}")

        print(f"\nProcesando clase '{clase}': {len(base_files)} imagen(es) base detectada(s)...")

        # Cargar imágenes base en memoria
        base_images = [cv2.imread(f) for f in base_files if cv2.imread(f) is not None]
        if not base_images:
            raise ValueError(f"No se pudo decodificar ninguna imagen de {clase_raw_dir}")

        num_train = int(SAMPLES_PER_CLASS * TRAIN_RATIO)  # 40
        num_val = SAMPLES_PER_CLASS - num_train           # 10

        # Generar 40 imágenes para train
        for i in range(num_train):
            src_img = base_images[i % len(base_images)]
            aug_img = augment_by_class(src_img, clase)
            dst_path = os.path.join(clase_train_dir, f"{clase.lower()}_train_{i+1:03d}.jpg")
            cv2.imwrite(dst_path, aug_img)

        # Generar 10 imágenes para val
        for j in range(num_val):
            src_img = base_images[(j + num_train) % len(base_images)]
            aug_img = augment_by_class(src_img, clase)
            dst_path = os.path.join(clase_val_dir, f"{clase.lower()}_val_{j+1:03d}.jpg")
            cv2.imwrite(dst_path, aug_img)

        print(f"  -> Generadas: 40 train + 10 val = 50 imágenes ({clase})")
        total_generadas += SAMPLES_PER_CLASS

    print("\n" + "=" * 70)
    print(f"RESUMEN FINAL DEL DATASET CLÍNICO (YOLOv8-cls):")
    print(f"  - Total de imágenes generadas: {total_generadas}")
    print(f"  - Conjunto de Entrenamiento (train): 160 imágenes (40 por clase)")
    print(f"  - Conjunto de Validación (val):       40 imágenes (10 por clase)")
    print(f"  - Resolución fija: {IMG_SIZE[0]}x{IMG_SIZE[1]} px")
    print(f"  - Ubicación: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
