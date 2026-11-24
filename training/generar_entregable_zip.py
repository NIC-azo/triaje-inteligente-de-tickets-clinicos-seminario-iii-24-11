# -*- coding: utf-8 -*-
"""
training/generar_entregable_zip.py
Genera el paquete zip entregable del proyecto Triaje Multimodal Clínico.
Excluye: node_modules, __pycache__, .git, runs, archivos .env con secretos locales.
Incluye: backend, frontend, training, modelos compilados, datasets, documentación técnica y bitácora.
"""

import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = PROJECT_ROOT.parent / "entregable_triaje_multimodal_clinico.zip"
BITACORA_MD = PROJECT_ROOT.parent / "registro_ejecucion_fases.md"

EXCLUDE_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    "runs",
    ".vscode",
    ".idea",
    "dist"
}

EXCLUDE_FILES = {
    ".env",
    ".DS_Store",
    "Thumbs.db"
}

def should_exclude(rel_path_str: str) -> bool:
    parts = rel_path_str.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    filename = parts[-1]
    if filename in EXCLUDE_FILES:
        return True
    if filename.endswith(".pyc"):
        return True
    return False

def main():
    print(f"[*] Iniciando empaquetado del proyecto desde: {PROJECT_ROOT}")
    print(f"[*] Destino del entregable: {OUTPUT_ZIP}")

    total_files = 0
    total_bytes = 0

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        # 1. Empaquetar todo el directorio triaje-multimodal-clinico
        for root, dirs, files in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT)
                rel_path_str = str(rel_path)

                if should_exclude(rel_path_str):
                    continue

                # Guardar dentro de carpeta triaje-multimodal-clinico/ en el zip
                archive_name = f"triaje-multimodal-clinico/{rel_path_str.replace(chr(92), '/')}"
                zipf.write(file_path, arcname=archive_name)
                total_files += 1
                total_bytes += file_path.stat().st_size

        # 2. Agregar la bitácora registro_ejecucion_fases.md si existe en la raíz superior
        if BITACORA_MD.exists():
            zipf.write(BITACORA_MD, arcname="registro_ejecucion_fases.md")
            total_files += 1
            total_bytes += BITACORA_MD.stat().st_size

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"[+] Empaquetado completado exitosamente.")
    print(f"    - Archivos empaquetados: {total_files}")
    print(f"    - Tamaño original: {total_bytes / (1024 * 1024):.2f} MB")
    print(f"    - Tamaño final del .zip: {zip_size_mb:.2f} MB")
    print(f"    - Ruta del entregable: {OUTPUT_ZIP}")

if __name__ == "__main__":
    main()
