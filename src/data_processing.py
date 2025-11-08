"""
src/data_processing.py

Module de préparation des données pour le projet COVID-19 EfficientNet.

Fonctionnalités :
- Récupération des chemins d'images organisés par classe depuis data/raw/Covid19-dataset
- Statistiques sur les images (résolution moyenne, formats, tailles)
- Création des splits train/val/test (copie dans data/processed/)
- Génération d'ImageDataGenerator (Keras) pour entraînement/validation
- Calcul des class weights pour gérer le déséquilibre

Usage (exemples) :
from src.data_processing import (
    gather_image_paths,
    get_class_distribution,
    compute_image_statistics,
    create_splits,
    get_generators,
    compute_class_weights
)
"""
import os
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Optionnel : Keras / TensorFlow ImageDataGenerator
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    ImageDataGenerator = None  # handled downstream


def gather_image_paths(raw_root: str = "data/raw/Covid19-dataset") -> Dict[str, List[str]]:
    """
    Parcourt data/raw/Covid19-dataset et renvoie un dict {class_name: [filepaths]}.

    raw_root: chemin vers le dossier racine du dataset (chaque sous-dossier correspond à une classe)
    """
    raw_root = Path(raw_root)
    classes = {}
    if not raw_root.exists():
        raise FileNotFoundError(f"Le dossier raw n'existe pas: {raw_root}")
    for sub in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        class_name = sub.name
        files = [str(p) for p in sub.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        classes[class_name] = sorted(files)
    return classes


def get_class_distribution(paths_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Retourne un DataFrame avec le nombre d'images par classe.
    """
    data = [{"class": k, "n_images": len(v)} for k, v in paths_dict.items()]
    df = pd.DataFrame(data).sort_values("n_images", ascending=False).reset_index(drop=True)
    return df


def compute_image_statistics(paths: List[str]) -> pd.DataFrame:
    """
    Pour une liste de chemins d'images, calcule:
    - width, height
    - format (PNG, JPEG)
    - file_size_bytes

    Retourne un DataFrame.
    """
    records = []
    for p in paths:
        try:
            with Image.open(p) as img:
                w, h = img.size
                fmt = img.format or Path(p).suffix.replace(".", "").upper()
            size = Path(p).stat().st_size
            records.append({"path": p, "width": w, "height": h, "format": fmt, "file_size": size})
        except Exception as e:
            # Ignorer les images corrompues mais enregistrer l'erreur
            records.append({"path": p, "width": None, "height": None, "format": None, "file_size": None, "error": str(e)})
    df = pd.DataFrame(records)
    return df


def summarize_image_stats(df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Donne des statistiques descriptives (moyenne, médiane, min, max) pour width, height, file_size et counts par format.
    """
    numeric = df_stats[["width", "height", "file_size"]].dropna()
    stats = numeric.agg(["count", "mean", "median", "min", "max"]).transpose()
    fmt_counts = df_stats["format"].value_counts(dropna=True).rename_axis("format").reset_index(name="count")
    return {"numeric_stats": stats, "format_counts": fmt_counts}


def create_splits(
    paths_dict: Dict[str, List[str]],
    output_root: str = "data/processed",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    copy_files: bool = True,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Crée des splits train/val/test et (optionnel) copie les fichiers dans data/processed/{train,val,test}/{class}/

    Retourne une structure dict:
    {
      "train": {"classA": [paths], ...},
      "val": {...},
      "test": {...}
    }
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Les tailles doivent sommer à 1.0"

    out = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    output_root = Path(output_root)
    for cls, files in paths_dict.items():
        if len(files) == 0:
            continue
        # split train vs temp
        train_files, temp_files = train_test_split(files, train_size=train_size, random_state=random_state, stratify=None)
        if val_size == 0 and test_size == 0:
            val_files, test_files = [], []
        else:
            # proportion of val relative to temp
            rel_val = val_size / (val_size + test_size) if (val_size + test_size) > 0 else 0
            val_files, test_files = train_test_split(temp_files, train_size=rel_val, random_state=random_state, stratify=None)
        out["train"][cls] = sorted(train_files)
        out["val"][cls] = sorted(val_files)
        out["test"][cls] = sorted(test_files)

        if copy_files:
            for split in ("train", "val", "test"):
                for src in out[split][cls]:
                    dest_dir = output_root / split / cls
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_path = dest_dir / Path(src).name
                    if not dest_path.exists():
                        shutil.copy2(src, dest_path)

    # convert defaultdict to normal dicts
    out = {k: dict(v) for k, v in out.items()}
    return out


def compute_class_weights_from_split(split_dict: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calcule les class weights à partir d'un dictionnaire de split (ex: train split dict {class: [paths]})

    Retourne dict {class: weight}
    """
    classes = []
    for cls, files in split_dict.items():
        classes += [cls] * len(files)
    if len(classes) == 0:
        return {}
    labels = np.array(classes)
    unique = np.unique(labels)
    class_weights = compute_class_weight(class_weight="balanced", classes=unique, y=labels)
    return {int(i): w for i, w in enumerate(class_weights)} if False else dict(zip(unique.tolist(), class_weights.tolist()))


def get_generators(
    processed_root: str = "data/processed",
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    augment: bool = True,
    seed: int = 42,
):
    """
    Retourne (train_generator, val_generator, test_generator) utilisant Keras ImageDataGenerator.flow_from_directory

    Structure attendue:
    data/processed/train/<class>/*.jpg
    data/processed/val/<class>/*.jpg
    data/processed/test/<class>/*.jpg
    """
    if ImageDataGenerator is None:
        raise ImportError("TensorFlow/Keras non disponible. Installez tensorflow pour utiliser get_generators().")

    train_dir = os.path.join(processed_root, "train")
    val_dir = os.path.join(processed_root, "val")
    test_dir = os.path.join(processed_root, "test")

    # augmentation
    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=(0.8, 1.2),
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=seed,
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        seed=seed,
    )

    return train_generator, val_generator, test_generator


def sample_augmented_images(generator, n=9):
    """
    Retourne un batch d'images augmentées (n images) depuis un generator Keras.
    Utile pour visualiser l'effet d'augmentation dans un notebook.
    """
    batch_x, batch_y = next(generator)
    return batch_x[:n], batch_y[:n]


def save_summary_csv(distribution_df: pd.DataFrame, out_path: str = "results/metrics/data_distribution.csv"):
    """
    Sauvegarde le DataFrame de distribution en CSV.
    """
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    distribution_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # petit test local pour vérifier que le module fonctionne (ne s'exécute que si lancé directement)
    import pprint

    print("Test rapide du module data_processing.py")
    try:
        paths = gather_image_paths("data/raw/Covid19-dataset")
        dist = get_class_distribution(paths)
        pprint.pprint(dist.to_dict(orient="records"))
    except Exception as e:
        print("Erreur (attendue si le dataset n'est pas présent):", e)