# 🦠 Classification COVID-19 avec Transfer Learning

> Projet de Deep Learning utilisant EfficientNet pour la classification d'images radiologiques COVID-19

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Ce projet implémente un modèle de classification d'images médicales basé sur le **Transfer Learning** avec l'architecture **EfficientNet**. L'objectif est de classifier automatiquement des images radiologiques pour détecter le COVID-19 et d'autres pathologies pulmonaires.

**Techniques clés :**
- Transfer Learning (pré-entraînement ImageNet)
- Fine-tuning progressif
- Data Augmentation avancée
- Grad-CAM pour l'interprétabilité
- Optimisation des hyperparamètres

## 👥 Équipe

| Membre | Rôle |
|--------|------|
| **LEMKHARBECH Yahya** | Chef de projet, Training pipeline |
| **ARGANE Mohammed Rida** | Architecture du modèle, Fine-tuning |
| **EL AOUMARI Abdelmoughith** | Exploration des données, Évaluation |
| **WAAZIZ Othmane** | Preprocessing, Visualisations |

## 🗂️ Dataset

**Source :** [COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) (Kaggle)

**Description :** Images radiologiques classifiées en plusieurs catégories (COVID, Normal, Pneumonie virale, etc.)

## 🚀 Quick Start
```bash
# 1. Cloner le repository
git clone https://github.com/YahyaLem02/covid19-efficientnet-transfer-learning.git
cd covid19-efficientnet-transfer-learning

# 2. Configurer l'environnement
bash scripts/setup_env.sh

# 3. Télécharger le dataset
bash scripts/download_dataset.sh

# 4. Explorer les données
jupyter notebook notebooks/01_exploration_donnees.ipynb

# 5. Entraîner le modèle
python src/training.py --config config/config.yaml

# 6. Évaluer les performances
python src/evaluation.py --model models/saved_models/best_model.h5
```

## 📊 Structure du Projet
```
.
├── data/                   # Données (raw, processed, external)
├── notebooks/              # Notebooks Jupyter (exploration, training, eval)
├── src/                    # Code source Python
│   ├── data_processing.py  # Preprocessing et augmentation
│   ├── model.py            # Architecture EfficientNet
│   ├── training.py         # Entraînement du modèle
│   ├── evaluation.py       # Évaluation et métriques
│   └── utils.py            # Fonctions utilitaires
├── models/                 # Modèles entraînés et checkpoints
├── results/                # Résultats (figures, métriques, logs)
├── config/                 # Fichiers de configuration (YAML)
├── docs/                   # Documentation (rapport, présentation)
└── scripts/                # Scripts utilitaires (setup, download)
```

## 🎯 Objectifs

* ✅ Accuracy > 90% sur le test set
* ✅ F1-Score > 0.88
* ✅ Implémentation Grad-CAM pour l'interprétabilité
* ✅ Comparaison de plusieurs variantes (B0, B1, B3)
* ✅ Documentation complète et reproductibilité

## 🛠️ Technologies

**Frameworks :** TensorFlow/Keras • PyTorch (optionnel)  
**Data Science :** NumPy • Pandas • Scikit-learn  
**Visualisation :** Matplotlib • Seaborn • Plotly  
**Outils :** Jupyter • Git • Docker (optionnel)

## 📈 Résultats

Les résultats détaillés seront mis à jour après l'entraînement des modèles.

| Modèle | Accuracy | F1-Score | AUC | Temps |
|--------|----------|----------|-----|-------|
| EfficientNet-B0 | - | - | - | - |
| EfficientNet-B1 | - | - | - | - |
| EfficientNet-B3 | - | - | - | - |

## 📝 Documentation

* 📄 **Rapport technique** (à venir)
* 🎤 **Présentation** (à venir)
* 📓 **Notebooks interactifs**

## 🤝 Contribution

Ce projet est développé dans le cadre d'un projet académique sur le Transfer Learning. Contributions et suggestions sont les bienvenues via Issues ou Pull Requests.

## 📄 Licence

Ce projet est sous licence MIT. Voir **LICENSE** pour plus de détails.

## 🙏 Remerciements

* Dataset : Pranav Raikokte (Kaggle)
* Architecture EfficientNet : Google Research
* Communauté TensorFlow/Keras

## 📧 Contact

GitHub : **@YahyaLem02**