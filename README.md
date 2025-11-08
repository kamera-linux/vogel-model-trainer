# Vogel Model Trainer

🐦 **Train custom bird species classifiers from your own video footage**

[![PyPI version](https://badge.fury.io/py/vogel-model-trainer.svg)](https://pypi.org/project/vogel-model-trainer/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Was ist das?

`vogel-model-trainer` ist ein spezialisiertes Tool zum Trainieren von Custom Bird Species Classifiers aus eigenen Vogelhaus-Videos. Perfekt für:

- 🏠 Vogelhaus-Monitoring mit spezifischen europäischen Arten
- 📹 Extraktion von Trainingsdaten aus Videos
- 🤖 Training mit EfficientNet auf deinen eigenen Daten
- 🎓 Iteratives Training für >96% Accuracy

## 🚀 Installation

```bash
pip install vogel-model-trainer
```

### Development Installation

```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
pip install -e ".[dev]"
```

## 📋 Quick Start

### 1. Bilder aus Videos extrahieren

```bash
# Manuelle Sortierung (wenn du die Art kennst)
vogel-trainer extract video.mp4 \
  --bird kohlmeise \
  --output ~/training-data/ \
  --threshold 0.5

# Auto-Sortierung mit bestehendem Modell
vogel-trainer extract "videos/*.mp4" \
  --species-model ~/models/classifier/ \
  --output ~/training-data/ \
  --recursive
```

### 2. Dataset organisieren

```bash
vogel-trainer organize \
  --source ~/training-data/ \
  --output ~/training-data/organized/ \
  --train-ratio 0.8
```

### 3. Modell trainieren

```bash
vogel-trainer train \
  --data ~/training-data/organized/ \
  --output ~/models/ \
  --epochs 50 \
  --batch-size 16
```

### 4. Modell testen

```bash
vogel-trainer test ~/models/final/ image.jpg
```

## 🛠️ Features

- ✅ **YOLO-basierte Bird Detection** mit automatischem Cropping
- ✅ **3 Extraktions-Modi**: Manual, Auto-Sort, Standard
- ✅ **Wildcard & Recursive Processing** für Batch-Operationen
- ✅ **Automatic 224x224 Resizing** für optimales Training
- ✅ **EfficientNet-B0** als Basis-Modell (8.5M Parameter)
- ✅ **Enhanced Data Augmentation** (Rotation, Affine, ColorJitter, GaussianBlur)
- ✅ **Optimized Training** (Cosine LR, Label Smoothing, Early Stopping)
- ✅ **Graceful Shutdown** mit Strg+C und Modell-Speicherung
- ✅ **Automatic Species Detection** aus Verzeichnis-Struktur

## 📚 Workflow-Beispiel

### Erste Datensammlung

```bash
# Extrahiere Bilder von verschiedenen Arten
vogel-trainer extract ~/Videos/kohlmeise*.mp4 --bird kohlmeise --output ~/data/
vogel-trainer extract ~/Videos/blaumeise*.mp4 --bird blaumeise --output ~/data/
vogel-trainer extract ~/Videos/rotkehlchen*.mp4 --bird rotkehlchen --output ~/data/

# Organisiere Dataset
vogel-trainer organize --source ~/data/ --output ~/data/organized/

# Trainiere Modell
vogel-trainer train --data ~/data/organized/ --output ~/models/
```

### Iteratives Training

```bash
# 1. Nutze trainiertes Modell für Auto-Extraktion
vogel-trainer extract ~/Videos/neue_videos/ \
  --species-model ~/models/bird-classifier-*/final/ \
  --output ~/data/iteration2/ \
  --recursive

# 2. Review und kombiniere Daten
# (Manuelles Verschieben falscher Klassifizierungen)

# 3. Neues Training mit erweiterten Daten
vogel-trainer organize --source ~/data/combined/
vogel-trainer train --data ~/data/combined/organized/
```

## 🎓 Training-Konfiguration

**Optimierte Hyperparameter:**
- Basis-Modell: `google/efficientnet-b0`
- Epochs: 50 (Early Stopping nach 7 Epochen)
- Batch Size: 16
- Learning Rate: 2e-4 mit Cosine Annealing
- Image Size: 224x224px
- Weight Decay: 0.01
- Label Smoothing: 0.1

**Data Augmentation:**
- RandomResizedCrop (70-100% scale)
- RandomRotation (±15°)
- RandomAffine (10% translation)
- ColorJitter (Brightness/Contrast/Saturation)
- GaussianBlur (Fokus-Variationen)

## 📊 Erfahrungswerte

### Datenmengen
- **Minimum:** ~20-30 Bilder pro Art
- **Gut:** ~50-100 Bilder pro Art  
- **Optimal:** 100+ Bilder pro Art

### Performance
- **Validation Accuracy:** 96%+ bei guten Daten
- **Training Zeit:** ~3-4 Stunden (500 Bilder, 5 Arten, Raspberry Pi 5)
- **Per-Species Accuracy:** 95-100% bei ausgewogenen Klassen

## 🔗 Integration

### Mit vogel-video-analyzer

```bash
# Nutze trainiertes Modell in vogel-analyze
vogel-analyze --identify-species \
  --species-model ~/models/final/ \
  --species-threshold 0.3 \
  video.mp4
```

## 📖 Dokumentation

Ausführliche Dokumentation und Beispiele findest du im [GitHub Repository](https://github.com/kamera-linux/vogel-model-trainer).

## 🤝 Contributing

Contributions sind willkommen! Bitte erstelle einen Pull Request oder öffne ein Issue.

## 📝 License

MIT License - siehe [LICENSE](LICENSE) für Details.

## 🙏 Credits

- YOLO von [Ultralytics](https://github.com/ultralytics/ultralytics)
- EfficientNet von [Google](https://github.com/google/automl)
- Transformers von [Hugging Face](https://huggingface.co/transformers)

---

Made with ❤️ for bird watching enthusiasts 🐦
