# 🐦 Vogel Model Trainer

**Sprachen:** [🇬🇧 English](README.md) | [🇩🇪 Deutsch](README.de.md)

<p align="left">
  <a href="https://pypi.org/project/vogel-model-trainer/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/vogel-model-trainer.svg"></a>
  <a href="https://pypi.org/project/vogel-model-trainer/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/vogel-model-trainer.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://pypi.org/project/vogel-model-trainer/"><img alt="PyPI Status" src="https://img.shields.io/pypi/status/vogel-model-trainer.svg"></a>
  <a href="https://pepy.tech/project/vogel-model-trainer"><img alt="Downloads" src="https://static.pepy.tech/badge/vogel-model-trainer"></a>
</p>

**Trainiere eigene Vogelarten-Klassifizierer aus deinen eigenen Video-Aufnahmen mit YOLOv8 und EfficientNet.**

Ein spezialisiertes Toolkit zum Erstellen von hochgenauen Vogelarten-Klassifizierern, die auf dein spezifisches Monitoring-Setup zugeschnitten sind. Extrahiere Trainingsdaten aus Videos, organisiere Datasets und trainiere eigene Modelle mit >96% Genauigkeit.

---

## ✨ Features

- 🎯 **YOLO-basierte Vogelerkennung** - Automatisches Cropping von Vögeln aus Videos mit YOLOv8
- 🤖 **Drei Extraktions-Modi** - Manuelle Beschriftung, Auto-Sortierung oder Standard-Extraktion
- 📁 **Wildcard-Unterstützung** - Batch-Verarbeitung mehrerer Videos mit Glob-Patterns
- 🖼️ **Auto-Resize auf 224x224** - Optimale Bildgröße fürs Training
- 🧠 **EfficientNet-B0 Training** - Leichtgewichtiges aber leistungsstarkes Klassifizierungsmodell
- 🎨 **Erweiterte Data Augmentation** - Rotation, Affine-Transformationen, Color Jitter, Gaussian Blur
- 📊 **Optimiertes Training** - Cosine LR Scheduling, Label Smoothing, Early Stopping
- ⏸️ **Graceful Shutdown** - Modellzustand bei Strg+C-Unterbrechung speichern
- 🔄 **Iteratives Training** - Nutze trainierte Modelle zum Erweitern deines Datasets
- 📈 **Pro-Art-Metriken** - Detaillierte Genauigkeits-Aufschlüsselung pro Vogelart

---

## 🚀 Schnellstart

### Installation

```bash
# Installation von PyPI
pip install vogel-model-trainer

# Oder Installation aus Quellcode
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
pip install -e .
```

### Grundlegender Workflow

```bash
# 1. Vogelbilder aus Videos extrahieren
vogel-trainer extract video.mp4 --bird kohlmeise --output ~/training-data/

# 2. In Train/Validation Split organisieren
vogel-trainer organize --source ~/training-data/ --output ~/training-data/organized/

# 3. Eigenen Klassifizierer trainieren
vogel-trainer train --data ~/training-data/organized/ --output ~/models/

# 4. Das trainierte Modell testen
vogel-trainer test ~/models/final/ test_image.jpg
```

---

## 📖 Nutzungsanleitung

### 1. Trainingsbilder extrahieren

#### Manueller Modus (Empfohlen für erste Sammlung)

Wenn du die Art in deinem Video kennst:

```bash
vogel-trainer extract ~/Videos/kohlmeise-*.mp4 \
  --bird kohlmeise \
  --output ~/training-data/ \
  --threshold 0.5 \
  --sample-rate 3
```

#### Auto-Sort Modus (Für iteratives Training)

Nutze ein bestehendes Modell zum automatischen Klassifizieren und Sortieren:

```bash
vogel-trainer extract ~/Videos/gemischt-*.mp4 \
  --species-model ~/models/classifier/final/ \
  --output ~/training-data/ \
  --threshold 0.6
```

#### Batch-Verarbeitung mit Wildcards

```bash
# Alle Videos in einem Verzeichnis verarbeiten
vogel-trainer extract "~/Videos/*.mp4" --bird blaumeise --output ~/data/

# Rekursive Verzeichnis-Suche
vogel-trainer extract ~/Videos/ \
  --species-model ~/models/classifier/final/ \
  --output ~/data/ \
  --recursive
```

**Parameter:**
- `--threshold`: YOLO Confidence-Schwellwert (Standard: 0.5)
- `--sample-rate`: Verarbeite jeden N-ten Frame (Standard: 3)
- `--bird`: Manuelle Arten-Beschriftung
- `--species-model`: Pfad zu trainiertem Modell für Auto-Klassifizierung
- `--no-resize`: Originalgröße beibehalten (Standard: Resize auf 224x224)

### 2. Dataset organisieren

```bash
vogel-trainer organize \
  --source ~/training-data/ \
  --output ~/training-data/organized/ \
  --train-ratio 0.8
```

Erstellt einen 80/20 Train/Validation Split:
```
organized/
├── train/
│   ├── kohlmeise/
│   ├── blaumeise/
│   └── rotkehlchen/
└── val/
    ├── kohlmeise/
    ├── blaumeise/
    └── rotkehlchen/
```

### 3. Klassifizierer trainieren

```bash
vogel-trainer train \
  --data ~/training-data/organized/ \
  --output ~/models/ \
  --epochs 50 \
  --batch-size 16
```

**Training-Konfiguration:**
- Basis-Modell: `google/efficientnet-b0` (8.5M Parameter)
- Optimizer: AdamW mit Cosine LR Schedule
- Augmentation: Rotation, Affine, Color Jitter, Gaussian Blur
- Regularisierung: Weight Decay 0.01, Label Smoothing 0.1
- Early Stopping: Patience von 7 Epochen

**Output:**
```
~/models/bird-classifier-20251108_143000/
├── checkpoints/     # Zwischencheckpoints
├── logs/           # TensorBoard Logs
└── final/          # Finales trainiertes Modell
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

### 4. Modell testen

```bash
# Test auf einzelnem Bild
vogel-trainer test ~/models/final/ image.jpg

# Output:
# 🖼️  Testing: image.jpg
#    🐦 Predicted: kohlmeise (98.5% confidence)
```

---

## 🔄 Iterativer Training-Workflow

Verbessere dein Modell durch iteratives Erweitern deines Datasets:

```bash
# 1. Initiales Training mit manuellen Labels
vogel-trainer extract ~/Videos/batch1/*.mp4 --bird kohlmeise --output ~/data/
vogel-trainer organize --source ~/data/ --output ~/data/organized/
vogel-trainer train --data ~/data/organized/ --output ~/models/v1/

# 2. Nutze trainiertes Modell um mehr Daten zu extrahieren
vogel-trainer extract ~/Videos/batch2/*.mp4 \
  --species-model ~/models/v1/final/ \
  --output ~/data/iteration2/

# 3. Review und korrigiere Fehlklassifizierungen manuell
# Verschiebe falsche Vorhersagen in korrekte Arten-Ordner

# 4. Kombiniere Datasets und trainiere neu
cp -r ~/data/iteration2/* ~/data/
vogel-trainer organize --source ~/data/ --output ~/data/organized/
vogel-trainer train --data ~/data/organized/ --output ~/models/v2/

# Ergebnis: Höhere Genauigkeit! 🎉
```

---

## 📊 Performance & Best Practices

### Empfehlungen zur Dataset-Größe

| Qualität | Bilder pro Art | Erwartete Genauigkeit |
|----------|----------------|----------------------|
| Minimum  | 20-30         | ~85-90%             |
| Gut      | 50-100        | ~92-96%             |
| Optimal  | 100+          | >96%                |

### Tipps für bessere Ergebnisse

1. **Dataset-Diversität**
   - Verschiedene Lichtverhältnisse einbeziehen
   - Verschiedene Posen erfassen (Seite, Vorne, Hinten)
   - Verschiedene Jahreszeiten abdecken (Federkleid ändert sich)

2. **Klassen-Balance**
   - Ähnliche Bildzahl pro Art anstreben
   - Vermeide eine dominierende Klasse

3. **Qualität vor Quantität**
   - Nutze Threshold 0.5-0.6 für klare Detektionen
   - Manuelle Review von auto-sortierten Bildern verbessert Qualität

4. **Training monitoren**
   - Prüfe Pro-Klassen-Genauigkeit für schwache Arten
   - Nutze Confusion Matrix um ähnliche Arten zu identifizieren
   - Füge mehr Daten für schlecht performende Klassen hinzu

---

## 🔗 Integration mit vogel-video-analyzer

Nutze dein trainiertes Modell zur Artenerkennung:

```bash
vogel-analyze --identify-species \
  --species-model ~/models/final/ \
  --species-threshold 0.3 \
  video.mp4
```

---

## 🛠️ Entwicklung

```bash
# Repository klonen
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer

# Im Entwicklungsmodus installieren
pip install -e ".[dev]"

# Tests ausführen
pytest tests/
```

---

## 📝 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

---

## 🙏 Credits

- **YOLO** von [Ultralytics](https://github.com/ultralytics/ultralytics)
- **EfficientNet** von [Google Research](https://github.com/google/automl)
- **Transformers** von [Hugging Face](https://huggingface.co/transformers)

---

## 📮 Support & Contributing

- **Issues**: [GitHub Issues](https://github.com/kamera-linux/vogel-model-trainer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kamera-linux/vogel-model-trainer/discussions)
- **Pull Requests**: Contributions willkommen!

---

Made with ❤️ for bird watching enthusiasts 🐦
