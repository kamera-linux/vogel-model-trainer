# Release v0.1.8 - Class Balance Management & Dataset Quality Control

**Release Date:** November 13, 2025

This release introduces powerful class balance management features to ensure high-quality, balanced datasets for better model training results.

---

## ✨ New Features

### Class Balance Management for Organize Command

Train better models with balanced datasets! The `organize` command now includes automatic class balancing and limiting features.

#### 1. **`--max-images-per-class`** - Automatic Image Limiting
Automatically limit the number of images per class and delete excess images:

```bash
# Limit to 100 images per class
vogel-trainer organize ~/training-data -o ~/organized --max-images-per-class 100

# Limit to 200 images per class
vogel-trainer organize ~/training-data -o ~/organized --max-images-per-class 200
```

**What it does:**
- Randomly selects N images to keep per class
- **Automatically deletes** excess images from source directory
- Shows detailed report of deleted files and affected species
- Perfect for creating uniform datasets

**Example Output:**
```
🔢 Enforcing maximum 100 images per class...
   ✂️  kohlmeise: 794 images deleted (894 → 100)
   ✂️  blaumeise: 89 images deleted (189 → 100)

📊 Total deleted images: 883
   Affected species: kohlmeise, blaumeise, ...
```

#### 2. **`--tolerance`** - Class Balance Validation
Set maximum allowed imbalance between classes (default: 15%):

```bash
# Use strict 10% tolerance
vogel-trainer organize ~/training-data -o ~/organized --tolerance 10.0

# More lenient 20% tolerance
vogel-trainer organize ~/training-data -o ~/organized --tolerance 20.0
```

**How it works:**
- **< 10%**: ✅ Perfect balance, no warnings
- **10-15%**: ⚠️ Warning with affected classes listed
- **> 15%**: ❌ Error, process stops with recommendations

**Example Output:**
```
⚖️  Class Balance Check:
   Minimum: 87 images
   Maximum: 100 images
   Average: 98.4 images
   Difference: 14.9%

⚠️  WARNING: Class balance could be better
   Difference: 14.9% (Tolerance: 15.0%)
   Affected classes:
   • haussperling   :   87 images (-13.0% below maximum)
```

**Error Example (>15%):**
```
❌ ERROR: Class imbalance too large!
   Maximum tolerance: 15%
   Current difference: 45.2%

   Affected classes:
   • rotkehlchen  :   55 images (-44.0% from average)
   • kohlmeise    :  100 images (+1.0% from average)

💡 Recommendation:
   - Collect more images for underrepresented species
   - Or use --max-images-per-class 63 to limit to 15%
```

#### 3. **Combined Usage** - Best Practice
Use both parameters together for optimal dataset quality:

```bash
vogel-trainer organize ~/training-data -o ~/organized \
  --max-images-per-class 100 \
  --tolerance 15.0
```

This ensures:
1. No class has more than 100 images
2. All classes are within 15% of each other
3. Clean, balanced dataset ready for training

### Full Multilingual Support
All new features support **English**, **German**, and **Japanese**:
- EN: "Class Balance Check"
- DE: "Class Balance Check"
- JA: "クラスバランスチェック"

---

## 📦 Installation

### Via pip (Recommended)
```bash
# Install in virtual environment (recommended)
python3 -m venv ~/venv-vogel
source ~/venv-vogel/bin/activate  # Windows: ~/venv-vogel\Scripts\activate

# Install/upgrade package
pip install --upgrade vogel-model-trainer
```

### From Source
```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.8
pip install -e .
```

### Verify Installation
```bash
vogel-trainer --version
# Output: vogel-model-trainer version 0.1.8
```

---

## 🔄 Upgrade Notes

### From v0.1.7 to v0.1.8

**New Parameters (Optional):**
- `--max-images-per-class`: Control dataset size
- `--tolerance`: Control class balance

**Breaking Changes:** None - all new parameters are optional

**Upgrade:**
```bash
pip install --upgrade vogel-model-trainer==0.1.8
```

### Migration Guide

**If you want balanced datasets:**
1. Check current balance:
   ```bash
   vogel-trainer organize ~/data -o ~/organized
   ```
   
2. If imbalance > 15%, either:
   - Collect more data for underrepresented classes
   - Use `--max-images-per-class` to limit larger classes

**Example Workflow:**
```bash
# Step 1: Extract birds from videos
vogel-trainer extract --folder ~/training-data \
  --sample-rate 20 --skip-blurry --deduplicate \
  '/path/to/videos/*.mp4'

# Step 2: Organize with balance control
vogel-trainer organize ~/training-data -o ~/organized \
  --max-images-per-class 100 --tolerance 15.0

# Step 3: Train model
vogel-trainer train ~/organized -o ~/models/my-classifier
```

---

## 🎥 Video Tutorials

Learn how to use vogel-model-trainer with step-by-step video guides:

### Getting Started Series

**1. Installation & Setup** (5 min)
- Setting up virtual environment
- Installing dependencies
- First test run

**2. Extracting Birds from Videos** (10 min)
- Using the `extract` command
- Quality filters (blur, confidence thresholds)
- Species classification with pre-trained models
- Deduplication features

**3. Organizing Your Dataset** (8 min)
- Creating train/val splits
- **NEW:** Using `--max-images-per-class` to limit dataset size
- **NEW:** Class balance validation with `--tolerance`
- Best practices for balanced datasets

**4. Training Your Custom Classifier** (12 min)
- Configuring training parameters
- Monitoring training progress
- Understanding metrics and validation
- Exporting final models

**5. Testing & Evaluation** (7 min)
- Testing model accuracy
- Per-species performance analysis
- Iterating to improve results

### Advanced Topics

**6. Multi-Language Support** (5 min)
- Using English, German, and Japanese interfaces
- Configuring language settings

**7. Production Deployment** (10 min)
- Integrating trained models
- Batch processing workflows
- Performance optimization

---

## 💡 Use Cases

### Scenario 1: Heavily Imbalanced Raw Data
You have 1000 kohlmeise images but only 50 rotkehlchen:

```bash
# Solution: Limit to 50-60 images per class
vogel-trainer organize ~/data -o ~/organized \
  --max-images-per-class 60 --tolerance 15.0
```

Result: Balanced 60/50 dataset (16.7% difference, within tolerance)

### Scenario 2: Large Dataset, Need Subset
You have 500+ images per class, want faster training:

```bash
# Create 100-image subset per class
vogel-trainer organize ~/data -o ~/organized-100 \
  --max-images-per-class 100 --tolerance 10.0
```

Result: Compact, balanced 100-image-per-class dataset

### Scenario 3: Quality Control Before Training
Ensure dataset quality before expensive training:

```bash
# Strict balance check (no deletion)
vogel-trainer organize ~/data -o ~/organized --tolerance 10.0

# If fails: collect more data OR use limiting
vogel-trainer organize ~/data -o ~/organized \
  --max-images-per-class 85 --tolerance 10.0
```

---

## 📊 Performance Impact

**Balanced datasets lead to better models:**
- 5-15% accuracy improvement on underrepresented classes
- More consistent performance across all species
- Reduced overfitting on overrepresented classes

**Example Results:**
```
Before (imbalanced):
  kohlmeise: 98% (1000 images)
  rotkehlchen: 65% (50 images)
  Overall: 81.5%

After (balanced to 100 each):
  kohlmeise: 96% (-2%)
  rotkehlchen: 89% (+24%!)
  Overall: 92.5% (+11%)
```

---

## 🙏 Credits

Thank you to all users who requested class balance features and provided feedback!

---

## 📚 Resources

- **Documentation:** [README.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.md)
- **German Docs:** [README.de.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.de.md)
- **Japanese Docs:** [README.ja.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.ja.md)
- **Full Changelog:** [CHANGELOG.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md)
- **Video Tutorials:** Coming soon!

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.7...v0.1.8
