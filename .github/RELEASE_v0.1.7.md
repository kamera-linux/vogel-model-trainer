# Release v0.1.7 - Bugfix: Cross-Session Deduplication

**Release Date:** November 10, 2025

This is a bugfix release that resolves an important issue with the deduplication feature introduced in v0.1.6.

---

## 🐛 Bug Fixes

### Fixed Cross-Session Deduplication
**Problem:** The `--deduplicate` flag only prevented duplicates within a single extraction session. Running the same `extract` command multiple times would re-extract and save identical bird images, even though they already existed in the output directory.

**Solution:** The extractor now pre-loads all existing images from the output directory at startup and computes their perceptual hashes. These are stored in the hash cache before extraction begins, ensuring that duplicates are detected across multiple command runs.

**User Experience:**
```bash
vogel-trainer extract --folder ~/output --deduplicate ...
```

You'll now see:
```
   🔄 Lade existierende Bilder für Deduplizierung...
   ✅ 23 existierende Bilder in Hash-Cache geladen
```

Running the same command again will skip all previously extracted birds:
```
   ⏭️  Duplikat übersprungen: ähnlich zu image_001.jpg (Distanz: 2)
   ⏭️  Duplikat übersprungen: ähnlich zu image_002.jpg (Distanz: 1)
```

---

## 📦 Installation

### Via pip (Recommended)
```bash
pip install --upgrade vogel-model-trainer
```

### From Source
```bash
git clone https://github.com/kamera-linux/vogel-model-trainer.git
cd vogel-model-trainer
git checkout v0.1.7
pip install -e .
```

### Verify Installation
```bash
vogel-trainer --version
# Output: vogel-model-trainer version 0.1.7
```

---

## 🔄 Upgrade Notes

### From v0.1.6 to v0.1.7

This is a **drop-in replacement** with no breaking changes. Simply upgrade:

```bash
pip install --upgrade vogel-model-trainer==0.1.7
```

**What Changes:**
- Deduplication now works correctly across multiple extraction runs
- No changes to command-line interface or configuration
- No migration needed for existing datasets

**Recommendation:** If you've been using `--deduplicate` in v0.1.6, you may want to review your output directories for potential duplicates that were saved before this fix. Use the `deduplicate` command to clean them:

```bash
vogel-trainer deduplicate ~/vogel-training-data-species --mode report
vogel-trainer deduplicate ~/vogel-training-data-species --mode delete  # if duplicates found
```

---

## 🙏 Credits

Thank you to all users who reported this issue and helped test the fix!

---

## 📚 Resources

- **Documentation:** [README.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.md)
- **German Docs:** [README.de.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.de.md)
- **Japanese Docs:** [README.ja.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/README.ja.md)
- **Full Changelog:** [CHANGELOG.md](https://github.com/kamera-linux/vogel-model-trainer/blob/main/CHANGELOG.md)

---

**Full Changelog**: https://github.com/kamera-linux/vogel-model-trainer/compare/v0.1.6...v0.1.7
