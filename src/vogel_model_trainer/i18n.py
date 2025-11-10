"""
Internationalization (i18n) module for vogel-model-trainer
Provides translations for command-line output
"""

import os
import locale

# Available translations
TRANSLATIONS = {
    'en': {
        # Extraction
        'loading_yolo': '🤖 Loading YOLO model:',
        'loading_species': '🧠 Loading species classifier:',
        'loaded_species_classes': '   ✅ Loaded with {count} species classes',
        'video_info': '📹 Video:',
        'total_frames': '   📊 {total} frames, {fps:.1f} FPS',
        'analyzing_every_nth': '   🔍 Analyzing every {n}. frame...',
        'detection_threshold': '   🎯 Detection threshold: {threshold}',
        'species_threshold': '   🎯 Species threshold: {threshold}',
        'image_size': '   📐 Image size: {size}x{size}px',
        'image_size_original': '   📐 Image size: Original',
        'mode_autosorting': '   🤖 Auto-sorting mode: Using species classifier',
        'mode_manual': '   🏷️  Manual mode: Species = {species}',
        'mode_standard': '   📦 Standard mode: All birds in one directory',
        'cannot_open_video': '❌ Cannot open video: {path}',
        'bird_extracted': '   ✅ Bird #{count}: {species} (conf {conf:.2f}), frame {frame}',
        'bird_extracted_simple': '   ✅ Extracted bird #{count}: frame {frame}, conf {conf:.2f}',
        'bird_skipped': '   ⏭️  Skipped: {species} (conf {conf:.2f} < {threshold:.2f}), frame {frame}',
        'progress': '   ⏳ Progress: {percent:.1f}% ({current}/{total} frames)',
        'extraction_interrupted': '\n⚠️  Extraction interrupted by user',
        'extraction_complete': '\n✅ Extraction complete!',
        'output_directory': '   📁 Output directory: {path}',
        'detected_birds_total': '   🔍 Detected birds total: {count}',
        'exported_birds_total': '   🐦 Exported birds: {count}',
        'skipped_birds_total': '   ⏭️  Skipped (< {threshold:.2f}): {count}',
        'total_birds': '   🐦 Total birds extracted: {count}',
        'species_breakdown': '\n📊 Species breakdown:',
        'species_count': '   • {species}: {count} birds',
        'session_id': '   🆔 Session ID: {id}',
        'filename_format': '\n💡 Filename format: {format}',
        'next_steps': '\n💡 Next steps:',
        'next_step_review': '   1. Review extracted images in species subdirectories: {path}',
        'next_step_verify': '   2. Manually verify auto-classifications (if using species model)',
        'next_step_manual_sort': '   2. Create subdirectories for each species and move images into correct folders',
        'next_step_organize': '   3. Use organize_dataset.py to create train/val split',
        'next_step_train': '   4. Train improved model with new data!',
        'processing_video': '\n{"="*70}\n📹 Processing video {idx}/{total}: {name}\n{"="*70}',
        'error_processing': '\n❌ Error processing {name}: {error}',
        'continuing': '   Continuing with next video...',
        'all_videos_processed': '\n{"="*70}\n✅ All videos processed!\n   📁 Output directory: {path}\n{"="*70}',
        
        # Organization
        'organizing_dataset': '📊 Organizing dataset: {path}',
        'output_dir': '📁 Output directory: {path}',
        'train_ratio': '📈 Train/Val ratio: {ratio:.0%}/{val:.0%}',
        'found_species': '🐦 Found {count} species:',
        'species_images': '   • {species}: {count} images',
        'creating_splits': '🔄 Creating train/val splits...',
        'split_created': '   ✅ {species}: {train} train / {val} val',
        'organization_complete': '\n✅ Dataset organized!',
        'dataset_summary': '\n📊 Dataset Summary:',
        'total_images': '   📷 Total images: {count}',
        'training_images': '   🏋️  Training: {count} images',
        'validation_images': '   ✅ Validation: {count} images',
        
        # Training
        'training_model': '🎓 Training model on dataset: {path}',
        'model_output': '📁 Output directory: {path}',
        'loading_dataset': '📂 Loading dataset from: {path}',
        'detected_species': '🐦 Detected {count} species: {species}',
        'train_images': '   📊 Training images: {count}',
        'val_images': '   📊 Validation images: {count}',
        'loading_model': '🤖 Loading model: {model}',
        'model_params': '   ℹ️  Model parameters: {params:,}',
        'training_config': '⚙️  Training Configuration:',
        'config_epochs': '   📈 Epochs: {epochs}',
        'config_batch': '   📦 Batch size: {batch}',
        'config_learning_rate': '   📊 Learning rate: {lr}',
        'config_optimizer': '   🔧 Optimizer: {optimizer}',
        'config_scheduler': '   📉 LR Scheduler: {scheduler}',
        'starting_training': '🚀 Starting training...',
        'training_interrupted': '\n\n⚠️  Training interrupted!',
        'saving_checkpoint': '💾 Saving checkpoint...',
        'checkpoint_saved': '✅ Checkpoint saved: {path}',
        'training_complete': '\n✅ Training complete!',
        'final_model_saved': '💾 Final model saved: {path}',
        'training_summary': '\n📊 Training Summary:',
        'best_accuracy': '   🏆 Best Accuracy: {acc:.2%}',
        'final_loss': '   📉 Final Loss: {loss:.4f}',
        
        # Testing
        'testing_model': '🧪 Testing model: {path}',
        'loading_test_model': '🤖 Loading model and processor...',
        'testing_validation': '🧪 Testing model on validation set...',
        'testing_image': '🖼️  Testing single image: {path}',
        'image_not_found': '❌ Image not found: {path}',
        'predicted_species': '   🐦 Predicted: {species} ({conf:.1%} confidence)',
        'top_predictions': '\n📊 Top 5 predictions:',
        'prediction_item': '   {rank}. {species}: {conf:.1%}',
        'testing_species': '🐦 Testing {species}...',
        'test_result': '   🐦 Predicted: {predicted} ({conf:.1%} confidence)',
        'test_summary': '\n📊 Test Results:',
        'overall_accuracy': '   🎯 Overall Accuracy: {acc:.2%}',
        'correct_predictions': '   ✅ Correct: {correct}/{total}',
        'species_accuracy': '\n📈 Per-Species Accuracy:',
        'species_acc_item': '   • {species}: {acc:.1%} ({correct}/{total})',
        
        # CLI
        'cli_extracting_from': '🎥 Extracting birds from: {path}',
        'cli_output_folder': '📁 Output folder: {path}',
        'cli_output_directory': '📁 Output directory: {path}',
        'cli_species': '🐦 Species: {species}',
        'cli_using_classifier': '🤖 Using species classifier: {path}',
        'cli_processing_video': '\n📹 Processing: {path}',
        'cli_organizing_dataset': '📊 Organizing dataset: {path}',
        'cli_training_model': '🎓 Training model on dataset: {path}',
        
        # Training
        'train_loading_model': '🤖 Loading base model: {model}',
        'train_header': 'Bird Species Recognition Training',
        'train_ctrl_c_hint': '(Press Ctrl+C to cleanly exit)',
        'train_detecting_species': '\n🔍 Detecting bird species from directory structure...',
        'train_output_dir': '\n📁 Output directory: {path}',
        'train_species': '🐦 Bird species: {species}',
        'train_num_classes': '🔢 Number of classes: {count}',
        'train_loading_dataset': '\n📂 Loading dataset...',
        'train_dataset_size': '   Training:   {train} images',
        'train_val_size': '   Validation: {val} images',
        'train_dataset_labels': '\n📋 Dataset Labels: {labels}',
        'train_detected_species': '📋 Detected Species: {species}',
        'train_species_mismatch_warning': '\n⚠️  WARNING: Species list does not match dataset!',
        'train_species_mismatch_details': '   Dataset has: {dataset}\n   Detected: {detected}',
        'train_using_dataset_labels': '\n✅ Using Dataset Label-Mapping: {labels}',
        'train_applying_transforms': '\n🔄 Applying transformations...',
        'train_starting': '\n🚀 Starting training...',
        'train_batch_size': '📦 Batch Size: {size}',
        'train_learning_rate': '📈 Learning Rate: {rate}',
        'train_epochs': '🔄 Epochs: {epochs}',
        'train_saving_model': '\n💾 Saving final model: {path}',
        'train_complete': '\n✅ Training complete!',
        'train_model_saved': '📁 Model saved in: {path}',
        'train_interrupted': '\n\n⚠️  Training interrupted (Ctrl+C detected)',
        'train_waiting_clean_exit': 'Waiting for clean exit of current step...',
        'train_force_exit_hint': '(Press Ctrl+C again for immediate exit)',
        'train_force_exit': '\n⚠️  Immediate exit forced!',
        'train_error': '\n❌ Training error: {error}',
        
        # Testing
        'test_loading_model': '🤖 Loading model: {path}',
        'test_classifying_image': '🖼️  Classifying image: {path}',
        'test_results': '\nResults:',
        'test_on_validation': '\n🧪 Testing on validation set: {path}',
        'test_no_images_found': '⚠️  No images found for {species}',
        'test_overall_accuracy': '📊 Overall accuracy: {correct}/{total} = {accuracy:.1f}%',
        'test_usage': 'Usage:',
        'test_usage_single': '  Test single image:',
        'test_usage_single_cmd': '    python test_model.py <model_path> <image_path>',
        'test_usage_validation': '  Test on validation set:',
        'test_usage_validation_cmd': '    python test_model.py <model_path> --data-dir <data_dir>',
        'test_error_no_input': '❌ Please specify either image path or --data-dir',
        
        # General
        'error': 'Error',
        'warning': 'Warning',
    },
    
    'de': {
        # Extraction
        'loading_yolo': '🤖 Lade YOLO-Modell:',
        'loading_species': '🧠 Lade Arten-Klassifizierer:',
        'loaded_species_classes': '   ✅ Geladen mit {count} Arten-Klassen',
        'video_info': '📹 Video:',
        'total_frames': '   📊 {total} Frames, {fps:.1f} FPS',
        'analyzing_every_nth': '   🔍 Analysiere jeden {n}. Frame...',
        'detection_threshold': '   🎯 Erkennungs-Schwellwert: {threshold}',
        'species_threshold': '   🎯 Arten-Schwellwert: {threshold}',
        'image_size': '   📐 Bildgröße: {size}x{size}px',
        'image_size_original': '   📐 Bildgröße: Original',
        'mode_autosorting': '   🤖 Auto-Sortier-Modus: Nutze Arten-Klassifizierer',
        'mode_manual': '   🏷️  Manueller Modus: Art = {species}',
        'mode_standard': '   📦 Standard-Modus: Alle Vögel in einem Verzeichnis',
        'cannot_open_video': '❌ Kann Video nicht öffnen: {path}',
        'bird_extracted': '   ✅ Vogel #{count}: {species} (Konf {conf:.2f}), Frame {frame}',
        'bird_extracted_simple': '   ✅ Vogel extrahiert #{count}: Frame {frame}, Konf {conf:.2f}',
        'bird_skipped': '   ⏭️  Übersprungen: {species} (Konf {conf:.2f} < {threshold:.2f}), Frame {frame}',
        'progress': '   ⏳ Fortschritt: {percent:.1f}% ({current}/{total} Frames)',
        'extraction_interrupted': '\n⚠️  Extraktion vom Benutzer unterbrochen',
        'extraction_complete': '\n✅ Extraktion abgeschlossen!',
        'output_directory': '   📁 Ausgabe-Verzeichnis: {path}',
        'detected_birds_total': '   🔍 Erkannte Vögel gesamt: {count}',
        'exported_birds_total': '   🐦 Exportierte Vögel: {count}',
        'skipped_birds_total': '   ⏭️  Übersprungen (< {threshold:.2f}): {count}',
        'total_birds': '   🐦 Extrahierte Vögel gesamt: {count}',
        'species_breakdown': '\n📊 Arten-Aufschlüsselung:',
        'species_count': '   • {species}: {count} Vögel',
        'session_id': '   🆔 Sitzungs-ID: {id}',
        'filename_format': '\n💡 Dateinamen-Format: {format}',
        'next_steps': '\n💡 Nächste Schritte:',
        'next_step_review': '   1. Überprüfe extrahierte Bilder in Arten-Unterverzeichnissen: {path}',
        'next_step_verify': '   2. Manuell Auto-Klassifizierungen verifizieren (falls Arten-Modell verwendet)',
        'next_step_manual_sort': '   2. Erstelle Unterverzeichnisse für jede Art und verschiebe Bilder in korrekte Ordner',
        'next_step_organize': '   3. Nutze organize_dataset.py um Train/Val Split zu erstellen',
        'next_step_train': '   4. Trainiere verbessertes Modell mit neuen Daten!',
        'processing_video': '\n{"="*70}\n📹 Verarbeite Video {idx}/{total}: {name}\n{"="*70}',
        'error_processing': '\n❌ Fehler beim Verarbeiten von {name}: {error}',
        'continuing': '   Fahre mit nächstem Video fort...',
        'all_videos_processed': '\n{"="*70}\n✅ Alle Videos verarbeitet!\n   📁 Ausgabe-Verzeichnis: {path}\n{"="*70}',
        
        # Organization
        'organizing_dataset': '📊 Organisiere Dataset: {path}',
        'output_dir': '📁 Ausgabe-Verzeichnis: {path}',
        'train_ratio': '📈 Train/Val Verhältnis: {ratio:.0%}/{val:.0%}',
        'found_species': '🐦 {count} Arten gefunden:',
        'species_images': '   • {species}: {count} Bilder',
        'creating_splits': '🔄 Erstelle Train/Val Splits...',
        'split_created': '   ✅ {species}: {train} Train / {val} Val',
        'organization_complete': '\n✅ Dataset organisiert!',
        'dataset_summary': '\n📊 Dataset-Zusammenfassung:',
        'total_images': '   📷 Bilder gesamt: {count}',
        'training_images': '   🏋️  Training: {count} Bilder',
        'validation_images': '   ✅ Validierung: {count} Bilder',
        
        # Training
        'training_model': '🎓 Trainiere Modell auf Dataset: {path}',
        'model_output': '📁 Ausgabe-Verzeichnis: {path}',
        'loading_dataset': '📂 Lade Dataset von: {path}',
        'detected_species': '🐦 {count} Arten erkannt: {species}',
        'train_images': '   📊 Trainingsbilder: {count}',
        'val_images': '   📊 Validierungsbilder: {count}',
        'loading_model': '🤖 Lade Modell: {model}',
        'model_params': '   ℹ️  Modell-Parameter: {params:,}',
        'training_config': '⚙️  Trainings-Konfiguration:',
        'config_epochs': '   📈 Epochen: {epochs}',
        'config_batch': '   📦 Batch-Größe: {batch}',
        'config_learning_rate': '   📊 Lernrate: {lr}',
        'config_optimizer': '   🔧 Optimizer: {optimizer}',
        'config_scheduler': '   📉 LR Scheduler: {scheduler}',
        'starting_training': '🚀 Starte Training...',
        'training_interrupted': '\n\n⚠️  Training unterbrochen!',
        'saving_checkpoint': '💾 Speichere Checkpoint...',
        'checkpoint_saved': '✅ Checkpoint gespeichert: {path}',
        'training_complete': '\n✅ Training abgeschlossen!',
        'final_model_saved': '💾 Finales Modell gespeichert: {path}',
        'training_summary': '\n📊 Trainings-Zusammenfassung:',
        'best_accuracy': '   🏆 Beste Genauigkeit: {acc:.2%}',
        'final_loss': '   📉 Finaler Loss: {loss:.4f}',
        
        # Testing
        'testing_model': '🧪 Teste Modell: {path}',
        'loading_test_model': '🤖 Lade Modell und Prozessor...',
        'testing_validation': '🧪 Teste Modell auf Validierungs-Set...',
        'testing_image': '🖼️  Teste einzelnes Bild: {path}',
        'image_not_found': '❌ Bild nicht gefunden: {path}',
        'predicted_species': '   🐦 Vorhersage: {species} ({conf:.1%} Konfidenz)',
        'top_predictions': '\n📊 Top 5 Vorhersagen:',
        'prediction_item': '   {rank}. {species}: {conf:.1%}',
        'testing_species': '🐦 Teste {species}...',
        'test_result': '   🐦 Vorhersage: {predicted} ({conf:.1%} Konfidenz)',
        'test_summary': '\n📊 Test-Ergebnisse:',
        'overall_accuracy': '   🎯 Gesamt-Genauigkeit: {acc:.2%}',
        'correct_predictions': '   ✅ Korrekt: {correct}/{total}',
        'species_accuracy': '\n📈 Genauigkeit pro Art:',
        'species_acc_item': '   • {species}: {acc:.1%} ({correct}/{total})',
        
        # CLI
        'cli_extracting_from': '🎥 Extrahiere Vögel aus: {path}',
        'cli_output_folder': '📁 Ausgabe-Ordner: {path}',
        'cli_output_directory': '📁 Ausgabe-Verzeichnis: {path}',
        'cli_species': '🐦 Art: {species}',
        'cli_using_classifier': '🤖 Nutze Arten-Klassifizierer: {path}',
        'cli_processing_video': '\n📹 Verarbeite: {path}',
        'cli_organizing_dataset': '📊 Organisiere Dataset: {path}',
        'cli_training_model': '🎓 Trainiere Modell auf Dataset: {path}',
        
        # Training
        'train_loading_model': '🤖 Lade Basis-Modell: {model}',
        'train_header': 'Vogel-Artenerkennung Training',
        'train_ctrl_c_hint': '(Drücke Strg+C zum sauberen Beenden)',
        'train_detecting_species': '\n🔍 Erkenne Vogelarten aus Verzeichnisstruktur...',
        'train_output_dir': '\n📁 Ausgabe-Verzeichnis: {path}',
        'train_species': '🐦 Vogelarten: {species}',
        'train_num_classes': '🔢 Anzahl Klassen: {count}',
        'train_loading_dataset': '\n📂 Lade Dataset...',
        'train_dataset_size': '   Training:   {train} Bilder',
        'train_val_size': '   Validation: {val} Bilder',
        'train_dataset_labels': '\n📋 Dataset Labels: {labels}',
        'train_detected_species': '📋 Erkannte Arten: {species}',
        'train_species_mismatch_warning': '\n⚠️  WARNUNG: Arten-Liste stimmt nicht mit Dataset überein!',
        'train_species_mismatch_details': '   Dataset hat: {dataset}\n   Erkannt wurden: {detected}',
        'train_using_dataset_labels': '\n✅ Verwende Dataset Label-Mapping: {labels}',
        'train_applying_transforms': '\n🔄 Appliziere Transformationen...',
        'train_starting': '\n🚀 Starte Training...',
        'train_batch_size': '📦 Batch Size: {size}',
        'train_learning_rate': '📈 Learning Rate: {rate}',
        'train_epochs': '🔄 Epochen: {epochs}',
        'train_saving_model': '\n💾 Speichere finales Modell: {path}',
        'train_complete': '\n✅ Training abgeschlossen!',
        'train_model_saved': '📁 Modell gespeichert in: {path}',
        'train_interrupted': '\n\n⚠️  Training wird unterbrochen (Strg+C erkannt)',
        'train_waiting_clean_exit': 'Warte auf sauberes Beenden des aktuellen Schritts...',
        'train_force_exit_hint': '(Drücke Strg+C erneut für sofortiges Beenden)',
        'train_force_exit': '\n⚠️  Sofortiges Beenden erzwungen!',
        'train_error': '\n❌ Fehler beim Training: {error}',
        
        # Testing
        'test_loading_model': '🤖 Lade Modell: {path}',
        'test_classifying_image': '🖼️  Klassifiziere Bild: {path}',
        'test_results': '\nErgebnisse:',
        'test_on_validation': '\n🧪 Teste auf Validation Set: {path}',
        'test_no_images_found': '⚠️  Keine Bilder für {species} gefunden',
        'test_overall_accuracy': '📊 Gesamtgenauigkeit: {correct}/{total} = {accuracy:.1f}%',
        'test_usage': 'Verwendung:',
        'test_usage_single': '  Einzelnes Bild testen:',
        'test_usage_single_cmd': '    python test_model.py <model_path> <image_path>',
        'test_usage_validation': '  Auf Validation Set testen:',
        'test_usage_validation_cmd': '    python test_model.py <model_path> --data-dir <data_dir>',
        'test_error_no_input': '❌ Bitte entweder Bildpfad oder --data-dir angeben',
        
        # General
        'error': 'Fehler',
        'warning': 'Warnung',
    },
    
    'ja': {
        # Extraction
        'loading_yolo': '🤖 YOLOモデルを読み込んでいます：',
        'loading_species': '🧠 種分類器を読み込んでいます：',
        'loaded_species_classes': '   ✅ {count}種のクラスを読み込みました',
        'video_info': '📹 ビデオ：',
        'total_frames': '   📊 {total}フレーム、{fps:.1f} FPS',
        'analyzing_every_nth': '   🔍 {n}フレームごとに分析中...',
        'detection_threshold': '   🎯 検出しきい値：{threshold}',
        'species_threshold': '   🎯 種のしきい値：{threshold}',
        'image_size': '   📐 画像サイズ：{size}x{size}px',
        'image_size_original': '   📐 画像サイズ：オリジナル',
        'mode_autosorting': '   🤖 自動ソートモード：種分類器を使用',
        'mode_manual': '   🏷️  手動モード：種 = {species}',
        'mode_standard': '   📦 標準モード：すべての鳥を1つのディレクトリに',
        'cannot_open_video': '❌ ビデオを開けません：{path}',
        'bird_extracted': '   ✅ 鳥 #{count}：{species}（信頼度 {conf:.2f}）、フレーム {frame}',
        'bird_extracted_simple': '   ✅ 鳥を抽出 #{count}：フレーム {frame}、信頼度 {conf:.2f}',
        'bird_skipped': '   ⏭️  スキップ：{species}（信頼度 {conf:.2f} < {threshold:.2f}）、フレーム {frame}',
        'progress': '   ⏳ 進行状況：{percent:.1f}% （{current}/{total} フレーム）',
        'extraction_interrupted': '\n⚠️  ユーザーによって抽出が中断されました',
        'extraction_complete': '\n✅ 抽出完了！',
        'output_directory': '   📁 出力ディレクトリ：{path}',
        'detected_birds_total': '   🔍 検出された鳥の総数：{count}',
        'exported_birds_total': '   🐦 エクスポートされた鳥：{count}',
        'skipped_birds_total': '   ⏭️  スキップされた (< {threshold:.2f})：{count}',
        'total_birds': '   🐦 抽出された鳥の総数：{count}',
        'species_breakdown': '\n📊 種の内訳：',
        'species_count': '   • {species}：{count}羽',
        'session_id': '   🆔 セッションID：{id}',
        'filename_format': '\n💡 ファイル名形式：{format}',
        'next_steps': '\n💡 次のステップ：',
        'next_step_review': '   1. 種サブディレクトリ内の抽出画像を確認：{path}',
        'next_step_verify': '   2. 自動分類を手動で確認（種モデル使用時）',
        'next_step_manual_sort': '   2. 各種のサブディレクトリを作成し、画像を正しいフォルダに移動',
        'next_step_organize': '   3. organize_dataset.pyを使用してトレーニング/検証分割を作成',
        'next_step_train': '   4. 新しいデータで改善されたモデルをトレーニング！',
        'processing_video': '\n{"="*70}\n📹 ビデオ処理中 {idx}/{total}：{name}\n{"="*70}',
        'error_processing': '\n❌ {name}の処理エラー：{error}',
        'continuing': '   次のビデオに続けます...',
        'all_videos_processed': '\n{"="*70}\n✅ すべてのビデオを処理しました！\n   📁 出力ディレクトリ：{path}\n{"="*70}',
        
        # Organization
        'organizing_dataset': '📊 データセットを整理中：{path}',
        'output_dir': '📁 出力ディレクトリ：{path}',
        'train_ratio': '📈 トレーニング/検証比率：{ratio:.0%}/{val:.0%}',
        'found_species': '🐦 {count}種を発見：',
        'species_images': '   • {species}：{count}枚の画像',
        'creating_splits': '🔄 トレーニング/検証分割を作成中...',
        'split_created': '   ✅ {species}：{train}トレーニング / {val}検証',
        'organization_complete': '\n✅ データセット整理完了！',
        'dataset_summary': '\n📊 データセット概要：',
        'total_images': '   📷 総画像数：{count}',
        'training_images': '   🏋️  トレーニング：{count}枚',
        'validation_images': '   ✅ 検証：{count}枚',
        
        # Training
        'training_model': '🎓 データセットでモデルをトレーニング中：{path}',
        'model_output': '📁 出力ディレクトリ：{path}',
        'loading_dataset': '📂 データセットを読み込んでいます：{path}',
        'detected_species': '🐦 {count}種を検出：{species}',
        'train_images': '   📊 トレーニング画像：{count}',
        'val_images': '   📊 検証画像：{count}',
        'loading_model': '🤖 モデルを読み込んでいます：{model}',
        'model_params': '   ℹ️  モデルパラメータ：{params:,}',
        'training_config': '⚙️  トレーニング設定：',
        'config_epochs': '   📈 エポック数：{epochs}',
        'config_batch': '   📦 バッチサイズ：{batch}',
        'config_learning_rate': '   📊 学習率：{lr}',
        'config_optimizer': '   🔧 オプティマイザー：{optimizer}',
        'config_scheduler': '   📉 LRスケジューラー：{scheduler}',
        'starting_training': '🚀 トレーニング開始...',
        'training_interrupted': '\n\n⚠️  トレーニングが中断されました！',
        'saving_checkpoint': '💾 チェックポイントを保存中...',
        'checkpoint_saved': '✅ チェックポイント保存完了：{path}',
        'training_complete': '\n✅ トレーニング完了！',
        'final_model_saved': '💾 最終モデル保存完了：{path}',
        'training_summary': '\n📊 トレーニング概要：',
        'best_accuracy': '   🏆 最高精度：{acc:.2%}',
        'final_loss': '   📉 最終損失：{loss:.4f}',
        
        # Testing
        'testing_model': '🧪 モデルをテスト中：{path}',
        'loading_test_model': '🤖 モデルとプロセッサーを読み込んでいます...',
        'testing_validation': '🧪 検証セットでモデルをテスト中...',
        'testing_image': '🖼️  単一画像をテスト中：{path}',
        'image_not_found': '❌ 画像が見つかりません：{path}',
        'predicted_species': '   🐦 予測：{species}（{conf:.1%}信頼度）',
        'top_predictions': '\n📊 上位5つの予測：',
        'prediction_item': '   {rank}. {species}：{conf:.1%}',
        'testing_species': '🐦 {species}をテスト中...',
        'test_result': '   🐦 予測：{predicted}（{conf:.1%}信頼度）',
        'test_summary': '\n📊 テスト結果：',
        'overall_accuracy': '   🎯 全体精度：{acc:.2%}',
        'correct_predictions': '   ✅ 正解：{correct}/{total}',
        'species_accuracy': '\n📈 種ごとの精度：',
        'species_acc_item': '   • {species}：{acc:.1%}（{correct}/{total}）',
        
        # CLI
        'cli_extracting_from': '🎥 鳥を抽出中：{path}',
        'cli_output_folder': '📁 出力フォルダ：{path}',
        'cli_output_directory': '📁 出力ディレクトリ：{path}',
        'cli_species': '🐦 種：{species}',
        'cli_using_classifier': '🤖 種分類器を使用：{path}',
        'cli_processing_video': '\n📹 処理中：{path}',
        'cli_organizing_dataset': '📊 データセット整理中：{path}',
        'cli_training_model': '🎓 データセットでモデルをトレーニング中：{path}',
        
        # Training
        'train_loading_model': '🤖 ベースモデルを読み込み中：{model}',
        'train_header': '鳥種認識トレーニング',
        'train_ctrl_c_hint': '（Ctrl+Cできれいに終了）',
        'train_detecting_species': '\n🔍 ディレクトリ構造から鳥種を検出中...',
        'train_output_dir': '\n📁 出力ディレクトリ：{path}',
        'train_species': '🐦 鳥の種：{species}',
        'train_num_classes': '🔢 クラス数：{count}',
        'train_loading_dataset': '\n📂 データセットを読み込み中...',
        'train_dataset_size': '   トレーニング：{train}枚の画像',
        'train_val_size': '   検証：{val}枚の画像',
        'train_dataset_labels': '\n📋 データセットラベル：{labels}',
        'train_detected_species': '📋 検出された種：{species}',
        'train_species_mismatch_warning': '\n⚠️  警告：種リストがデータセットと一致しません！',
        'train_species_mismatch_details': '   データセット：{dataset}\n   検出：{detected}',
        'train_using_dataset_labels': '\n✅ データセットラベルマッピングを使用：{labels}',
        'train_applying_transforms': '\n🔄 変換を適用中...',
        'train_starting': '\n🚀 トレーニング開始...',
        'train_batch_size': '📦 バッチサイズ：{size}',
        'train_learning_rate': '📈 学習率：{rate}',
        'train_epochs': '🔄 エポック：{epochs}',
        'train_saving_model': '\n💾 最終モデルを保存中：{path}',
        'train_complete': '\n✅ トレーニング完了！',
        'train_model_saved': '📁 モデル保存先：{path}',
        'train_interrupted': '\n\n⚠️  トレーニング中断（Ctrl+C検出）',
        'train_waiting_clean_exit': '現在のステップのクリーンな終了を待っています...',
        'train_force_exit_hint': '（Ctrl+Cを再度押すと即座に終了）',
        'train_force_exit': '\n⚠️  即座の終了を強制！',
        'train_error': '\n❌ トレーニングエラー：{error}',
        
        # Testing
        'test_loading_model': '🤖 モデルを読み込み中：{path}',
        'test_classifying_image': '🖼️  画像を分類中：{path}',
        'test_results': '\n結果：',
        'test_on_validation': '\n🧪 検証セットでテスト中：{path}',
        'test_no_images_found': '⚠️  {species}の画像が見つかりません',
        'test_overall_accuracy': '📊 全体精度：{correct}/{total} = {accuracy:.1f}%',
        'test_usage': '使用方法：',
        'test_usage_single': '  単一画像をテスト：',
        'test_usage_single_cmd': '    python test_model.py <model_path> <image_path>',
        'test_usage_validation': '  検証セットでテスト：',
        'test_usage_validation_cmd': '    python test_model.py <model_path> --data-dir <data_dir>',
        'test_error_no_input': '❌ 画像パスまたは--data-dirを指定してください',
        
        # General
        'error': 'エラー',
        'warning': '警告',
    }
}

# Current language (default: English)
_current_lang = 'en'


def detect_language():
    """Detect system language from environment variables."""
    lang = os.environ.get('LANG', '')
    
    if lang.startswith('de'):
        return 'de'
    elif lang.startswith('ja'):
        return 'ja'
    else:
        return 'en'


def set_language(lang):
    """Set the current language."""
    global _current_lang
    if lang in TRANSLATIONS:
        _current_lang = lang
    else:
        _current_lang = 'en'


def get_text(key, **kwargs):
    """
    Get translated text for the given key.
    
    Args:
        key: Translation key
        **kwargs: Format parameters for the translation string
        
    Returns:
        Formatted translation string
    """
    translation = TRANSLATIONS.get(_current_lang, TRANSLATIONS['en']).get(key, key)
    
    if kwargs:
        try:
            return translation.format(**kwargs)
        except KeyError:
            return translation
    
    return translation


# Alias for shorter usage
_ = get_text


# Auto-detect language on module import
_current_lang = detect_language()
