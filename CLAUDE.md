# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flutter application for object detection using YOLOv8 models (supports both regular and OBB - Oriented Bounding Box). The app allows users to pick images from gallery or camera, runs inference via TensorFlow Lite, and displays detection results with bounding boxes.

## Commands

```bash
flutter run                    # Run on connected device/emulator
flutter test                   # Run all tests
flutter analyze                # Static analysis
flutter build apk              # Build Android APK
flutter build ios              # Build iOS app
```

## Architecture

**Core Services:**
- `lib/yolo_service.dart` - YOLO inference engine with automatic model type detection (regular vs OBB), NCHW/NHWC input format detection, and Non-Maximum Suppression
- `lib/widgets/box_painter.dart` - CustomPainter for rendering rotated and axis-aligned bounding boxes

**Main Entry:** `lib/main.dart` - Single-page app with `YoloApp` widget handling image picking, inference, and result display

**Model Assets:**
- `assets/my_model_float32.tflite` - TensorFlow Lite model
- `assets/labels.txt` - Class labels

**Key Dependencies:**
- `tflite_flutter` - TensorFlow Lite interpreter
- `image_picker` - Camera/gallery access
- `image` - Image decoding/manipulation

## Model Detection Logic

The `YoloService` auto-detects model configuration at runtime:
- Input format: NCHW vs NHWC from tensor shape
- Model type: Regular (84 outputs) vs OBB (85 outputs with rotation angle)
- Coordinate system: Normalized [0-1] vs pixel values
