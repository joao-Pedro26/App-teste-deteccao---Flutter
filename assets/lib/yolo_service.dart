import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class Recognition {
  final int classId;
  final String label;
  final double score;
  final Rect location; // Coordenadas normalizadas 0.0 a 1.0

  Recognition(this.classId, this.label, this.score, this.location);
}

class YoloService {
  Interpreter? _interpreter;
  List<String> _labels = [];

  static const int inputSize = 640;
  static const double confidenceThreshold = 0.25;
  static const double nmsThreshold = 0.45;

  bool _isNCHW = false;

  Future<void> init() async {
    _interpreter = await Interpreter.fromAsset('assets/yolov8n_float32.tflite');

    final labelsData = await rootBundle.loadString('assets/labels.txt');
    _labels = labelsData
        .split('\n')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();

    final inputShape  = _interpreter!.getInputTensor(0).shape;
    final outputShape = _interpreter!.getOutputTensor(0).shape;
    _isNCHW = (inputShape.length == 4 && inputShape[1] == 3);

    debugPrint('[YOLO] Input shape : $inputShape => ${_isNCHW ? "NCHW" : "NHWC"}');
    debugPrint('[YOLO] Output shape: $outputShape');
    debugPrint('[YOLO] Labels: ${_labels.length}');
  }

  Future<List<Recognition>> runInference(img.Image image) async {
    if (_interpreter == null || _labels.isEmpty) return [];

    // ── 1. Resize ──────────────────────────────────────────────────────────
    final img.Image resized = img.copyResize(
      image,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );

    // ── 2. Montar tensor de input (NCHW ou NHWC) ───────────────────────────
    List inputTensor;
    if (_isNCHW) {
      inputTensor = List.generate(1, (_) =>
        List.generate(3, (c) =>
          List.generate(inputSize, (y) =>
            List.generate(inputSize, (x) {
              final p = resized.getPixel(x, y);
              return c == 0 ? p.r / 255.0 : c == 1 ? p.g / 255.0 : p.b / 255.0;
            })
          )
        )
      );
    } else {
      inputTensor = List.generate(1, (_) =>
        List.generate(inputSize, (y) =>
          List.generate(inputSize, (x) {
            final p = resized.getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          })
        )
      );
    }

    // ── 3. Output [1, 84, 8400] ────────────────────────────────────────────
    final output = List.generate(1, (_) =>
      List.generate(84, (_) => List.filled(8400, 0.0))
    );
    _interpreter!.run(inputTensor, output);

    final raw = output[0]; // [84][8400]

    // ── 4. Detectar se coordenadas já são normalizadas (0-1) ou pixel (0-640)
    //
    // Estratégia: pega as primeiras detecções com score alto e verifica
    // se cx/cy estão abaixo de 1.0 (normalizado) ou acima (pixels).
    // Se a maioria dos cx detectados for <= 1.0, já está normalizado.
    //
    bool coordsAreNormalized = _detectIfNormalized(raw);
    debugPrint('[YOLO] Coordenadas normalizadas pelo modelo: $coordsAreNormalized');

    // ── 5. Decodificar predições ───────────────────────────────────────────
    final List<Recognition> candidates = [];

    for (int i = 0; i < 8400; i++) {
      int bestClass = 0;
      double bestScore = 0.0;
      for (int c = 0; c < _labels.length; c++) {
        final s = raw[4 + c][i];
        if (s > bestScore) { bestScore = s; bestClass = c; }
      }
      if (bestScore < confidenceThreshold) continue;

      final double cx = raw[0][i];
      final double cy = raw[1][i];
      final double w  = raw[2][i];
      final double h  = raw[3][i];

      // Se já normalizado: usa direto. Se em pixels: divide por inputSize.
      final double divisor = coordsAreNormalized ? 1.0 : inputSize.toDouble();

      final double left   = ((cx - w / 2) / divisor).clamp(0.0, 1.0);
      final double top    = ((cy - h / 2) / divisor).clamp(0.0, 1.0);
      final double right  = ((cx + w / 2) / divisor).clamp(0.0, 1.0);
      final double bottom = ((cy + h / 2) / divisor).clamp(0.0, 1.0);

      if (right <= left || bottom <= top) continue;

      candidates.add(Recognition(
        bestClass, _labels[bestClass], bestScore,
        Rect.fromLTRB(left, top, right, bottom),
      ));
    }

    debugPrint('[YOLO] Candidatos pre-NMS: ${candidates.length}');

    final results = _applyNMS(candidates);

    debugPrint('[YOLO] Deteccoes finais: ${results.length}');
    for (var r in results) {
      debugPrint('  => ${r.label} ${(r.score * 100).toStringAsFixed(1)}%'
          '  LTRB=(${r.location.left.toStringAsFixed(3)},'
          '${r.location.top.toStringAsFixed(3)},'
          '${r.location.right.toStringAsFixed(3)},'
          '${r.location.bottom.toStringAsFixed(3)})');
    }

    return results;
  }

  /// Amostra as primeiras predições com score razoável e verifica se cx > 1.0.
  /// Se a maioria estiver <= 1.0 → modelo já normaliza as coordenadas.
  bool _detectIfNormalized(List<List<double>> raw) {
    int pixelCount = 0;
    int normalCount = 0;
    int sampled = 0;

    for (int i = 0; i < 8400 && sampled < 20; i++) {
      // Verifica score mínimo para não amostrar lixo
      double maxScore = 0.0;
      for (int c = 0; c < min(_labels.length, 80); c++) {
        if (raw[4 + c][i] > maxScore) maxScore = raw[4 + c][i];
      }
      if (maxScore < 0.1) continue;

      final double cx = raw[0][i];
      if (cx > 1.5) {
        pixelCount++;
      } else {
        normalCount++;
      }
      sampled++;
    }

    debugPrint('[YOLO] Amostra coords: ${normalCount}x normalized, ${pixelCount}x pixels');
    // Considera normalizado se a maioria das amostras estiver <= 1.5
    return normalCount >= pixelCount;
  }

  // ── NMS ───────────────────────────────────────────────────────────────────
  List<Recognition> _applyNMS(List<Recognition> candidates) {
    if (candidates.isEmpty) return [];

    final Map<int, List<Recognition>> byClass = {};
    for (final c in candidates) {
      byClass.putIfAbsent(c.classId, () => []).add(c);
    }

    final List<Recognition> out = [];
    for (final boxes in byClass.values) {
      boxes.sort((a, b) => b.score.compareTo(a.score));
      final suppressed = List.filled(boxes.length, false);
      for (int i = 0; i < boxes.length; i++) {
        if (suppressed[i]) continue;
        out.add(boxes[i]);
        for (int j = i + 1; j < boxes.length; j++) {
          if (!suppressed[j] && _iou(boxes[i].location, boxes[j].location) > nmsThreshold) {
            suppressed[j] = true;
          }
        }
      }
    }
    return out;
  }

  double _iou(Rect a, Rect b) {
    final l  = max(a.left,   b.left);
    final t  = max(a.top,    b.top);
    final r  = min(a.right,  b.right);
    final bt = min(a.bottom, b.bottom);
    if (r <= l || bt <= t) return 0.0;
    final inter = (r - l) * (bt - t);
    final union = a.width * a.height + b.width * b.height - inter;
    return union <= 0 ? 0.0 : inter / union;
  }
}
