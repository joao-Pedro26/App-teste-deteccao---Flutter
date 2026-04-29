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

  /// Ângulo de rotação em RADIANOS (apenas modelos OBB).
  /// null para modelos regulares.
  final double? angle;

  bool get isOBB => angle != null;

  Recognition(
    this.classId,
    this.label,
    this.score,
    this.location, {
    this.angle,
  });
}

// ─── Tipo do modelo detectado automaticamente ──────────────────────────────
enum YoloModelType { regular, obb }

class YoloService {
  Interpreter? _interpreter;
  List<String> _labels = [];

  // Getter para acessar labels externamente
  List<String> get labels => _labels;

  static const int inputSize = 640;
  static const double confidenceThreshold = 0.25;
  static const double nmsThreshold = 0.45;

  bool _isNCHW = false;
  YoloModelType _modelType = YoloModelType.regular;

  // Shape real do output — detectado no init()
  int _numRows = 6; // 84 para regular, 85 para OBB
  int _numAnchors = 8400;
  int _numClasses = 80; // _numRows - 4 (regular) ou - 5 (OBB)

  Future<void> init() async {
    _interpreter = await Interpreter.fromAsset(
      'assets/my_model_float32.tflite',
    );

    final labelsData = await rootBundle.loadString('assets/labels.txt');
    _labels = labelsData
        .split('\n')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();

    // ── Detecta formato do input (NCHW ou NHWC) ─────────────────────────
    final inputShape = _interpreter!.getInputTensor(0).shape;
    _isNCHW = (inputShape.length == 4 && inputShape[1] == 3);

    // ── Detecta tipo e shape real do output ──────────────────────────────
    // Regular : [1, 84, 8400]  →  4 coords + 80 classes
    // OBB     : [1, 85, 8400]  →  4 coords + 1 ângulo + 80 classes
    final outputShape = _interpreter!.getOutputTensor(0).shape;
    _numRows = outputShape[1]; // 84 ou 85
    _numAnchors = outputShape[2]; // geralmente 8400

    if (_numRows == 85) {
      // 4 coords + 1 ângulo + 80 classes = 85
      _modelType = YoloModelType.obb;
      _numClasses = _numRows - 5; // 80
    } else {
      // 4 coords + N classes = _numRows
      _modelType = YoloModelType.regular;
      _numClasses = _numRows - 4; // 80 para COCO, pode ser outro valor
    }

    // Garante que o número de labels bate com o modelo
    if (_labels.length != _numClasses) {
      debugPrint(
        '[YOLO] AVISO: labels.txt tem ${_labels.length} classes, '
        'mas o modelo espera $_numClasses. Ajuste o labels.txt!',
      );
    }

    debugPrint(
      '[YOLO] Input shape  : $inputShape  =>  '
      '${_isNCHW ? "NCHW" : "NHWC"}',
    );
    debugPrint(
      '[YOLO] Output shape : $outputShape  =>  '
      '${_modelType.name.toUpperCase()} | '
      '$_numClasses classes | $_numAnchors anchors',
    );
  }

  Future<List<Recognition>> runInference(img.Image image) async {
    if (_interpreter == null || _labels.isEmpty) return [];

    // ── 1. Resize ─────────────────────────────────────────────────────────
    final img.Image resized = img.copyResize(
      image,
      width: inputSize,
      height: inputSize,
      interpolation: img.Interpolation.linear,
    );

    // ── 2. Montar tensor de input (NCHW ou NHWC) ──────────────────────────
    List inputTensor;
    if (_isNCHW) {
      inputTensor = List.generate(
        1,
        (_) => List.generate(
          3,
          (c) => List.generate(
            inputSize,
            (y) => List.generate(inputSize, (x) {
              final p = resized.getPixel(x, y);
              return c == 0
                  ? p.r / 255.0
                  : c == 1
                  ? p.g / 255.0
                  : p.b / 255.0;
            }),
          ),
        ),
      );
    } else {
      inputTensor = List.generate(
        1,
        (_) => List.generate(
          inputSize,
          (y) => List.generate(inputSize, (x) {
            final p = resized.getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          }),
        ),
      );
    }

    // ── 3. Output com shape DINÂMICO ──────────────────────────────────────
    // Shape real lido do modelo no init(): [1, _numRows, _numAnchors]
    // Regular: [1, 84, 8400]
    // OBB    : [1, 85, 8400]
    final output = List.generate(
      1,
      (_) => List.generate(_numRows, (_) => List.filled(_numAnchors, 0.0)),
    );

    _interpreter!.run(inputTensor, output);

    final raw = output[0]; // [_numRows][_numAnchors]

    // ── 4. Detecta se coordenadas já são normalizadas ─────────────────────
    final coordsAreNormalized = _detectIfNormalized(raw);
    debugPrint('[YOLO] Coords normalizadas: $coordsAreNormalized');

    // ── 5. Decodificar predições ──────────────────────────────────────────
    final List<Recognition> candidates = [];

    for (int i = 0; i < _numAnchors; i++) {
      // Índice inicial dos scores depende do tipo:
      //   Regular: scores em [4 .. 4+numClasses-1]
      //   OBB    : ângulo em [4], scores em [5 .. 5+numClasses-1]
      final int scoreOffset = _modelType == YoloModelType.obb ? 5 : 4;

      // Encontra a classe com maior score
      int bestClass = 0;
      double bestScore = 0.0;
      for (int c = 0; c < _numClasses; c++) {
        final s = raw[scoreOffset + c][i];
        if (s > bestScore) {
          bestScore = s;
          bestClass = c;
        }
      }

      if (bestScore < confidenceThreshold) continue;

      // Coordenadas do centro
      final double cx = raw[0][i];
      final double cy = raw[1][i];
      final double w = raw[2][i];
      final double h = raw[3][i];

      // Ângulo de rotação (apenas OBB) — em radianos
      // O modelo OBB exporta θ no índice 4
      final double? theta = _modelType == YoloModelType.obb ? raw[4][i] : null;

      // Normaliza coordenadas para [0.0, 1.0]
      final double divisor = coordsAreNormalized ? 1.0 : inputSize.toDouble();

      final double left = ((cx - w / 2) / divisor).clamp(0.0, 1.0);
      final double top = ((cy - h / 2) / divisor).clamp(0.0, 1.0);
      final double right = ((cx + w / 2) / divisor).clamp(0.0, 1.0);
      final double bottom = ((cy + h / 2) / divisor).clamp(0.0, 1.0);

      if (right <= left || bottom <= top) continue;

      // Garante que o classId está dentro dos labels disponíveis
      if (bestClass >= _labels.length) continue;

      candidates.add(
        Recognition(
          bestClass,
          _labels[bestClass],
          bestScore,
          Rect.fromLTRB(left, top, right, bottom),
          angle: theta,
        ),
      );
    }

    debugPrint('[YOLO] Candidatos pre-NMS: ${candidates.length}');

    final results = _applyNMS(candidates);

    debugPrint(
      '[YOLO] Deteccoes finais: ${results.length}  '
      '(modelo: ${_modelType.name.toUpperCase()})',
    );
    for (var r in results) {
      final angStr = r.angle != null
          ? '  θ=${(r.angle! * 180 / pi).toStringAsFixed(1)}°'
          : '';
      debugPrint(
        '  => ${r.label} ${(r.score * 100).toStringAsFixed(1)}%'
        '  LTRB=(${r.location.left.toStringAsFixed(3)},'
        '${r.location.top.toStringAsFixed(3)},'
        '${r.location.right.toStringAsFixed(3)},'
        '${r.location.bottom.toStringAsFixed(3)})'
        '$angStr',
      );
    }

    return results;
  }

  // ── Detecta se as coordenadas já saem normalizadas do modelo ─────────────
  bool _detectIfNormalized(List<List<double>> raw) {
    int pixelCount = 0;
    int normalCount = 0;
    int sampled = 0;

    final int scoreOffset = _modelType == YoloModelType.obb ? 5 : 4;

    for (int i = 0; i < _numAnchors && sampled < 20; i++) {
      double maxScore = 0.0;
      for (int c = 0; c < min(_numClasses, 80); c++) {
        final s = raw[scoreOffset + c][i];
        if (s > maxScore) maxScore = s;
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

    debugPrint(
      '[YOLO] Amostra coords: '
      '${normalCount}x normalized, ${pixelCount}x pixels',
    );
    return normalCount >= pixelCount;
  }

  // ── NMS ───────────────────────────────────────────────────────────────────
  // Para OBB usamos o bounding box axis-aligned como aproximação do IoU.
  // Uma implementação de Rotated IoU exigiria computação de polígono
  // e está fora do escopo deste serviço.
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
          if (!suppressed[j] &&
              YoloService.iou(boxes[i].location, boxes[j].location) > nmsThreshold) {
            suppressed[j] = true;
          }
        }
      }
    }
    return out;
  }

  static double iou(Rect a, Rect b) {
    final l = max(a.left, b.left);
    final t = max(a.top, b.top);
    final r = min(a.right, b.right);
    final bt = min(a.bottom, b.bottom);
    if (r <= l || bt <= t) return 0.0;
    final inter = (r - l) * (bt - t);
    final union = a.width * a.height + b.width * b.height - inter;
    return union <= 0 ? 0.0 : inter / union;
  }
}
