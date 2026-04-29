import 'dart:io';
import 'dart:math' show cos, sin, min, max;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'yolo_service.dart';
import 'widgets/box_painter.dart';
import 'widgets/manual_box_editor.dart';

sealed class _EditAction {}

class _AddedDetections extends _EditAction {
  final List<Recognition> added;
  _AddedDetections(this.added);
}

class _RemovedDetection extends _EditAction {
  final Recognition removed;
  final int originalIndex;
  _RemovedDetection(this.removed, this.originalIndex);
}

void main() => runApp(const MaterialApp(
      home: YoloApp(),
      debugShowCheckedModeBanner: false,
    ));

class YoloApp extends StatefulWidget {
  const YoloApp({super.key});

  @override
  State<YoloApp> createState() => _YoloAppState();
}

class _YoloAppState extends State<YoloApp> {
  final YoloService _yoloService = YoloService();
  File? _imageFile;
  img.Image? _decodedImage; // Guarda a imagem decodificada para obter dimensões reais
  List<Recognition> _results = [];
  bool _isProcessing = false;
  bool _modelReady = false;

  // Novos estados para interação
  bool _isRegionMode = false;      // Modo de seleção por região
  bool _isEditMode = false;        // Modo de edição de boxes
  Rect? _draggingRegion;           // Retângulo sendo desenhado (coordenadas normalizadas)
  Size? _currentWidgetSize;        // Tamanho atual do widget para conversão de coordenadas
  bool _awaitingRegionSelection = false;  // Aguarda seleção de região antes de processar
  bool _regionProcessed = false;           // Região já foi processada
  final List<_EditAction> _undoStack = [];
  static const int _maxUndoDepth = 20;
  final TransformationController _transformationController = TransformationController();

  Rect? _manualBoxRect;
  bool _isManualBoxActive = false;
  int _activeHandleIndex = -1;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _transformationController.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    await _yoloService.init();
    setState(() => _modelReady = true);
  }

  Future<void> _processImage(ImageSource source) async {
    if (!_modelReady) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Modelo ainda carregando, aguarde...')),
      );
      return;
    }

    final XFile? picked = await ImagePicker().pickImage(
      source: source,
      maxWidth: 1280,
    );
    if (picked == null) return;

    setState(() {
      _isProcessing = false; // Não está processando ainda
      _results = [];
      _undoStack.clear();
      _imageFile = File(picked.path);
      _decodedImage = null;
      _awaitingRegionSelection = true; // Aguarda seleção
      _regionProcessed = false;
      _transformationController.value = Matrix4.identity();
    });
  }

  /// Callback para armazenar o tamanho do widget quando disponível
  void _onWidgetSizeAvailable(Size size) {
    setState(() => _currentWidgetSize = size);
  }

  /// Monta o texto de resumo com contagem por classe
  String _getSummary() {
    if (_isProcessing) return 'Processando...';
    if (_results.isEmpty) return 'Nenhum objeto detectado.';

    final Map<String, int> counts = {};
    for (var r in _results) {
      counts[r.label] = (counts[r.label] ?? 0) + 1;
    }
    return counts.entries.map((e) => '${e.value}x ${e.key}').join('  |  ');
  }

  /// Crop da imagem original para uma região e offset das boxes resultantes
  Future<void> _runInferenceOnRegion(Rect region) async {
    if (_decodedImage == null) {
      // Imagem ainda não decodificada - primeiro processamento
      await _confirmRegionAndProcess();
      return;
    }

    setState(() => _isProcessing = true);

    try {
      // Converter região normalizada para pixels na imagem original
      final int rx = (region.left * _decodedImage!.width).round();
      final int ry = (region.top * _decodedImage!.height).round();
      final int rw = (region.width * _decodedImage!.width).round();
      final int rh = (region.height * _decodedImage!.height).round();

      // Crop da imagem
      final cropped = img.copyCrop(_decodedImage!, x: rx, y: ry, width: rw, height: rh);

      // Run inference na região cropada
      final detections = await _yoloService.runInference(cropped);

      // Offset das boxes para coordenadas da imagem original
      final offsetDetections = detections.map((d) {
        final newLeft = region.left + (d.location.left * region.width);
        final newTop = region.top + (d.location.top * region.height);
        final newRight = region.left + (d.location.right * region.width);
        final newBottom = region.top + (d.location.bottom * region.height);

        return Recognition(
          d.classId,
          d.label,
          d.score,
          Rect.fromLTRB(newLeft, newTop, newRight, newBottom),
          angle: d.angle,
        );
      }).toList();

      setState(() {
        _results = offsetDetections;
        _undoStack.clear();
        _isProcessing = false;
        _regionProcessed = true;
      });
    } catch (e) {
      debugPrint('Erro na inferência regional: $e');
      setState(() => _isProcessing = false);
    }
  }

  /// Confirma a região selecionada e processa
  Future<void> _confirmRegionAndProcess() async {
    if (_imageFile == null) return;

    setState(() {
      _awaitingRegionSelection = false;
      _isProcessing = true;
    });

    try {
      final bytes = await _imageFile!.readAsBytes();
      final decoded = img.decodeImage(bytes);

      if (decoded != null) {
        // Se há região selecionada, usa ela; caso contrário, imagem toda
        Rect region = _draggingRegion ?? Rect.fromLTWH(0, 0, 1, 1);

        if (region != Rect.fromLTWH(0, 0, 1, 1) && region.width > 0.02 && region.height > 0.02) {
          // Processar apenas região
          final int rx = (region.left * decoded.width).round();
          final int ry = (region.top * decoded.height).round();
          final int rw = (region.width * decoded.width).round();
          final int rh = (region.height * decoded.height).round();

          final cropped = img.copyCrop(decoded, x: rx, y: ry, width: rw, height: rh);
          final detections = await _yoloService.runInference(cropped);

          // Offset das boxes para coordenadas da imagem original
          final offsetDetections = detections.map((d) {
            final newLeft = region.left + (d.location.left * region.width);
            final newTop = region.top + (d.location.top * region.height);
            final newRight = region.left + (d.location.right * region.width);
            final newBottom = region.top + (d.location.bottom * region.height);

            return Recognition(
              d.classId,
              d.label,
              d.score,
              Rect.fromLTRB(newLeft, newTop, newRight, newBottom),
              angle: d.angle,
            );
          }).toList();

          setState(() {
            _decodedImage = decoded;
            _results = offsetDetections;
            _undoStack.clear();
            _isProcessing = false;
            _regionProcessed = true;
          });
        } else {
          // Processar imagem toda
          final detections = await _yoloService.runInference(decoded);
          setState(() {
            _decodedImage = decoded;
            _results = detections;
            _undoStack.clear();
            _isProcessing = false;
            _regionProcessed = true;
          });
        }
      }
    } catch (e) {
      debugPrint('Erro no processamento: $e');
      setState(() => _isProcessing = false);
    }
  }

  /// Hit testing: verifica se um toque caiu dentro de alguma box
  Recognition? _hitTest(Offset tapPosition) {
    if (_currentWidgetSize == null || _results.isEmpty) return null;

    // Converter tap para coordenadas normalizadas
    final normalizedTap = Offset(
      tapPosition.dx / _currentWidgetSize!.width,
      tapPosition.dy / _currentWidgetSize!.height,
    );

    // Testar cada box (de trás para frente para pegar a mais "em cima")
    for (int i = _results.length - 1; i >= 0; i--) {
      final box = _results[i];
      if (box.isOBB && box.angle != null) {
        // Hit testing para OBB - teste de ponto em polígono
        if (_pointInRotatedRect(normalizedTap, box)) {
          return box;
        }
      } else {
        // Hit testing para box reto
        if (box.location.contains(normalizedTap)) {
          return box;
        }
      }
    }
    return null;
  }

  /// Testa se ponto está dentro de retângulo rotacionado
  bool _pointInRotatedRect(Offset point, Recognition box) {
    // Converter box para vértices rotacionados (mesma lógica do box_painter)
    final cx = (box.location.left + box.location.right) / 2;
    final cy = (box.location.top + box.location.bottom) / 2;
    final w = box.location.width;
    final h = box.location.height;
    final theta = box.angle!;

    final cosA = cos(theta);
    final sinA = sin(theta);
    final hw = w / 2;
    final hh = h / 2;

    // Cantos do retângulo rotacionado
    final corners = [
      Offset(-hw, -hh),
      Offset(hw, -hh),
      Offset(hw, hh),
      Offset(-hw, hh),
    ];

    final rotated = corners.map((c) => Offset(
      cx + c.dx * cosA - c.dy * sinA,
      cy + c.dx * sinA + c.dy * cosA,
    )).toList();

    // Teste de ponto em polígono convexo usando produto vetorial
    bool isInside = true;
    for (int i = 0; i < 4; i++) {
      final p1 = rotated[i];
      final p2 = rotated[(i + 1) % 4];
      // Produto vetorial 2D
      final cross = (p2.dx - p1.dx) * (point.dy - p1.dy) -
                    (p2.dy - p1.dy) * (point.dx - p1.dx);
      // Se todos os produtos tiverem mesmo sinal, ponto está dentro
      if (i == 0) {
        isInside = cross >= 0;
      } else if ((cross >= 0) != isInside) {
        return false;
      }
    }
    return true;
  }

  Future<void> _runInferenceOnTap(Offset normalizedTap) async {
    if (_decodedImage == null) return;

    const double halfCrop = 0.09;
    final cropRect = Rect.fromLTRB(
      (normalizedTap.dx - halfCrop).clamp(0.0, 1.0),
      (normalizedTap.dy - halfCrop).clamp(0.0, 1.0),
      (normalizedTap.dx + halfCrop).clamp(0.0, 1.0),
      (normalizedTap.dy + halfCrop).clamp(0.0, 1.0),
    );

    setState(() => _isProcessing = true);

    try {
      final rx = (cropRect.left * _decodedImage!.width).round();
      final ry = (cropRect.top * _decodedImage!.height).round();
      final rw = (cropRect.width * _decodedImage!.width).round().clamp(1, _decodedImage!.width);
      final rh = (cropRect.height * _decodedImage!.height).round().clamp(1, _decodedImage!.height);

      final cropped = img.copyCrop(_decodedImage!, x: rx, y: ry, width: rw, height: rh);
      final detections = await _yoloService.runInference(cropped);

      final offsetDetections = detections.map((d) => Recognition(
            d.classId,
            d.label,
            d.score,
            Rect.fromLTRB(
              cropRect.left + d.location.left * cropRect.width,
              cropRect.top + d.location.top * cropRect.height,
              cropRect.left + d.location.right * cropRect.width,
              cropRect.top + d.location.bottom * cropRect.height,
            ),
            angle: d.angle,
          )).toList();

      final newDetections = offsetDetections.where((newD) {
        return !_results.any(
          (existing) => YoloService.iou(newD.location, existing.location) > YoloService.nmsThreshold,
        );
      }).toList();

      setState(() => _isProcessing = false);

      if (newDetections.isNotEmpty) {
        setState(() {
          _results.addAll(newDetections);
          _undoStack.add(_AddedDetections(List.of(newDetections)));
          if (_undoStack.length > _maxUndoDepth) _undoStack.removeAt(0);
        });
      } else {
        _openManualBoxEditor(normalizedTap);
      }
    } catch (e) {
      debugPrint('Erro na inferência de tap: $e');
      setState(() => _isProcessing = false);
      _openManualBoxEditor(normalizedTap);
    }
  }

  void _openManualBoxEditor(Offset normalizedCenter) {
    const double defaultSize = 0.08;
    setState(() {
      _manualBoxRect = Rect.fromLTRB(
        (normalizedCenter.dx - defaultSize / 2).clamp(0.0, 1.0),
        (normalizedCenter.dy - defaultSize / 2).clamp(0.0, 1.0),
        (normalizedCenter.dx + defaultSize / 2).clamp(0.0, 1.0),
        (normalizedCenter.dy + defaultSize / 2).clamp(0.0, 1.0),
      );
      _isManualBoxActive = true;
      _activeHandleIndex = -1;
    });
  }

  void _updateManualBoxCorner(int cornerIndex, Offset delta) {
    if (_currentWidgetSize == null || _manualBoxRect == null) return;
    final dx = delta.dx / _currentWidgetSize!.width;
    final dy = delta.dy / _currentWidgetSize!.height;

    double l = _manualBoxRect!.left;
    double t = _manualBoxRect!.top;
    double r = _manualBoxRect!.right;
    double b = _manualBoxRect!.bottom;

    const double minSize = 0.02;

    switch (cornerIndex) {
      case 0: // TL
        l = (l + dx).clamp(0.0, r - minSize);
        t = (t + dy).clamp(0.0, b - minSize);
      case 1: // TR
        r = (r + dx).clamp(l + minSize, 1.0);
        t = (t + dy).clamp(0.0, b - minSize);
      case 2: // BL
        l = (l + dx).clamp(0.0, r - minSize);
        b = (b + dy).clamp(t + minSize, 1.0);
      case 3: // BR
        r = (r + dx).clamp(l + minSize, 1.0);
        b = (b + dy).clamp(t + minSize, 1.0);
    }

    setState(() => _manualBoxRect = Rect.fromLTRB(l, t, r, b));
  }

  void _confirmManualBox() {
    if (_manualBoxRect == null) return;
    final label = _yoloService.labels.isNotEmpty ? _yoloService.labels.first : 'objeto';
    const classId = 0;
    final newBox = Recognition(
      classId,
      label,
      1.0,
      _manualBoxRect!,
    );
    setState(() {
      _results.add(newBox);
      _undoStack.add(_AddedDetections([newBox]));
      if (_undoStack.length > _maxUndoDepth) _undoStack.removeAt(0);
      _manualBoxRect = null;
      _isManualBoxActive = false;
      _activeHandleIndex = -1;
    });
  }

  void _cancelManualBox() {
    setState(() {
      _manualBoxRect = null;
      _isManualBoxActive = false;
      _activeHandleIndex = -1;
    });
  }

  /// Remover box existente
  void _removeBox(Recognition box) {
    final index = _results.indexOf(box);
    setState(() {
      _results.remove(box);
      _undoStack.add(_RemovedDetection(box, index));
      if (_undoStack.length > _maxUndoDepth) _undoStack.removeAt(0);
    });
  }

  void _undo() {
    if (_undoStack.isEmpty) return;
    setState(() {
      final action = _undoStack.removeLast();
      switch (action) {
        case _RemovedDetection(:final removed, :final originalIndex):
          final idx = originalIndex.clamp(0, _results.length);
          _results.insert(idx, removed);
        case _AddedDetections(:final added):
          _results.removeWhere((r) => added.contains(r));
      }
    });
  }

  /// Handler para tap na imagem (modo edição)
  void _handleImageTap(Offset tapPosition) {
    if (!_isEditMode) return;
    if (_isManualBoxActive) return;

    final hitBox = _hitTest(tapPosition);
    if (hitBox != null) {
      _removeBox(hitBox);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${hitBox.label} removido'),
          duration: const Duration(seconds: 1),
          backgroundColor: Colors.orange,
        ),
      );
    } else {
      if (_currentWidgetSize == null) return;
      final normalizedTap = Offset(
        tapPosition.dx / _currentWidgetSize!.width,
        tapPosition.dy / _currentWidgetSize!.height,
      );
      _runInferenceOnTap(normalizedTap);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        title: const Text('YOLOv8 Detector'),
        backgroundColor: const Color(0xFF16213E),
        foregroundColor: Colors.white,
        elevation: 0,
        actions: [
          if (!_modelReady)
            const Padding(
              padding: EdgeInsets.all(12.0),
              child: SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Colors.white,
                ),
              ),
            ),
          if (_modelReady)
            const Padding(
              padding: EdgeInsets.all(14.0),
              child: Icon(Icons.check_circle, color: Colors.greenAccent, size: 22),
            ),
        ],
      ),
      body: Column(
        children: [
          // --- Área da imagem ---
          Expanded(
            child: _imageFile == null
                ? const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.image_search, size: 72, color: Colors.white24),
                        SizedBox(height: 16),
                        Text(
                          'Selecione uma imagem para começar',
                          style: TextStyle(color: Colors.white38, fontSize: 16),
                        ),
                      ],
                    ),
                  )
                : Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      // LayoutBuilder para obter o tamanho real do widget
                      child: LayoutBuilder(
                        builder: (context, constraints) {
                          // Captura o tamanho atual do widget para conversão de coordenadas
                          WidgetsBinding.instance.addPostFrameCallback((_) {
                            _onWidgetSizeAvailable(Size(constraints.maxWidth, constraints.maxHeight));
                          });

                          // Calcula o aspect ratio real da imagem original
                          // para que as bounding boxes se alinhem corretamente
                          double aspectRatio = 1.0;
                          if (_decodedImage != null) {
                            aspectRatio =
                                _decodedImage!.width / _decodedImage!.height;
                          }

                          return InteractiveViewer(
                            transformationController: _transformationController,
                            // RenderTransform.hitTestChildren applies inverse transform automatically,
                            // so GestureDetector.localPosition inside this viewer is already in
                            // the child's coordinate space — no manual matrix inversion needed.
                            panEnabled: !_isRegionMode && !_isManualBoxActive,
                            scaleEnabled: true,
                            minScale: 1.0,
                            maxScale: 6.0,
                            boundaryMargin: EdgeInsets.zero,
                            child: AspectRatio(
                              aspectRatio: aspectRatio,
                              child: GestureDetector(
                              onPanStart: (_isRegionMode || _isManualBoxActive)
                                  ? (details) {
                                      if (_isRegionMode) {
                                        setState(() {
                                          _draggingRegion = Rect.fromLTWH(
                                            details.localPosition.dx / constraints.maxWidth,
                                            details.localPosition.dy / constraints.maxHeight,
                                            0, 0,
                                          );
                                        });
                                      } else if (_isManualBoxActive && _manualBoxRect != null && _currentWidgetSize != null) {
                                        setState(() {
                                          _activeHandleIndex = ManualBoxEditorPainter.getHandleIndex(
                                            details.localPosition,
                                            _manualBoxRect!,
                                            _currentWidgetSize!,
                                          );
                                        });
                                      }
                                    }
                                  : null,
                              onPanUpdate: (_isRegionMode || _isManualBoxActive)
                                  ? (details) {
                                      if (_isRegionMode) {
                                        setState(() {
                                          final left = min(_draggingRegion!.left, details.localPosition.dx / constraints.maxWidth);
                                          final top = min(_draggingRegion!.top, details.localPosition.dy / constraints.maxHeight);
                                          final right = max(_draggingRegion!.right, details.localPosition.dx / constraints.maxWidth);
                                          final bottom = max(_draggingRegion!.bottom, details.localPosition.dy / constraints.maxHeight);
                                          _draggingRegion = Rect.fromLTRB(
                                            left.clamp(0.0, 1.0),
                                            top.clamp(0.0, 1.0),
                                            right.clamp(0.0, 1.0),
                                            bottom.clamp(0.0, 1.0),
                                          );
                                        });
                                      } else if (_isManualBoxActive && _activeHandleIndex >= 0) {
                                        _updateManualBoxCorner(_activeHandleIndex, details.delta);
                                      }
                                    }
                                  : null,
                              onPanEnd: (_isRegionMode || _isManualBoxActive)
                                  ? (details) {
                                      if (_isRegionMode) {
                                        if (_draggingRegion != null &&
                                            _draggingRegion!.width > 0.02 &&
                                            _draggingRegion!.height > 0.02) {
                                          _runInferenceOnRegion(_draggingRegion!);
                                        }
                                        setState(() => _draggingRegion = null);
                                      } else if (_isManualBoxActive) {
                                        setState(() => _activeHandleIndex = -1);
                                      }
                                    }
                                  : null,
                              onTapUp: _isEditMode
                                  ? (details) => _handleImageTap(details.localPosition)
                                  : null,
                              onDoubleTap: _awaitingRegionSelection && _draggingRegion != null
                                  ? _confirmRegionAndProcess
                                  : null,
                              child: Stack(
                                fit: StackFit.expand,
                                children: [
                                  // Imagem de fundo
                                  Image.file(
                                    _imageFile!,
                                    fit: BoxFit.fill,
                                  ),
                                  // Bounding Boxes
                                  if (_results.isNotEmpty)
                                    CustomPaint(
                                      painter: BoundingBoxPainter(_results),
                                    ),
                                  // Manual box editor overlay
                                  if (_isManualBoxActive && _manualBoxRect != null)
                                    CustomPaint(
                                      painter: ManualBoxEditorPainter(_manualBoxRect!),
                                    ),
                                  // Região de seleção (durante drag)
                                  if (_draggingRegion != null && _isRegionMode)
                                    CustomPaint(
                                      painter: _RegionSelectorPainter(_draggingRegion!),
                                    ),
                                  // Overlay de instrução
                                  if ((_awaitingRegionSelection || _isRegionMode) && !_isProcessing)
                                    Container(
                                      color: Colors.black26,
                                      child: Center(
                                        child: Column(
                                          mainAxisSize: MainAxisSize.min,
                                          children: [
                                            Container(
                                              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                                              decoration: BoxDecoration(
                                                color: Colors.black87,
                                                borderRadius: BorderRadius.circular(8),
                                                border: Border.all(color: Colors.blueAccent, width: 2),
                                              ),
                                              child: Text(
                                                _awaitingRegionSelection
                                                    ? 'Arraste para selecionar área'
                                                    : 'Arraste para selecionar área',
                                                style: const TextStyle(
                                                  color: Colors.white,
                                                  fontSize: 16,
                                                  fontWeight: FontWeight.bold,
                                                ),
                                              ),
                                            ),
                                            if (_awaitingRegionSelection) ...[
                                              const SizedBox(height: 8),
                                              const Text(
                                                'ou toque em "Processar Área"',
                                                style: TextStyle(color: Colors.white70, fontSize: 14),
                                              ),
                                            ],
                                          ],
                                        ),
                                      ),
                                    ),
                                  if (_isEditMode && !_isProcessing && !_isManualBoxActive)
                                    Container(
                                      color: Colors.black26,
                                      child: Center(
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(
                                            horizontal: 20,
                                            vertical: 12,
                                          ),
                                          decoration: BoxDecoration(
                                            color: Colors.black87,
                                            borderRadius: BorderRadius.circular(8),
                                            border: Border.all(
                                              color: Colors.orange,
                                              width: 2,
                                            ),
                                          ),
                                          child: const Text(
                                            'Toque na box para remover • Toque vazio para re-detectar',
                                            style: TextStyle(
                                              color: Colors.white,
                                              fontSize: 16,
                                              fontWeight: FontWeight.bold,
                                            ),
                                          ),
                                        ),
                                      ),
                                    ),
                                  // Loading overlay
                                  if (_isProcessing)
                                    Container(
                                      color: Colors.black45,
                                      child: const Center(
                                        child: Column(
                                          mainAxisSize: MainAxisSize.min,
                                          children: [
                                            CircularProgressIndicator(
                                                color: Colors.white),
                                            SizedBox(height: 12),
                                            Text('Detectando...',
                                                style: TextStyle(
                                                    color: Colors.white)),
                                          ],
                                        ),
                                      ),
                                    ),
                                  // Manual box confirm/cancel buttons
                                  if (_isManualBoxActive)
                                    Positioned(
                                      bottom: 16,
                                      left: 0,
                                      right: 0,
                                      child: Row(
                                        mainAxisAlignment: MainAxisAlignment.center,
                                        children: [
                                          ElevatedButton.icon(
                                            onPressed: _confirmManualBox,
                                            icon: const Icon(Icons.check),
                                            label: const Text('Confirmar'),
                                            style: ElevatedButton.styleFrom(
                                              backgroundColor: Colors.green,
                                              foregroundColor: Colors.white,
                                            ),
                                          ),
                                          const SizedBox(width: 16),
                                          ElevatedButton.icon(
                                            onPressed: _cancelManualBox,
                                            icon: const Icon(Icons.close),
                                            label: const Text('Cancelar'),
                                            style: ElevatedButton.styleFrom(
                                              backgroundColor: Colors.red,
                                              foregroundColor: Colors.white,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                ],
                              ),
                            ),
                          ),
                          );
                        },
                      ),
                    ),
                  ),
          ),

          // --- Painel de resultados ---
          Container(
            width: double.infinity,
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF16213E),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.blueAccent.withAlpha(102)), 
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Detecções',
                  style: TextStyle(
                      color: Colors.white54,
                      fontSize: 12,
                      fontWeight: FontWeight.w500),
                ),
                const SizedBox(height: 4),
                Text(
                  _getSummary(),
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                if (_results.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(
                      'Total: ${_results.length} objeto(s)',
                      style: const TextStyle(
                          color: Colors.blueAccent, fontSize: 13),
                    ),
                  ),
              ],
            ),
          ),

          // --- Botões ---
          Padding(
            padding: const EdgeInsets.only(bottom: 36, top: 8),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Botão de processar (só aparece quando aguardando região)
                if (_imageFile != null && _awaitingRegionSelection && !_regionProcessed)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: ElevatedButton.icon(
                      onPressed: _draggingRegion != null ? _confirmRegionAndProcess : null,
                      icon: const Icon(Icons.check),
                      label: const Text('Processar Área'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _draggingRegion != null ? Colors.green : Colors.grey,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),

                // Botões de modo (linha superior)
                if (_imageFile != null && !_awaitingRegionSelection)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        _ToggleActionButton(
                          icon: Icons.crop_free,
                          label: 'Selecionar Área',
                          isActive: _isRegionMode,
                          onTap: () {
                            setState(() {
                              _isRegionMode = !_isRegionMode;
                              if (_isRegionMode) _isEditMode = false;
                            });
                          },
                        ),
                        _ToggleActionButton(
                          icon: Icons.edit,
                          label: 'Editar',
                          isActive: _isEditMode,
                          onTap: () {
                            setState(() {
                              _isEditMode = !_isEditMode;
                              if (_isEditMode) _isRegionMode = false;
                            });
                          },
                        ),
                        if (_isEditMode)
                          IconButton(
                            icon: Icon(
                              Icons.undo,
                              color: _undoStack.isNotEmpty ? Colors.white : Colors.white30,
                            ),
                            onPressed: _undoStack.isNotEmpty ? _undo : null,
                            tooltip: 'Desfazer',
                            style: IconButton.styleFrom(
                              backgroundColor: _undoStack.isNotEmpty
                                  ? Colors.blueAccent.withAlpha(77)
                                  : Colors.transparent,
                            ),
                          ),
                      ],
                    ),
                  ),
                // Botões de ação (linha inferior)
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    _ActionButton(
                      icon: Icons.photo_library_rounded,
                      label: 'Galeria',
                      onTap: () => _processImage(ImageSource.gallery),
                    ),
                    _ActionButton(
                      icon: Icons.camera_alt_rounded,
                      label: 'Câmera',
                      onTap: () => _processImage(ImageSource.camera),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Painter para o retângulo de seleção de região
class _RegionSelectorPainter extends CustomPainter {
  final Rect region; // Coordenadas normalizadas

  _RegionSelectorPainter(this.region);

  @override
  void paint(Canvas canvas, Size size) {
    // Converter para pixels
    final rect = Rect.fromLTRB(
      region.left * size.width,
      region.top * size.height,
      region.right * size.width,
      region.bottom * size.height,
    );

    // Fundo semi-transparente fora da região
    final outerPaint = Paint()
      ..color = Colors.black54
      ..style = PaintingStyle.fill;

    final path = Path()
      ..addRect(Rect.fromLTWH(0, 0, size.width, size.height))
      ..addRect(rect);

    canvas.drawPath(path, outerPaint);

    // Borda da região selecionada
    final borderPaint = Paint()
      ..color = Colors.blueAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    canvas.drawRect(rect, borderPaint);

    // Cantos da seleção (handles visuais)
    final handlePaint = Paint()..color = Colors.white;
    const handleSize = 8.0;

    canvas.drawRect(
      Rect.fromLTWH(rect.left - handleSize / 2, rect.top - handleSize / 2, handleSize, handleSize),
      handlePaint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.right - handleSize / 2, rect.top - handleSize / 2, handleSize, handleSize),
      handlePaint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.left - handleSize / 2, rect.bottom - handleSize / 2, handleSize, handleSize),
      handlePaint,
    );
    canvas.drawRect(
      Rect.fromLTWH(rect.right - handleSize / 2, rect.bottom - handleSize / 2, handleSize, handleSize),
      handlePaint,
    );
  }

  @override
  bool shouldRepaint(covariant _RegionSelectorPainter oldDelegate) =>
      oldDelegate.region != region;
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: onTap,
      icon: Icon(icon),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.blueAccent,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}

/// Botão de toggle para modos especiais
class _ToggleActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final bool isActive;

  const _ToggleActionButton({
    required this.icon,
    required this.label,
    required this.onTap,
    this.isActive = false,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: onTap,
      icon: Icon(icon, color: isActive ? Colors.yellow : null),
      label: Text(label, style: TextStyle(fontWeight: isActive ? FontWeight.bold : null)),
      style: ElevatedButton.styleFrom(
        backgroundColor: isActive ? Colors.orange : Colors.blueAccent,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        elevation: isActive ? 4 : 2,
      ),
    );
  }
}
