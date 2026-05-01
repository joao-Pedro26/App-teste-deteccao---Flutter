import 'dart:io';
import 'dart:math' show cos, sin, min, max;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'yolo_service.dart';
import 'widgets/box_painter.dart';
import 'widgets/manual_box_editor.dart';
import 'theme/app_theme.dart';

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

void main() => runApp(const FiscalizaApp());

class FiscalizaApp extends StatefulWidget {
  const FiscalizaApp({super.key});

  @override
  State<FiscalizaApp> createState() => _FiscalizaAppState();
}

class _FiscalizaAppState extends State<FiscalizaApp> {
  ThemeMode _themeMode = ThemeMode.system;

  void toggleTheme() {
    setState(() {
      _themeMode =
          _themeMode == ThemeMode.dark ? ThemeMode.light : ThemeMode.dark;
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: AppTheme.light(),
      darkTheme: AppTheme.dark(),
      themeMode: _themeMode,
      debugShowCheckedModeBanner: false,
      home: YoloApp(
        onToggleTheme: toggleTheme,
        isDarkMode: _themeMode == ThemeMode.dark,
      ),
    );
  }
}

class YoloApp extends StatefulWidget {
  final VoidCallback onToggleTheme;
  final bool isDarkMode;

  const YoloApp({
    super.key,
    required this.onToggleTheme,
    required this.isDarkMode,
  });

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
  bool _modelError = false;

  // Novos estados para interação
  bool _isRegionMode = false;      // Modo de seleção por região
  bool _isEditMode = false;        // Modo de edição de boxes
  Rect? _draggingRegion;           // Retângulo sendo desenhado (coordenadas normalizadas)
  Size? _currentWidgetSize;        // Tamanho atual do widget para conversão de coordenadas
  bool _awaitingRegionSelection = false;  // Aguarda seleção de região antes de processar
  final List<_EditAction> _undoStack = [];
  static const int _maxUndoDepth = 20;
  final TransformationController _transformationController = TransformationController();

  Rect? _manualBoxRect;
  bool _isManualBoxActive = false;
  int _activeHandleIndex = -1;
  List<Rect> _savedRegions = [];

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
    try {
      await _yoloService.init();
      setState(() => _modelReady = true);
    } catch (e) {
      debugPrint('[YOLO] Failed to load model: $e');
      setState(() => _modelError = true);
    }
  }

  Future<void> _processImage(ImageSource source) async {
    if (!_modelReady) {
      final msg = _modelError
          ? 'Erro ao carregar modelo. Reinicie o app.'
          : 'Modelo ainda carregando, aguarde...';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(msg)),
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
      _savedRegions = [];
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

  /// Confirma as regiões selecionadas e processa
  Future<void> _confirmRegionAndProcess() async {
    if (_imageFile == null) return;

    setState(() {
      _awaitingRegionSelection = false;
      _isRegionMode = false;
      _isProcessing = true;
      _draggingRegion = null;
    });

    try {
      final bytes = await _imageFile!.readAsBytes();
      final decoded = img.decodeImage(bytes);

      if (decoded != null) {
        final regions = _savedRegions.isNotEmpty
            ? List<Rect>.from(_savedRegions)
            : [Rect.fromLTWH(0, 0, 1, 1)];

        final List<Recognition> allDetections = [];

        for (final region in regions) {
          List<Recognition> detections;

          if (region.left == 0 && region.top == 0 && region.right == 1 && region.bottom == 1) {
            detections = await _yoloService.runInference(decoded);
          } else {
            final int rx = (region.left * decoded.width).round();
            final int ry = (region.top * decoded.height).round();
            final int rw = (region.width * decoded.width).round().clamp(1, decoded.width);
            final int rh = (region.height * decoded.height).round().clamp(1, decoded.height);

            final cropped = img.copyCrop(decoded, x: rx, y: ry, width: rw, height: rh);
            final regional = await _yoloService.runInference(cropped);

            detections = regional.map((d) => Recognition(
              d.classId, d.label, d.score,
              Rect.fromLTRB(
                region.left + d.location.left * region.width,
                region.top + d.location.top * region.height,
                region.left + d.location.right * region.width,
                region.top + d.location.bottom * region.height,
              ),
              angle: d.angle,
            )).toList();
          }

          for (final newD in detections) {
            final isDup = allDetections.any(
              (e) => YoloService.iou(newD.location, e.location) > YoloService.nmsThreshold,
            );
            if (!isDup) allDetections.add(newD);
          }
        }

        setState(() {
          _decodedImage = decoded;
          _results = allDetections;
          _undoStack.clear();
          _isProcessing = false;
        });
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

  ButtonStyle _ghostButtonStyle({
    required bool isActive,
    required ColorScheme colorScheme,
    required bool isDarkMode,
  }) {
    if (isActive) {
      return OutlinedButton.styleFrom(
        side: const BorderSide(color: AppTheme.emerald, width: 1.5),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
        minimumSize: const Size.fromHeight(40),
        backgroundColor: isDarkMode ? AppTheme.activeDarkBg : AppTheme.activeLightBg,
        foregroundColor: isDarkMode ? AppTheme.activeDarkText : AppTheme.activeLightText,
      );
    }
    return OutlinedButton.styleFrom(
      side: BorderSide(color: colorScheme.outline, width: 1.5),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      minimumSize: const Size.fromHeight(40),
      foregroundColor: colorScheme.onSurfaceVariant,
    );
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
      _openManualBoxEditor(normalizedTap);
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: Text('Fiscaliza', style: AppTheme.titleStyle.copyWith(color: colorScheme.onSurface)),
        elevation: 0,
        scrolledUnderElevation: 0,
        backgroundColor: colorScheme.surface,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Divider(height: 1, thickness: 1, color: colorScheme.outline),
        ),
        actions: [
          // Theme toggle button
          IconButton(
            icon: Icon(
              widget.isDarkMode ? Icons.light_mode : Icons.dark_mode,
              color: colorScheme.onSurfaceVariant,
            ),
            onPressed: widget.onToggleTheme,
            tooltip: widget.isDarkMode ? 'Modo claro' : 'Modo escuro',
          ),
          // Model status indicator
          Padding(
            padding: const EdgeInsets.all(12.0),
            child: _modelError
                ? Icon(Icons.error_outline, color: AppTheme.errorRed, size: 20)
                : !_modelReady
                    ? SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: colorScheme.onSurfaceVariant,
                        ),
                      )
                    : Container(
                        width: 8,
                        height: 8,
                        decoration: const BoxDecoration(
                          color: AppTheme.emerald,
                          shape: BoxShape.circle,
                        ),
                      ),
          ),
        ],
      ),
      body: Column(
        children: [
          // --- Área da imagem ---
          Expanded(
            child: _imageFile == null
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(Icons.image_search_outlined, size: 64, color: colorScheme.outlineVariant),
                        const SizedBox(height: 16),
                        Text(
                          'Selecione uma imagem para começar',
                          style: AppTheme.secondaryStyle.copyWith(color: colorScheme.onSurfaceVariant),
                          textAlign: TextAlign.center,
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
                            panEnabled: !_isRegionMode && !_isManualBoxActive && !_awaitingRegionSelection,
                            scaleEnabled: true,
                            minScale: 1.0,
                            maxScale: 6.0,
                            boundaryMargin: EdgeInsets.zero,
                            child: AspectRatio(
                              aspectRatio: aspectRatio,
                              child: GestureDetector(
                              onPanStart: (_isRegionMode || _awaitingRegionSelection || _isManualBoxActive)
                                  ? (details) {
                                      if (_isRegionMode || _awaitingRegionSelection) {
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
                              onPanUpdate: (_isRegionMode || _awaitingRegionSelection || _isManualBoxActive)
                                  ? (details) {
                                      if (_isRegionMode || _awaitingRegionSelection) {
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
                              onPanEnd: (_isRegionMode || _awaitingRegionSelection || _isManualBoxActive)
                                  ? (details) {
                                      if (_isRegionMode || _awaitingRegionSelection) {
                                        if (_draggingRegion != null &&
                                            _draggingRegion!.width > 0.02 &&
                                            _draggingRegion!.height > 0.02) {
                                          setState(() {
                                            _savedRegions.add(_draggingRegion!);
                                            _draggingRegion = null;
                                          });
                                        } else {
                                          setState(() => _draggingRegion = null);
                                        }
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
                                      painter: BoundingBoxPainter(
                                        _results,
                                        isDarkMode: true, // TODO Task 10: wire widget.isDarkMode
                                      ),
                                    ),
                                  // Manual box editor overlay
                                  if (_isManualBoxActive && _manualBoxRect != null)
                                    CustomPaint(
                                      painter: ManualBoxEditorPainter(_manualBoxRect!),
                                    ),
                                  // Região de seleção (regiões salvas + drag atual)
                                  if ((_savedRegions.isNotEmpty || _draggingRegion != null) && (_isRegionMode || _awaitingRegionSelection))
                                    CustomPaint(
                                      painter: _RegionSelectorPainter(_savedRegions, _draggingRegion),
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
                                                    : 'Solte para confirmar',
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
                                  // X buttons para excluir regiões (acima do overlay para receber toques)
                                  if (_isRegionMode || _awaitingRegionSelection)
                                    for (int i = 0; i < _savedRegions.length; i++)
                                      Positioned(
                                        left: (_savedRegions[i].right * constraints.maxWidth - 16).clamp(0.0, constraints.maxWidth - 28),
                                        top: (_savedRegions[i].top * constraints.maxHeight - 16).clamp(0.0, constraints.maxHeight - 28),
                                        child: GestureDetector(
                                          behavior: HitTestBehavior.opaque,
                                          onTap: () => setState(() {
                                            final region = _savedRegions.removeAt(i);
                                            _results.removeWhere((r) {
                                              final cx = (r.location.left + r.location.right) / 2;
                                              final cy = (r.location.top + r.location.bottom) / 2;
                                              return region.contains(Offset(cx, cy));
                                            });
                                          }),
                                          child: Container(
                                            width: 32,
                                            height: 32,
                                            decoration: const BoxDecoration(color: Colors.red, shape: BoxShape.circle),
                                            child: const Icon(Icons.close, size: 18, color: Colors.white),
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

          // --- Detection card (only when image loaded) ---
          if (_imageFile != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: colorScheme.surfaceContainerLow,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: colorScheme.outline),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'DETECÇÕES',
                      style: AppTheme.labelStyle.copyWith(color: colorScheme.outlineVariant),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _getSummary(),
                      style: AppTheme.valueStyle.copyWith(color: colorScheme.onSurface),
                    ),
                    if (_results.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(
                          'Total: ${_results.length} objeto(s)',
                          style: AppTheme.secondaryStyle.copyWith(color: colorScheme.onSurfaceVariant),
                        ),
                      ),
                  ],
                ),
              ),
            ),

          // --- Botões ---
          Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // A) Manual box Confirm/Cancel (replaces toolbar + FABs)
                if (_isManualBoxActive)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                    child: Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _confirmManualBox,
                            icon: const Icon(Icons.check),
                            label: const Text('Confirmar'),
                            style: OutlinedButton.styleFrom(
                              side: const BorderSide(color: AppTheme.emerald, width: 1.5),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                              minimumSize: const Size.fromHeight(40),
                              backgroundColor: widget.isDarkMode ? AppTheme.activeDarkBg : AppTheme.activeLightBg,
                              foregroundColor: widget.isDarkMode ? AppTheme.activeDarkText : AppTheme.activeLightText,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: _cancelManualBox,
                            icon: const Icon(Icons.close),
                            label: const Text('Cancelar'),
                            style: OutlinedButton.styleFrom(
                              side: BorderSide(color: AppTheme.errorRed, width: 1.5),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                              minimumSize: const Size.fromHeight(40),
                              backgroundColor: AppTheme.errorRed.withValues(alpha: 0.10),
                              foregroundColor: AppTheme.errorRed,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                // B) Detectar button (process button)
                if (!_isManualBoxActive && _imageFile != null && (_awaitingRegionSelection || _isRegionMode) && !_isProcessing)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                    child: SizedBox(
                      width: double.infinity,
                      height: 48,
                      child: ElevatedButton.icon(
                        onPressed: _confirmRegionAndProcess,
                        icon: const Icon(Icons.check),
                        label: Text(
                          _savedRegions.isEmpty ? 'Detectar' : 'Detectar ${_savedRegions.length} Área(s)',
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.emerald,
                          foregroundColor: Colors.white,
                          textStyle: AppTheme.buttonStyle,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(50)),
                        ),
                      ),
                    ),
                  ),

                // C) Ghost toolbar
                if (_imageFile != null && !_isManualBoxActive && !_awaitingRegionSelection && !_isRegionMode)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                    child: Row(
                      children: [
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: () => setState(() {
                              _isRegionMode = !_isRegionMode;
                              if (_isRegionMode) _isEditMode = false;
                            }),
                            icon: const Icon(Icons.crop_free, size: 18),
                            label: const Text('Selecionar Área'),
                            style: _ghostButtonStyle(isActive: _isRegionMode, colorScheme: colorScheme, isDarkMode: widget.isDarkMode),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: OutlinedButton.icon(
                            onPressed: () => setState(() {
                              _isEditMode = !_isEditMode;
                              if (_isEditMode) _isRegionMode = false;
                            }),
                            icon: const Icon(Icons.edit, size: 18),
                            label: const Text('Editar'),
                            style: _ghostButtonStyle(isActive: _isEditMode, colorScheme: colorScheme, isDarkMode: widget.isDarkMode),
                          ),
                        ),
                        if (_isEditMode && _undoStack.isNotEmpty)
                          Padding(
                            padding: const EdgeInsets.only(left: 8),
                            child: SizedBox(
                              width: 40,
                              height: 40,
                              child: OutlinedButton(
                                onPressed: _undo,
                                style: OutlinedButton.styleFrom(
                                  side: BorderSide(color: colorScheme.outline, width: 1.5),
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                  padding: EdgeInsets.zero,
                                  foregroundColor: colorScheme.onSurfaceVariant,
                                ),
                                child: const Icon(Icons.undo, size: 18),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),

                // D) FAB row
                if (!_isManualBoxActive)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => _processImage(ImageSource.gallery),
                            icon: const Icon(Icons.photo_library_rounded),
                            label: Text('Galeria', style: AppTheme.buttonStyle),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.emerald,
                              foregroundColor: Colors.white,
                              minimumSize: const Size.fromHeight(48),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(50)),
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => _processImage(ImageSource.camera),
                            icon: const Icon(Icons.camera_alt_rounded),
                            label: Text('Câmera', style: AppTheme.buttonStyle),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.emerald,
                              foregroundColor: Colors.white,
                              minimumSize: const Size.fromHeight(48),
                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(50)),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Painter para os retângulos de seleção de região
class _RegionSelectorPainter extends CustomPainter {
  final List<Rect> savedRegions;
  final Rect? draggingRegion;

  _RegionSelectorPainter(this.savedRegions, this.draggingRegion);

  @override
  void paint(Canvas canvas, Size size) {
    final allRegions = [
      ...savedRegions,
      ?draggingRegion,
    ];

    for (int i = 0; i < allRegions.length; i++) {
      final region = allRegions[i];
      final rect = Rect.fromLTRB(
        region.left * size.width,
        region.top * size.height,
        region.right * size.width,
        region.bottom * size.height,
      );

      // Draw a colored border for each region (saved = solid, dragging = dashed-ish)
      final isSaved = i < savedRegions.length;
      final borderPaint = Paint()
        ..color = isSaved ? Colors.blueAccent : Colors.blueAccent.withAlpha(180)
        ..style = PaintingStyle.stroke
        ..strokeWidth = isSaved ? 2.5 : 1.5;

      canvas.drawRect(rect, borderPaint);

      // Corner handles
      final handlePaint = Paint()..color = Colors.white;
      const handleSize = 8.0;
      for (final corner in [
        Offset(rect.left, rect.top),
        Offset(rect.right, rect.top),
        Offset(rect.left, rect.bottom),
        Offset(rect.right, rect.bottom),
      ]) {
        canvas.drawRect(
          Rect.fromCenter(center: corner, width: handleSize, height: handleSize),
          handlePaint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant _RegionSelectorPainter oldDelegate) =>
      oldDelegate.savedRegions != savedRegions || oldDelegate.draggingRegion != draggingRegion;
}
