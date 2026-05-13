import 'dart:io';
import 'dart:math' show min, max;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'yolo_service.dart';
import 'widgets/box_painter.dart';
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

class _MovedDetection extends _EditAction {
  final Recognition oldDetection;
  final Recognition newDetection;
  _MovedDetection(this.oldDetection, this.newDetection);
}

class PhotoSession {
  final File imageFile;
  img.Image? decodedImage;
  List<Recognition> results;
  List<_EditAction> undoStack;
  List<Rect> savedRegions;
  bool awaitingRegionSelection;

  PhotoSession({
    required this.imageFile,
    this.decodedImage,
    List<Recognition>? results,
    List<_EditAction>? undoStack,
    List<Rect>? savedRegions,
    this.awaitingRegionSelection = true,
  })  : results = results ?? [],
        undoStack = undoStack ?? [],
        savedRegions = savedRegions ?? [];
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

  int? _draggingCircleIndex;
  Recognition? _draggingOriginalDetection;
  Offset? _draggingCenterOverride;
  bool _significantDrag = false;
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

  int? _hitTestCircle(Offset localPosition, {double extraPadding = 16.0}) {
    if (_currentWidgetSize == null || _results.isEmpty) return null;
    for (int i = _results.length - 1; i >= 0; i--) {
      final d = _results[i];
      final cx = (d.location.left + d.location.right) / 2 * _currentWidgetSize!.width;
      final cy = (d.location.top + d.location.bottom) / 2 * _currentWidgetSize!.height;
      final bw = d.location.width * _currentWidgetSize!.width;
      final bh = d.location.height * _currentWidgetSize!.height;
      final drawnRadius = (min(bw, bh) * 0.30).clamp(5.0, 11.0);
      final hitR = drawnRadius + extraPadding;
      final dx = localPosition.dx - cx;
      final dy = localPosition.dy - cy;
      if (dx * dx + dy * dy <= hitR * hitR) return i;
    }
    return null;
  }

  Size _averageDetectionSize() {
    if (_results.isEmpty) return const Size(0.06, 0.06);
    final avgW = _results.map((r) => r.location.width).reduce((a, b) => a + b) / _results.length;
    final avgH = _results.map((r) => r.location.height).reduce((a, b) => a + b) / _results.length;
    return Size(avgW, avgH);
  }

  void _createCircleAtPosition(Offset localPosition) {
    if (_currentWidgetSize == null) return;
    final center = Offset(
      (localPosition.dx / _currentWidgetSize!.width).clamp(0.0, 1.0),
      (localPosition.dy / _currentWidgetSize!.height).clamp(0.0, 1.0),
    );
    final s = _averageDetectionSize();
    final newDet = Recognition(
      0,
      _yoloService.labels.isNotEmpty ? _yoloService.labels.first : 'objeto',
      1.0,
      Rect.fromCenter(center: center, width: s.width.clamp(0.01, 0.5), height: s.height.clamp(0.01, 0.5)),
    );
    setState(() {
      _results.add(newDet);
      _undoStack.add(_AddedDetections([newDet]));
      if (_undoStack.length > _maxUndoDepth) _undoStack.removeAt(0);
    });
  }

  List<Recognition> get _resultsForDisplay {
    if (_draggingCircleIndex == null || _draggingCenterOverride == null) return _results;
    final list = List<Recognition>.from(_results);
    final old = list[_draggingCircleIndex!];
    list[_draggingCircleIndex!] = Recognition(
      old.classId, old.label, old.score,
      Rect.fromCenter(center: _draggingCenterOverride!, width: old.location.width, height: old.location.height),
      angle: old.angle,
    );
    return list;
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
        case _MovedDetection(:final oldDetection, :final newDetection):
          final idx = _results.indexOf(newDetection);
          if (idx >= 0) _results[idx] = oldDetection;
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
                    child: Container(
                      decoration: _isEditMode
                          ? BoxDecoration(
                              borderRadius: BorderRadius.circular(8),
                              border: Border.all(color: AppTheme.emerald, width: 2),
                            )
                          : null,
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(8),
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
                            panEnabled: !_isRegionMode && !_awaitingRegionSelection && !_isEditMode,
                            scaleEnabled: true,
                            minScale: 1.0,
                            maxScale: 6.0,
                            boundaryMargin: EdgeInsets.zero,
                            child: AspectRatio(
                              aspectRatio: aspectRatio,
                              child: GestureDetector(
                              onTapUp: _isEditMode
                                  ? (details) {
                                      final hitIdx = _hitTestCircle(details.localPosition, extraPadding: 0);
                                      if (hitIdx != null) {
                                        final removed = _results[hitIdx];
                                        _removeBox(removed);
                                        ScaffoldMessenger.of(context).showSnackBar(
                                          SnackBar(
                                            content: Text('${removed.label} removido'),
                                            duration: const Duration(seconds: 1),
                                          ),
                                        );
                                      } else {
                                        _createCircleAtPosition(details.localPosition);
                                      }
                                    }
                                  : null,
                              onPanStart: (_isRegionMode || _awaitingRegionSelection || _isEditMode)
                                  ? (details) {
                                      if (_isRegionMode || _awaitingRegionSelection) {
                                        setState(() {
                                          _draggingRegion = Rect.fromLTWH(
                                            details.localPosition.dx / constraints.maxWidth,
                                            details.localPosition.dy / constraints.maxHeight,
                                            0, 0,
                                          );
                                        });
                                      } else if (_isEditMode) {
                                        final hitIdx = _hitTestCircle(details.localPosition, extraPadding: 5);
                                        if (hitIdx != null) {
                                          setState(() {
                                            _draggingCircleIndex = hitIdx;
                                            _draggingOriginalDetection = _results[hitIdx];
                                            _draggingCenterOverride = Offset(
                                              _results[hitIdx].location.center.dx,
                                              _results[hitIdx].location.center.dy,
                                            );
                                          });
                                        }
                                      }
                                    }
                                  : null,
                              onPanUpdate: (_isRegionMode || _awaitingRegionSelection || _isEditMode)
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
                                      } else if (_isEditMode && _draggingCircleIndex != null) {
                                        final dx = details.delta.dx / constraints.maxWidth;
                                        final dy = details.delta.dy / constraints.maxHeight;
                                        setState(() {
                                          _draggingCenterOverride = Offset(
                                            (_draggingCenterOverride!.dx + dx).clamp(0.0, 1.0),
                                            (_draggingCenterOverride!.dy + dy).clamp(0.0, 1.0),
                                          );
                                          _significantDrag = true;
                                        });
                                      }
                                    }
                                  : null,
                              onPanEnd: (_isRegionMode || _awaitingRegionSelection || _isEditMode)
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
                                      } else if (_isEditMode && _draggingCircleIndex != null && _significantDrag) {
                                        final old = _draggingOriginalDetection!;
                                        final newDet = Recognition(
                                          old.classId, old.label, old.score,
                                          Rect.fromCenter(center: _draggingCenterOverride!, width: old.location.width, height: old.location.height),
                                          angle: old.angle,
                                        );
                                        setState(() {
                                          _results[_draggingCircleIndex!] = newDet;
                                          _undoStack.add(_MovedDetection(old, newDet));
                                          if (_undoStack.length > _maxUndoDepth) _undoStack.removeAt(0);
                                        });
                                      }
                                      if (_isEditMode) {
                                        setState(() {
                                          _draggingCircleIndex = null;
                                          _draggingOriginalDetection = null;
                                          _draggingCenterOverride = null;
                                          _significantDrag = false;
                                        });
                                      }
                                    }
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
                                        _resultsForDisplay,
                                        isDarkMode: widget.isDarkMode,
                                      ),
                                    ),
                                  // Região de seleção (regiões salvas + drag atual)
                                  if ((_savedRegions.isNotEmpty || _draggingRegion != null) && (_isRegionMode || _awaitingRegionSelection))
                                    CustomPaint(
                                      painter: _RegionSelectorPainter(_savedRegions, _draggingRegion),
                                    ),
                                  // Region selection instruction chip
                                  if ((_awaitingRegionSelection || _isRegionMode) && !_isProcessing)
                                    Positioned(
                                      bottom: 8,
                                      left: 0,
                                      right: 0,
                                      child: Center(
                                        child: Container(
                                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                                          decoration: BoxDecoration(
                                            color: widget.isDarkMode
                                                ? Colors.black.withValues(alpha: 0.72)
                                                : Colors.white.withValues(alpha: 0.90),
                                            borderRadius: BorderRadius.circular(20),
                                            border: Border.all(color: AppTheme.emerald, width: 1.5),
                                          ),
                                          child: Column(
                                            mainAxisSize: MainAxisSize.min,
                                            children: [
                                              Text(
                                                _awaitingRegionSelection
                                                    ? 'Arraste para selecionar área'
                                                    : 'Solte para confirmar',
                                                style: AppTheme.buttonStyle.copyWith(
                                                  color: Theme.of(context).colorScheme.onSurface,
                                                ),
                                              ),
                                              if (_awaitingRegionSelection)
                                                Padding(
                                                  padding: const EdgeInsets.only(top: 4),
                                                  child: Text(
                                                    'ou toque em "Detectar"',
                                                    style: AppTheme.secondaryStyle.copyWith(
                                                      color: Theme.of(context).colorScheme.secondary,
                                                    ),
                                                  ),
                                                ),
                                            ],
                                          ),
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
                                            width: 20,
                                            height: 20,
                                            decoration: const BoxDecoration(color: AppTheme.errorRed, shape: BoxShape.circle),
                                            child: const Icon(Icons.close, size: 12, color: Colors.white),
                                          ),
                                        ),
                                      ),
                                  if (_isEditMode)
                                    Positioned(
                                      bottom: 8,
                                      left: 0,
                                      right: 0,
                                      child: IgnorePointer(
                                        child: Center(
                                          child: Container(
                                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                                            decoration: BoxDecoration(
                                              borderRadius: BorderRadius.circular(20),
                                              color: widget.isDarkMode
                                                  ? Colors.black.withValues(alpha: 0.7)
                                                  : Colors.white.withValues(alpha: 0.9),
                                              border: Border.all(color: colorScheme.outline),
                                            ),
                                            child: Text(
                                              '✏  TOQUE PARA EDITAR',
                                              style: TextStyle(
                                                fontSize: 10,
                                                fontWeight: FontWeight.w600,
                                                color: colorScheme.outlineVariant,
                                                letterSpacing: 0.5,
                                              ),
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
                // B) Detectar button (process button)
                if (_imageFile != null && (_awaitingRegionSelection || _isRegionMode) && !_isProcessing)
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
                if (_imageFile != null && !_awaitingRegionSelection && !_isRegionMode)
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
        ..color = isSaved ? AppTheme.emerald : AppTheme.emerald.withValues(alpha: 0.7)
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
