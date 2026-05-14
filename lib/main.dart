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

  // Per-photo state lives in PhotoSession; _current is the active one
  final List<PhotoSession> _photos = [];
  int _currentIndex = 0;
  PhotoSession? get _current => _photos.isEmpty ? null : _photos[_currentIndex];

  bool _isProcessing = false;
  bool _modelReady = false;
  bool _modelError = false;

  // Global interaction state (not per-photo)
  bool _isRegionMode = false;
  bool _isEditMode = false;
  Rect? _draggingRegion;
  Size? _currentWidgetSize;
  static const int _maxUndoDepth = 20;
  final TransformationController _transformationController = TransformationController();

  int? _draggingCircleIndex;
  Recognition? _draggingOriginalDetection;
  Offset? _draggingCenterOverride;
  bool _significantDrag = false;

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

    // Decode immediately so the image viewer has the correct aspect ratio
    // before detection runs (avoids the 1:1 default causing empty space).
    final bytes = await File(picked.path).readAsBytes();
    final decoded = img.decodeImage(bytes);

    setState(() {
      _isProcessing = false;
      _isEditMode = false;
      _isRegionMode = false;
      _transformationController.value = Matrix4.identity();
      _photos.add(PhotoSession(imageFile: File(picked.path), decodedImage: decoded));
      _currentIndex = _photos.length - 1;
    });
  }

  /// Callback para armazenar o tamanho do widget quando disponível
  void _onWidgetSizeAvailable(Size size) {
    setState(() => _currentWidgetSize = size);
  }

  /// Monta o texto de resumo com contagem por classe
  String _getSummary() {
    if (_isProcessing) return 'Processando...';
    final results = _current?.results ?? [];
    if (results.isEmpty) return 'Nenhum objeto detectado.';

    final Map<String, int> counts = {};
    for (var r in results) {
      counts[r.label] = (counts[r.label] ?? 0) + 1;
    }
    return counts.entries.map((e) => '${e.value}x ${e.key}').join('  |  ');
  }

  /// Confirma as regiões selecionadas e processa
  Future<void> _confirmRegionAndProcess() async {
    if (_current == null) return;

    setState(() {
      _current!.awaitingRegionSelection = false;
      _isRegionMode = false;
      _isProcessing = true;
      _draggingRegion = null;
    });

    try {
      final decoded = _current!.decodedImage ??
          img.decodeImage(await _current!.imageFile.readAsBytes());

      if (decoded != null) {
        final regions = _current!.savedRegions.isNotEmpty
            ? List<Rect>.from(_current!.savedRegions)
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
          _current!.decodedImage = decoded;
          _current!.results = allDetections;
          _current!.undoStack.clear();
          _isProcessing = false;
        });
      }
    } catch (e) {
      debugPrint('Erro no processamento: $e');
      setState(() => _isProcessing = false);
    }
  }

  int? _hitTestCircle(Offset localPosition, {double extraPadding = 16.0}) {
    if (_currentWidgetSize == null || _current == null) return null;
    final results = _current!.results;
    for (int i = results.length - 1; i >= 0; i--) {
      final d = results[i];
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
    final results = _current?.results ?? [];
    if (results.isEmpty) return const Size(0.06, 0.06);
    final avgW = results.map((r) => r.location.width).reduce((a, b) => a + b) / results.length;
    final avgH = results.map((r) => r.location.height).reduce((a, b) => a + b) / results.length;
    return Size(avgW, avgH);
  }

  void _createCircleAtPosition(Offset localPosition) {
    if (_currentWidgetSize == null || _current == null) return;
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
      _current!.results.add(newDet);
      _current!.undoStack.add(_AddedDetections([newDet]));
      if (_current!.undoStack.length > _maxUndoDepth) _current!.undoStack.removeAt(0);
    });
  }

  List<Recognition> get _resultsForDisplay {
    final results = _current?.results ?? [];
    if (_draggingCircleIndex == null || _draggingCenterOverride == null) return results;
    final list = List<Recognition>.from(results);
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
    if (_current == null) return;
    final index = _current!.results.indexOf(box);
    setState(() {
      _current!.results.remove(box);
      _current!.undoStack.add(_RemovedDetection(box, index));
      if (_current!.undoStack.length > _maxUndoDepth) _current!.undoStack.removeAt(0);
    });
  }

  void _undo() {
    if (_current == null || _current!.undoStack.isEmpty) return;
    setState(() {
      final action = _current!.undoStack.removeLast();
      switch (action) {
        case _RemovedDetection(:final removed, :final originalIndex):
          final idx = originalIndex.clamp(0, _current!.results.length);
          _current!.results.insert(idx, removed);
        case _AddedDetections(:final added):
          _current!.results.removeWhere((r) => added.contains(r));
        case _MovedDetection(:final oldDetection, :final newDetection):
          final idx = _current!.results.indexOf(newDetection);
          if (idx >= 0) _current!.results[idx] = oldDetection;
      }
    });
  }

  void _navigateTo(int index) {
    setState(() {
      _currentIndex = index;
      _isEditMode = false;
      _isRegionMode = false;
      _transformationController.value = Matrix4.identity();
    });
  }

  Future<void> _confirmDeletePhoto(int index) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Remover foto?'),
        content: const Text('Esta foto e suas detecções serão removidas.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancelar')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Remover', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    setState(() {
      _photos.removeAt(index);
      if (_photos.isEmpty) {
        _currentIndex = 0;
      } else {
        _currentIndex = _currentIndex.clamp(0, _photos.length - 1);
      }
    });
  }

  Future<void> _confirmNewSession() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Nova sessão'),
        content: const Text('Limpar todas as fotos e começar do zero?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancelar')),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Limpar', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    setState(() {
      _photos.clear();
      _currentIndex = 0;
      _isEditMode = false;
      _isRegionMode = false;
      _transformationController.value = Matrix4.identity();
    });
  }

  void _showImageSourceSheet() {
    showModalBottomSheet<void>(
      context: context,
      builder: (ctx) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.photo_library_rounded),
              title: const Text('Galeria'),
              onTap: () {
                Navigator.pop(ctx);
                _processImage(ImageSource.gallery);
              },
            ),
            ListTile(
              leading: const Icon(Icons.camera_alt_rounded),
              title: const Text('Câmera'),
              onTap: () {
                Navigator.pop(ctx);
                _processImage(ImageSource.camera);
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildThumbnailStrip() {
    return SizedBox(
      height: 48,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        padding: const EdgeInsets.symmetric(horizontal: 12),
        itemCount: _photos.length + 1,
        itemBuilder: (context, i) {
          if (i == _photos.length) {
            return Padding(
              padding: const EdgeInsets.only(left: 4),
              child: GestureDetector(
                onTap: _showImageSourceSheet,
                child: Container(
                  width: 40,
                  height: 32,
                  decoration: BoxDecoration(
                    color: AppTheme.emerald.withValues(alpha: 0.75),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: const Icon(Icons.add, color: Colors.white, size: 20),
                ),
              ),
            );
          }
          final isActive = i == _currentIndex;
          final count = _photos[i].results.length;
          return Padding(
            padding: const EdgeInsets.only(right: 4),
            child: GestureDetector(
              onTap: () => _navigateTo(i),
              onLongPress: () => _confirmDeletePhoto(i),
              child: Stack(
                alignment: Alignment.bottomRight,
                children: [
                  Container(
                    width: 40,
                    height: 32,
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(4),
                      border: isActive
                          ? Border.all(color: AppTheme.emerald, width: 2)
                          : Border.all(color: const Color(0xFF444444), width: 1),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(isActive ? 2 : 3),
                      child: Image.file(
                        _photos[i].imageFile,
                        fit: BoxFit.cover,
                      ),
                    ),
                  ),
                  Container(
                    margin: const EdgeInsets.all(2),
                    padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 1),
                    decoration: BoxDecoration(
                      color: Colors.black.withValues(alpha: 0.65),
                      borderRadius: BorderRadius.circular(3),
                    ),
                    child: Text(
                      '$count',
                      style: const TextStyle(
                        color: AppTheme.emerald,
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
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
          // Nova sessão button
          if (_photos.isNotEmpty)
            TextButton(
              onPressed: _confirmNewSession,
              child: Text(
                'Nova sessão',
                style: TextStyle(color: colorScheme.onSurfaceVariant, fontSize: 13),
              ),
            ),
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
            child: _current == null
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
                    child: Stack(fit: StackFit.expand, children: [
                      Container(
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
                          if (_current?.decodedImage != null) {
                            aspectRatio =
                                _current!.decodedImage!.width / _current!.decodedImage!.height;
                          }

                          return InteractiveViewer(
                            transformationController: _transformationController,
                            // RenderTransform.hitTestChildren applies inverse transform automatically,
                            // so GestureDetector.localPosition inside this viewer is already in
                            // the child's coordinate space — no manual matrix inversion needed.
                            panEnabled: !_isRegionMode && !(_current?.awaitingRegionSelection ?? false) && !_isEditMode,
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
                                        final removed = _current!.results[hitIdx];
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
                              onPanStart: (_isRegionMode || (_current?.awaitingRegionSelection ?? false) || _isEditMode)
                                  ? (details) {
                                      if (_isRegionMode || (_current?.awaitingRegionSelection ?? false)) {
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
                                            _draggingOriginalDetection = _current!.results[hitIdx];
                                            _draggingCenterOverride = Offset(
                                              _current!.results[hitIdx].location.center.dx,
                                              _current!.results[hitIdx].location.center.dy,
                                            );
                                          });
                                        }
                                      }
                                    }
                                  : null,
                              onPanUpdate: (_isRegionMode || (_current?.awaitingRegionSelection ?? false) || _isEditMode)
                                  ? (details) {
                                      if (_isRegionMode || (_current?.awaitingRegionSelection ?? false)) {
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
                              onPanEnd: (_isRegionMode || (_current?.awaitingRegionSelection ?? false) || _isEditMode)
                                  ? (details) {
                                      if (_isRegionMode || (_current?.awaitingRegionSelection ?? false)) {
                                        if (_draggingRegion != null &&
                                            _draggingRegion!.width > 0.02 &&
                                            _draggingRegion!.height > 0.02) {
                                          setState(() {
                                            _current!.savedRegions.add(_draggingRegion!);
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
                                          _current!.results[_draggingCircleIndex!] = newDet;
                                          _current!.undoStack.add(_MovedDetection(old, newDet));
                                          if (_current!.undoStack.length > _maxUndoDepth) _current!.undoStack.removeAt(0);
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
                              onDoubleTap: (_current?.awaitingRegionSelection ?? false) && _draggingRegion != null
                                  ? _confirmRegionAndProcess
                                  : null,
                              child: Stack(
                                fit: StackFit.expand,
                                children: [
                                  // Imagem de fundo
                                  Image.file(
                                    _current!.imageFile,
                                    fit: BoxFit.fill,
                                  ),
                                  // Bounding Boxes
                                  if (_current!.results.isNotEmpty)
                                    CustomPaint(
                                      painter: BoundingBoxPainter(
                                        _resultsForDisplay,
                                        isDarkMode: widget.isDarkMode,
                                      ),
                                    ),
                                  // Região de seleção (regiões salvas + drag atual)
                                  if (((_current?.savedRegions.isNotEmpty ?? false) || _draggingRegion != null) && (_isRegionMode || (_current?.awaitingRegionSelection ?? false)))
                                    CustomPaint(
                                      painter: _RegionSelectorPainter(_current!.savedRegions, _draggingRegion),
                                    ),
                                  // Region selection instruction chip
                                  if (((_current?.awaitingRegionSelection ?? false) || _isRegionMode) && !_isProcessing)
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
                                                (_current?.awaitingRegionSelection ?? false)
                                                    ? 'Arraste para selecionar área'
                                                    : 'Solte para confirmar',
                                                style: AppTheme.buttonStyle.copyWith(
                                                  color: Theme.of(context).colorScheme.onSurface,
                                                ),
                                              ),
                                              if (_current?.awaitingRegionSelection ?? false)
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
                                  if (_isRegionMode || (_current?.awaitingRegionSelection ?? false))
                                    for (int i = 0; i < (_current?.savedRegions.length ?? 0); i++)
                                      Positioned(
                                        left: (_current!.savedRegions[i].right * constraints.maxWidth - 16).clamp(0.0, constraints.maxWidth - 28),
                                        top: (_current!.savedRegions[i].top * constraints.maxHeight - 16).clamp(0.0, constraints.maxHeight - 28),
                                        child: GestureDetector(
                                          behavior: HitTestBehavior.opaque,
                                          onTap: () => setState(() {
                                            final region = _current!.savedRegions.removeAt(i);
                                            _current!.results.removeWhere((r) {
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
                  // Left navigation arrow
                  if (_currentIndex > 0)
                    Positioned(
                      left: 4,
                      top: 0,
                      bottom: 0,
                      child: Center(
                        child: GestureDetector(
                          onTap: () => _navigateTo(_currentIndex - 1),
                          child: Container(
                            width: 28,
                            height: 28,
                            decoration: BoxDecoration(
                              color: AppTheme.emerald.withValues(alpha: 0.85),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.chevron_left, color: Colors.white, size: 20),
                          ),
                        ),
                      ),
                    ),
                  // Right navigation arrow
                  if (_currentIndex < _photos.length - 1)
                    Positioned(
                      right: 4,
                      top: 0,
                      bottom: 0,
                      child: Center(
                        child: GestureDetector(
                          onTap: () => _navigateTo(_currentIndex + 1),
                          child: Container(
                            width: 28,
                            height: 28,
                            decoration: BoxDecoration(
                              color: AppTheme.emerald.withValues(alpha: 0.85),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.chevron_right, color: Colors.white, size: 20),
                          ),
                        ),
                      ),
                    ),
                  ],
                  ),
                ),
          ),

          // --- Thumbnail strip ---
          if (_photos.isNotEmpty) _buildThumbnailStrip(),

          // --- Detection card (only when image loaded) ---
          if (_current != null)
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
                    if ((_current?.results.isNotEmpty ?? false))
                      Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text(
                          'Foto atual: ${_current!.results.length} peça(s)',
                          style: AppTheme.secondaryStyle.copyWith(color: colorScheme.onSurfaceVariant),
                        ),
                      ),
                    if (_photos.length > 1)
                      Padding(
                        padding: const EdgeInsets.only(top: 2),
                        child: Text(
                          'Total da sessão: ${_photos.fold(0, (sum, p) => sum + p.results.length)} peça(s)',
                          style: AppTheme.secondaryStyle.copyWith(color: AppTheme.emerald),
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
                if (_current != null && ((_current?.awaitingRegionSelection ?? false) || _isRegionMode) && !_isProcessing)
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                    child: SizedBox(
                      width: double.infinity,
                      height: 48,
                      child: ElevatedButton.icon(
                        onPressed: _confirmRegionAndProcess,
                        icon: const Icon(Icons.check),
                        label: Text(
                          (_current?.savedRegions.isEmpty ?? true) ? 'Detectar' : 'Detectar ${_current!.savedRegions.length} Área(s)',
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
                if (_current != null && !(_current?.awaitingRegionSelection ?? false) && !_isRegionMode)
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
                        if (_isEditMode && (_current?.undoStack.isNotEmpty ?? false))
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
