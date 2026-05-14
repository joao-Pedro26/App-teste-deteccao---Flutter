# Multi-Photo Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow the user to accumulate detections across multiple photos in a scrollable carousel with per-photo editing and a running total.

**Architecture:** Add a `PhotoSession` data class that holds all per-photo state (image file, decoded image, detections, undo stack, regions). `_YoloAppState` replaces its single-image fields with `List<PhotoSession> _photos` + `int _currentIndex`. All existing methods that read/write per-photo state are updated to go through a `_current` convenience getter.

**Tech Stack:** Flutter (Dart), existing dependencies only — no new packages required.

---

## File Structure

| File | Change |
|---|---|
| `lib/main.dart` | Only file modified. Add `PhotoSession` class, replace state variables, add thumbnail strip + arrows + session total + "Nova sessão" button. |
| `lib/widgets/box_painter.dart` | No changes — already receives `detections` as a parameter. |
| `lib/yolo_service.dart` | No changes. |
| `test/widget_test.dart` | Add unit tests for `PhotoSession` and navigation logic. |

---

## Task 1: Add `PhotoSession` class

**Files:**
- Modify: `lib/main.dart` (after line 27, before `void main()`)
- Modify: `test/widget_test.dart`

- [ ] **Step 1: Write the failing test**

Open `test/widget_test.dart` and replace its contents with:

```dart
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:tryingg_flutter/main.dart';

void main() {
  group('PhotoSession', () {
    test('initialises with empty lists and awaitingRegionSelection true', () {
      final session = PhotoSession(imageFile: File('fake.jpg'));
      expect(session.results, isEmpty);
      expect(session.undoStack, isEmpty);
      expect(session.savedRegions, isEmpty);
      expect(session.awaitingRegionSelection, isTrue);
      expect(session.decodedImage, isNull);
    });
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

```
flutter test test/widget_test.dart
```

Expected: FAIL — `PhotoSession` not defined.

- [ ] **Step 3: Add `PhotoSession` class to `lib/main.dart`**

Insert after line 27 (after the `_MovedDetection` class, before `void main()`):

```dart
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
```

Also add `export 'main.dart' show PhotoSession;` — actually, for the test to access `PhotoSession`, the class just needs to be public (no underscore). It already is.

- [ ] **Step 4: Run test to verify it passes**

```
flutter test test/widget_test.dart
```

Expected: PASS — 1 test.

- [ ] **Step 5: Commit**

```
git add lib/main.dart test/widget_test.dart
git commit -m "feat: add PhotoSession data class"
```

---

## Task 2: Replace per-photo state variables with `_photos` list

**Files:**
- Modify: `lib/main.dart` lines 79–100

- [ ] **Step 1: Replace state variable declarations**

In `_YoloAppState`, replace these fields:

```dart
// REMOVE these lines (79-100):
File? _imageFile;
img.Image? _decodedImage;
List<Recognition> _results = [];
// ...
bool _awaitingRegionSelection = false;
final List<_EditAction> _undoStack = [];
static const int _maxUndoDepth = 20;
// ...
List<Rect> _savedRegions = [];
```

With:

```dart
// ADD these instead:
final List<PhotoSession> _photos = [];
int _currentIndex = 0;
static const int _maxUndoDepth = 20;

PhotoSession? get _current =>
    _photos.isEmpty ? null : _photos[_currentIndex];
```

Keep these unchanged (they are global/transient):
```dart
bool _isProcessing = false;
bool _modelReady = false;
bool _modelError = false;
bool _isRegionMode = false;
bool _isEditMode = false;
Rect? _draggingRegion;
Size? _currentWidgetSize;
int? _draggingCircleIndex;
Recognition? _draggingOriginalDetection;
Offset? _draggingCenterOverride;
bool _significantDrag = false;
final TransformationController _transformationController = TransformationController();
```

- [ ] **Step 2: Run analyzer — expect errors pointing to all usages of removed fields**

```
flutter analyze lib/main.dart
```

Expected: Multiple errors referencing `_imageFile`, `_results`, `_undoStack`, `_savedRegions`, `_decodedImage`, `_awaitingRegionSelection`. That's expected — tasks 3–6 fix them.

- [ ] **Step 3: Commit the structural change**

```
git add lib/main.dart
git commit -m "refactor: replace single-image state with PhotoSession list"
```

---

## Task 3: Update `_processImage` to create a new session

**Files:**
- Modify: `lib/main.dart` — `_processImage` method (previously lines 124–151)

- [ ] **Step 1: Replace `_processImage` body**

Find and replace the `setState` block inside `_processImage` (after `if (picked == null) return;`):

```dart
// REMOVE:
setState(() {
  _isProcessing = false;
  _results = [];
  _undoStack.clear();
  _imageFile = File(picked.path);
  _decodedImage = null;
  _awaitingRegionSelection = true;
  _savedRegions = [];
  _transformationController.value = Matrix4.identity();
});
```

```dart
// ADD:
final session = PhotoSession(imageFile: File(picked.path));
setState(() {
  _photos.add(session);
  _currentIndex = _photos.length - 1;
  _isEditMode = false;
  _isRegionMode = false;
  _isProcessing = false;
  _draggingRegion = null;
  _transformationController.value = Matrix4.identity();
});
```

- [ ] **Step 2: Verify no regression — run analyzer**

```
flutter analyze lib/main.dart
```

Expected: Fewer errors — `_processImage`-related errors gone, others remain.

- [ ] **Step 3: Commit**

```
git add lib/main.dart
git commit -m "feat: _processImage creates new PhotoSession instead of replacing state"
```

---

## Task 4: Update `_confirmRegionAndProcess` to use `_current`

**Files:**
- Modify: `lib/main.dart` — `_confirmRegionAndProcess` method

- [ ] **Step 1: Update the method**

Replace the entire `_confirmRegionAndProcess` method:

```dart
Future<void> _confirmRegionAndProcess() async {
  if (_current == null) return;

  setState(() {
    _current!.awaitingRegionSelection = false;
    _isRegionMode = false;
    _isProcessing = true;
    _draggingRegion = null;
  });

  try {
    final bytes = await _current!.imageFile.readAsBytes();
    final decoded = img.decodeImage(bytes);

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
```

- [ ] **Step 2: Run analyzer**

```
flutter analyze lib/main.dart
```

Expected: Errors for `_confirmRegionAndProcess` gone.

- [ ] **Step 3: Commit**

```
git add lib/main.dart
git commit -m "feat: _confirmRegionAndProcess writes results to current PhotoSession"
```

---

## Task 5: Update all remaining methods that use per-photo state

**Files:**
- Modify: `lib/main.dart` — methods `_hitTestCircle`, `_averageDetectionSize`, `_createCircleAtPosition`, `_resultsForDisplay`, `_removeBox`, `_undo`, `_getSummary`

- [ ] **Step 1: Update `_hitTestCircle`**

```dart
int? _hitTestCircle(Offset localPosition, {double extraPadding = 16.0}) {
  if (_currentWidgetSize == null || _current == null || _current!.results.isEmpty) return null;
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
```

- [ ] **Step 2: Update `_averageDetectionSize`**

```dart
Size _averageDetectionSize() {
  if (_current == null || _current!.results.isEmpty) return const Size(0.06, 0.06);
  final results = _current!.results;
  final avgW = results.map((r) => r.location.width).reduce((a, b) => a + b) / results.length;
  final avgH = results.map((r) => r.location.height).reduce((a, b) => a + b) / results.length;
  return Size(avgW, avgH);
}
```

- [ ] **Step 3: Update `_createCircleAtPosition`**

```dart
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
```

- [ ] **Step 4: Update `_resultsForDisplay`**

```dart
List<Recognition> get _resultsForDisplay {
  if (_current == null) return [];
  if (_draggingCircleIndex == null || _draggingCenterOverride == null) return _current!.results;
  final list = List<Recognition>.from(_current!.results);
  final old = list[_draggingCircleIndex!];
  list[_draggingCircleIndex!] = Recognition(
    old.classId, old.label, old.score,
    Rect.fromCenter(center: _draggingCenterOverride!, width: old.location.width, height: old.location.height),
    angle: old.angle,
  );
  return list;
}
```

- [ ] **Step 5: Update `_removeBox`**

```dart
void _removeBox(Recognition box) {
  if (_current == null) return;
  final index = _current!.results.indexOf(box);
  setState(() {
    _current!.results.remove(box);
    _current!.undoStack.add(_RemovedDetection(box, index));
    if (_current!.undoStack.length > _maxUndoDepth) _current!.undoStack.removeAt(0);
  });
}
```

- [ ] **Step 6: Update `_undo`**

```dart
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
```

- [ ] **Step 7: Update `_getSummary`**

```dart
String _getSummary() {
  if (_isProcessing) return 'Processando...';
  if (_current == null || _current!.results.isEmpty) return 'Nenhum objeto detectado.';
  final Map<String, int> counts = {};
  for (var r in _current!.results) {
    counts[r.label] = (counts[r.label] ?? 0) + 1;
  }
  return counts.entries.map((e) => '${e.value}x ${e.key}').join('  |  ');
}
```

- [ ] **Step 8: Run analyzer — expect zero errors**

```
flutter analyze lib/main.dart
```

Expected: No errors.

- [ ] **Step 9: Run tests**

```
flutter test
```

Expected: PASS.

- [ ] **Step 10: Commit**

```
git add lib/main.dart
git commit -m "refactor: update all per-photo state accesses to use _current"
```

---

## Task 6: Update `build` method — fix all references to removed fields

**Files:**
- Modify: `lib/main.dart` — `build` method

There are several places in the `build` method that reference `_imageFile`, `_results`, `_undoStack`, `_savedRegions`, `_awaitingRegionSelection`, and `_decodedImage`. Each must be updated to use `_current`.

- [ ] **Step 1: Replace `_imageFile` with `_current?.imageFile` in image area**

Find this in the `build` method:

```dart
child: _imageFile == null
    ? Center(
```

Replace with:

```dart
child: _current == null
    ? Center(
```

Find:
```dart
child: Container(
  decoration: _isEditMode
```

This is fine — no change needed here.

Find:
```dart
Image.file(
  _imageFile!,
  fit: BoxFit.fill,
),
```

Replace with:
```dart
Image.file(
  _current!.imageFile,
  fit: BoxFit.fill,
),
```

Find:
```dart
if (_decodedImage != null) {
  aspectRatio =
      _decodedImage!.width / _decodedImage!.height;
}
```

Replace with:
```dart
if (_current?.decodedImage != null) {
  aspectRatio =
      _current!.decodedImage!.width / _current!.decodedImage!.height;
}
```

- [ ] **Step 2: Replace `_savedRegions` and `_awaitingRegionSelection` in the GestureDetector and Stack**

Every reference to `_savedRegions` in `build` becomes `(_current?.savedRegions ?? [])`.
Every reference to `_awaitingRegionSelection` becomes `(_current?.awaitingRegionSelection ?? false)`.

Specifically, in the GestureDetector callbacks:

Replace all `_savedRegions` occurrences inside the `build` method with `_current!.savedRegions`.
Replace all `_awaitingRegionSelection` inside the `build` method with `(_current?.awaitingRegionSelection ?? false)`.

For the drag handlers that write to `_savedRegions`:
```dart
// FIND (in onPanEnd):
setState(() {
  _savedRegions.add(_draggingRegion!);
  _draggingRegion = null;
});
// REPLACE:
setState(() {
  _current!.savedRegions.add(_draggingRegion!);
  _draggingRegion = null;
});
```

For the X button region deletion:
```dart
// FIND:
onTap: () => setState(() {
  final region = _savedRegions.removeAt(i);
  _results.removeWhere((r) { ... });
}),
// REPLACE:
onTap: () => setState(() {
  final region = _current!.savedRegions.removeAt(i);
  _current!.results.removeWhere((r) { ... });
}),
```

- [ ] **Step 3: Replace `_results` references in detection card and undo button**

Detection card visibility check:
```dart
// FIND:
if (_imageFile != null)
  Padding(
// REPLACE:
if (_current != null)
  Padding(
```

Undo button visibility:
```dart
// FIND:
if (_isEditMode && _undoStack.isNotEmpty)
// REPLACE:
if (_isEditMode && (_current?.undoStack.isNotEmpty ?? false))
```

Detection count in card:
```dart
// FIND:
if (_results.isNotEmpty)
  Padding(
    ...
    child: Text(
      'Total: ${_results.length} objeto(s)',
// REPLACE:
if (_current != null && _current!.results.isNotEmpty)
  Padding(
    ...
    child: Text(
      'Total: ${_current!.results.length} objeto(s)',
```

Ghost toolbar visibility:
```dart
// FIND:
if (_imageFile != null && !_awaitingRegionSelection && !_isRegionMode)
// REPLACE:
if (_current != null && !(_current?.awaitingRegionSelection ?? false) && !_isRegionMode)
```

Detectar button visibility:
```dart
// FIND:
if (_imageFile != null && (_awaitingRegionSelection || _isRegionMode) && !_isProcessing)
// REPLACE:
if (_current != null && ((_current?.awaitingRegionSelection ?? false) || _isRegionMode) && !_isProcessing)
```

Detectar button label:
```dart
// FIND:
_savedRegions.isEmpty ? 'Detectar' : 'Detectar ${_savedRegions.length} Área(s)',
// REPLACE:
(_current?.savedRegions.isEmpty ?? true) ? 'Detectar' : 'Detectar ${_current!.savedRegions.length} Área(s)',
```

- [ ] **Step 4: Run analyzer and fix any remaining errors**

```
flutter analyze lib/main.dart
```

Expected: Zero errors.

- [ ] **Step 5: Run tests**

```
flutter test
```

Expected: PASS.

- [ ] **Step 6: Commit**

```
git add lib/main.dart
git commit -m "refactor: update build method to use _current for all per-photo state"
```

---

## Task 7: Add navigation method and thumbnail strip

**Files:**
- Modify: `lib/main.dart`

- [ ] **Step 1: Add `_navigateTo` method**

Insert after the `_undo` method:

```dart
void _navigateTo(int index) {
  if (index < 0 || index >= _photos.length) return;
  setState(() {
    _currentIndex = index;
    _isEditMode = false;
    _isRegionMode = false;
    _draggingRegion = null;
    _draggingCircleIndex = null;
    _draggingOriginalDetection = null;
    _draggingCenterOverride = null;
    _significantDrag = false;
    _transformationController.value = Matrix4.identity();
  });
}
```

- [ ] **Step 2: Write the test for `_navigateTo` bounds**

Add to `test/widget_test.dart` inside `main()`:

```dart
group('navigation', () {
  test('PhotoSession list index stays in bounds', () {
    final photos = [
      PhotoSession(imageFile: File('a.jpg')),
      PhotoSession(imageFile: File('b.jpg')),
    ];
    int idx = 0;

    void navigate(int to) {
      if (to < 0 || to >= photos.length) return;
      idx = to;
    }

    navigate(1);
    expect(idx, 1);
    navigate(2); // out of bounds
    expect(idx, 1); // unchanged
    navigate(-1); // out of bounds
    expect(idx, 1); // unchanged
    navigate(0);
    expect(idx, 0);
  });
});
```

- [ ] **Step 3: Run tests**

```
flutter test test/widget_test.dart
```

Expected: PASS — 2 tests.

- [ ] **Step 4: Add `_buildThumbnailStrip` widget method**

Insert after `_navigateTo`:

```dart
Widget _buildThumbnailStrip(ColorScheme colorScheme) {
  return SizedBox(
    height: 52,
    child: ListView.separated(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
      itemCount: _photos.length + 1, // +1 for the "+" button
      separatorBuilder: (_, __) => const SizedBox(width: 6),
      itemBuilder: (context, i) {
        if (i == _photos.length) {
          // "+" button
          return GestureDetector(
            onTap: () => _showImageSourceSheet(context),
            child: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: AppTheme.emerald,
                borderRadius: BorderRadius.circular(5),
              ),
              child: const Icon(Icons.add, color: Colors.white, size: 20),
            ),
          );
        }

        final isActive = i == _currentIndex;
        final photo = _photos[i];
        final count = photo.results.length;

        return GestureDetector(
          onTap: () => _navigateTo(i),
          onLongPress: () => _confirmDeletePhoto(context, i),
          child: Stack(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(5),
                  border: Border.all(
                    color: isActive ? AppTheme.emerald : colorScheme.outline,
                    width: isActive ? 2 : 1,
                  ),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: Image.file(photo.imageFile, fit: BoxFit.cover),
                ),
              ),
              if (count > 0)
                Positioned(
                  bottom: 1,
                  right: 1,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 3, vertical: 1),
                    decoration: BoxDecoration(
                      color: AppTheme.emerald,
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Text(
                      '$count',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 8,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    ),
  );
}
```

- [ ] **Step 5: Add `_showImageSourceSheet` helper method**

Insert after `_buildThumbnailStrip`:

```dart
void _showImageSourceSheet(BuildContext context) {
  showModalBottomSheet<void>(
    context: context,
    builder: (_) => SafeArea(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          ListTile(
            leading: const Icon(Icons.photo_library_rounded),
            title: const Text('Galeria'),
            onTap: () {
              Navigator.pop(context);
              _processImage(ImageSource.gallery);
            },
          ),
          ListTile(
            leading: const Icon(Icons.camera_alt_rounded),
            title: const Text('Câmera'),
            onTap: () {
              Navigator.pop(context);
              _processImage(ImageSource.camera);
            },
          ),
        ],
      ),
    ),
  );
}
```

- [ ] **Step 6: Add `_confirmDeletePhoto` helper method**

Insert after `_showImageSourceSheet`:

```dart
void _confirmDeletePhoto(BuildContext context, int index) {
  showDialog<void>(
    context: context,
    builder: (_) => AlertDialog(
      title: const Text('Remover foto?'),
      content: Text('Foto ${index + 1} e suas detecções serão removidas.'),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancelar'),
        ),
        TextButton(
          onPressed: () {
            Navigator.pop(context);
            setState(() {
              _photos.removeAt(index);
              _currentIndex = _photos.isEmpty ? 0 : _currentIndex.clamp(0, _photos.length - 1);
            });
          },
          style: TextButton.styleFrom(foregroundColor: AppTheme.errorRed),
          child: const Text('Remover'),
        ),
      ],
    ),
  );
}
```

- [ ] **Step 7: Insert thumbnail strip in `build` method**

In the `build` method's `Column` children list, insert `_buildThumbnailStrip` after the `Expanded` image area and before the detection card:

```dart
// After the closing of the Expanded(...) widget, before:
// // --- Detection card (only when image loaded) ---
// Add:
if (_photos.isNotEmpty)
  _buildThumbnailStrip(colorScheme),
```

- [ ] **Step 8: Run analyzer**

```
flutter analyze lib/main.dart
```

Expected: Zero errors.

- [ ] **Step 9: Commit**

```
git add lib/main.dart test/widget_test.dart
git commit -m "feat: add thumbnail strip with navigation, add/delete photo support"
```

---

## Task 8: Add navigation arrows to the image Stack

**Files:**
- Modify: `lib/main.dart` — the `Stack` children inside the image area

- [ ] **Step 1: Add left and right arrow overlays to the Stack**

Inside the `Stack(fit: StackFit.expand, children: [...])` in the image area, add two new `Positioned` widgets at the end of the children list (after the loading overlay):

```dart
// Left arrow
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

// Right arrow
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
```

- [ ] **Step 2: Run analyzer**

```
flutter analyze lib/main.dart
```

Expected: Zero errors.

- [ ] **Step 3: Commit**

```
git add lib/main.dart
git commit -m "feat: add left/right navigation arrows overlaid on image"
```

---

## Task 9: Add session total to detection card and "Nova sessão" button

**Files:**
- Modify: `lib/main.dart` — detection card and app bar

- [ ] **Step 1: Update detection card to show session total**

In the detection card's `Column` children, after the existing `'Total: ${_current!.results.length} objeto(s)'` text, add:

```dart
if (_photos.length > 1)
  Padding(
    padding: const EdgeInsets.only(top: 2),
    child: Text(
      'Total da sessão: ${_photos.fold(0, (sum, p) => sum + p.results.length)} objeto(s)',
      style: AppTheme.secondaryStyle.copyWith(
        color: AppTheme.emerald,
        fontWeight: FontWeight.w600,
      ),
    ),
  ),
```

- [ ] **Step 2: Add "Nova sessão" button to app bar**

In the `AppBar`'s `actions` list, insert before the theme toggle button:

```dart
if (_photos.isNotEmpty)
  TextButton(
    onPressed: () => _confirmNewSession(context),
    child: Text(
      'Nova sessão',
      style: AppTheme.buttonStyle.copyWith(color: AppTheme.emerald),
    ),
  ),
```

- [ ] **Step 3: Add `_confirmNewSession` method**

Insert after `_confirmDeletePhoto`:

```dart
void _confirmNewSession(BuildContext context) {
  showDialog<void>(
    context: context,
    builder: (_) => AlertDialog(
      title: const Text('Nova sessão?'),
      content: const Text('Todas as fotos e detecções serão removidas.'),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancelar'),
        ),
        TextButton(
          onPressed: () {
            Navigator.pop(context);
            setState(() {
              _photos.clear();
              _currentIndex = 0;
              _isEditMode = false;
              _isRegionMode = false;
              _draggingRegion = null;
              _transformationController.value = Matrix4.identity();
            });
          },
          style: TextButton.styleFrom(foregroundColor: AppTheme.errorRed),
          child: const Text('Limpar tudo'),
        ),
      ],
    ),
  );
}
```

- [ ] **Step 4: Run analyzer and tests**

```
flutter analyze lib/main.dart
flutter test
```

Expected: Zero analyzer errors, all tests pass.

- [ ] **Step 5: Commit**

```
git add lib/main.dart
git commit -m "feat: session total in detection card, Nova sessão button in app bar"
```

---

## Task 10: Manual verification

- [ ] **Step 1: Run the app**

```
flutter run
```

- [ ] **Step 2: Verify basic flow**

1. App opens → shows empty state (no image, no strip).
2. Tap "Galeria" → pick a photo → image appears, `_awaitingRegionSelection` chip shows.
3. Tap "Detectar" → detections appear, count badge shows on thumbnail.
4. Tap "Galeria" or "Câmera" again — opens source picker sheet.
5. Pick a second photo → thumbnail strip appears with 2 thumbnails → second photo is active.
6. Session total shows in detection card.

- [ ] **Step 3: Verify navigation**

1. Tap left arrow → navigates to photo 1, edit mode resets.
2. Tap thumbnail 2 → navigates to photo 2.
3. Activate edit mode on photo 2, move a box → navigate to photo 1 → navigate back to photo 2 → moved box persists.

- [ ] **Step 4: Verify deletion**

1. Long press on thumbnail 1 → dialog appears.
2. Tap "Cancelar" → photo not removed.
3. Long press on thumbnail 1 → tap "Remover" → photo removed, index adjusted.
4. Delete last remaining photo → empty state shown, no strip, no arrows.

- [ ] **Step 5: Verify "Nova sessão"**

1. With 2+ photos, tap "Nova sessão" in app bar → dialog appears.
2. Tap "Cancelar" → nothing changes.
3. Tap "Limpar tudo" → app returns to initial empty state.

- [ ] **Step 6: Final commit if all looks good**

```
git add -A
git commit -m "feat: multi-photo session complete — carousel, navigation, deletion, session total"
```
