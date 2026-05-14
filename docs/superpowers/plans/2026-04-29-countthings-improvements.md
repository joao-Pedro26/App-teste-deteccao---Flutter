# CountThings-style Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add re-run inference on tap, resizable manual bounding box editor, pinch-to-zoom on all image screens, and an undo stack to the Flutter YOLO detection app.

**Architecture:** All changes live in `lib/main.dart` and a new `lib/widgets/manual_box_editor.dart`. The existing `YoloService._iou()` is made public-static so `main.dart` can deduplicate re-run results against existing detections. `InteractiveViewer` wraps the image area for zoom; Flutter's hit-testing inverse-transforms tap positions automatically, so no manual coordinate math is needed for zoom.

**Tech Stack:** Flutter, Dart, `tflite_flutter`, `image` package, `InteractiveViewer` (Flutter built-in)

---

## File Map

| File | Change |
|---|---|
| `lib/yolo_service.dart` | Make `_iou` a public static method |
| `lib/main.dart` | Undo stack, re-run on tap, InteractiveViewer, manual box state + wiring |
| `lib/widgets/manual_box_editor.dart` | NEW — `ManualBoxEditorPainter` + handle hit logic |
| `test/widget_test.dart` | Extend smoke test to cover new UI elements |

---

## Task 1: Make `iou` public-static in `YoloService`

**Files:**
- Modify: `lib/yolo_service.dart:306-315`

- [ ] **Step 1: Change `_iou` to `static iou`**

In `lib/yolo_service.dart`, replace lines 306–315:

```dart
// BEFORE
double _iou(Rect a, Rect b) {

// AFTER
static double iou(Rect a, Rect b) {
```

- [ ] **Step 2: Fix the call site inside `_applyNMS`**

In `_applyNMS` (line ~297), the call `_iou(boxes[i].location, boxes[j].location)` must become `YoloService.iou(...)` because it is now static:

```dart
// BEFORE (lib/yolo_service.dart ~297)
if (!suppressed[j] &&
    _iou(boxes[i].location, boxes[j].location) > nmsThreshold) {

// AFTER
if (!suppressed[j] &&
    YoloService.iou(boxes[i].location, boxes[j].location) > nmsThreshold) {
```

- [ ] **Step 3: Verify with `flutter analyze`**

```bash
flutter analyze lib/yolo_service.dart
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add lib/yolo_service.dart
git commit -m "refactor: expose iou as public static for external deduplication"
```

---

## Task 2: Add undo stack infrastructure

**Files:**
- Modify: `lib/main.dart`

- [ ] **Step 1: Add `_EditAction` sealed classes before `YoloApp`**

Insert after the imports and before `void main()`:

```dart
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
```

- [ ] **Step 2: Add undo state fields to `_YoloAppState`**

In `_YoloAppState`, after `bool _regionProcessed = false;` (line ~35), add:

```dart
final List<_EditAction> _undoStack = [];
static const int _maxUndoDepth = 20;
```

- [ ] **Step 3: Add `_undo()` method**

After `_removeBox()` (around line 340), add:

```dart
void _undo() {
  if (_undoStack.isEmpty) return;
  final action = _undoStack.removeLast();
  setState(() {
    switch (action) {
      case _RemovedDetection(:final removed, :final originalIndex):
        final idx = originalIndex.clamp(0, _results.length);
        _results.insert(idx, removed);
      case _AddedDetections(:final added):
        _results.removeWhere((r) => added.contains(r));
    }
  });
}
```

- [ ] **Step 4: Push to undo stack in `_removeBox()`**

Replace the existing `_removeBox` (lines 336–340):

```dart
void _removeBox(Recognition box) {
  final index = _results.indexOf(box);
  setState(() {
    _results.remove(box);
    _undoStack.add(_RemovedDetection(box, index));
    if (_undoStack.length > _maxUndoDepth) _undoStack.removeAt(0);
  });
}
```

- [ ] **Step 5: Add undo button to edit-mode UI**

In the `build()` method, find the `_ToggleActionButton` for edit mode (~line 670). Add an undo button right after the edit toggle row. The existing mode-buttons `Row` is inside `if (_imageFile != null && !_awaitingRegionSelection)`. Add the undo button as a third child of that `Row`:

```dart
// Inside the Row of mode buttons, add after the Edit _ToggleActionButton:
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
```

- [ ] **Step 6: Verify and commit**

```bash
flutter analyze lib/main.dart
git add lib/main.dart
git commit -m "feat: add undo stack for edit actions (delete/add)"
```

---

## Task 3: Add `InteractiveViewer` for pinch-to-zoom

**Files:**
- Modify: `lib/main.dart`

- [ ] **Step 1: Add `TransformationController` field**

In `_YoloAppState`, after `_undoStack`:

```dart
final TransformationController _transformationController = TransformationController();
```

- [ ] **Step 2: Dispose the controller**

Override `dispose()` after `initState()`:

```dart
@override
void dispose() {
  _transformationController.dispose();
  super.dispose();
}
```

- [ ] **Step 3: Reset controller when a new image is picked**

In `_processImage()`, inside `setState(...)` after `_imageFile = File(picked.path)`:

```dart
_transformationController.value = Matrix4.identity();
```

- [ ] **Step 4: Wrap `AspectRatio` with `InteractiveViewer` inside `LayoutBuilder`**

In `build()`, inside the `LayoutBuilder` builder function (~line 430), the current structure is `return AspectRatio(...)`. Wrap it:

```dart
return InteractiveViewer(
  transformationController: _transformationController,
  panEnabled: !_isRegionMode,
  scaleEnabled: true,
  minScale: 1.0,
  maxScale: 6.0,
  boundaryMargin: EdgeInsets.zero,
  child: AspectRatio(
    aspectRatio: aspectRatio,
    child: GestureDetector(
      // ... existing gesture detector content unchanged ...
    ),
  ),
);
```

> **Note:** `panEnabled: !_isRegionMode` disables InteractiveViewer's pan only when the user is drawing a region (so the drag gesture goes to the region selector instead). In edit mode, pan remains enabled so the user can zoom/navigate while editing. `_isManualBoxActive` will be added in Task 6 — anticipate setting `panEnabled: !_isRegionMode && !_isManualBoxActive` there.

- [ ] **Step 5: Run on device and verify zoom works**

```bash
flutter run
```

- Pick an image, run inference, then pinch to zoom in/out. Bounding box circles should stay aligned with the objects.
- Tap "Selecionar Área", draw a region — zoom/pan should be disabled while drawing (no accidental pan).

- [ ] **Step 6: Commit**

```bash
git add lib/main.dart
git commit -m "feat: add pinch-to-zoom via InteractiveViewer on image screen"
```

---

## Task 4: Replace empty-tap with re-run inference

**Files:**
- Modify: `lib/main.dart`

- [ ] **Step 1: Add `_runInferenceOnTap()` method**

Add after `_runInferenceOnRegion()` (~line 137):

```dart
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

    // Discard detections that overlap significantly with existing ones
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
```

- [ ] **Step 2: Add `_openManualBoxEditor()` stub**

Add right after `_runInferenceOnTap()`:

```dart
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
```

> **Note:** Declare these fields here alongside `_openManualBoxEditor` to keep all manual-box state together:

```dart
// Add to _YoloAppState fields (after _undoStack):
Rect? _manualBoxRect;
bool _isManualBoxActive = false;
int _activeHandleIndex = -1;
```

- [ ] **Step 3: Replace the empty-tap branch in `_handleImageTap()`**

Replace the `else` branch of `_handleImageTap()` (currently calls `_addManualBox`):

```dart
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
```

- [ ] **Step 4: Delete `_addManualBox()` entirely**

Remove the entire `_addManualBox` method (lines 279–333). It is replaced by `_runInferenceOnTap` + `_openManualBoxEditor`.

- [ ] **Step 5: Verify and commit**

```bash
flutter analyze lib/main.dart
git add lib/main.dart
git commit -m "feat: replace manual-add dialog with re-run inference on empty tap"
```

---

## Task 5: Create `ManualBoxEditorPainter`

**Files:**
- Create: `lib/widgets/manual_box_editor.dart`

- [ ] **Step 1: Create the file**

```dart
import 'package:flutter/material.dart';

class ManualBoxEditorPainter extends CustomPainter {
  final Rect normalizedRect;

  static const double handleRadius = 12.0;
  static const double handleHitRadius = 24.0;

  ManualBoxEditorPainter(this.normalizedRect);

  @override
  void paint(Canvas canvas, Size size) {
    final pixelRect = Rect.fromLTRB(
      normalizedRect.left * size.width,
      normalizedRect.top * size.height,
      normalizedRect.right * size.width,
      normalizedRect.bottom * size.height,
    );

    // Semi-transparent fill
    canvas.drawRect(
      pixelRect,
      Paint()
        ..color = Colors.blue.withAlpha(40)
        ..style = PaintingStyle.fill,
    );

    // Border
    canvas.drawRect(
      pixelRect,
      Paint()
        ..color = Colors.blueAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // Corner handles
    for (final corner in _corners(pixelRect)) {
      canvas.drawCircle(
        corner,
        handleRadius,
        Paint()..color = Colors.blueAccent..style = PaintingStyle.fill,
      );
      canvas.drawCircle(
        corner,
        handleRadius,
        Paint()
          ..color = Colors.white
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }
  }

  List<Offset> _corners(Rect r) => [
        r.topLeft,
        r.topRight,
        r.bottomLeft,
        r.bottomRight,
      ];

  // Returns corner index (0=TL, 1=TR, 2=BL, 3=BR) within handleHitRadius, or -1
  static int getHandleIndex(
    Offset tapPixel,
    Rect normalizedRect,
    Size widgetSize,
  ) {
    final pixelRect = Rect.fromLTRB(
      normalizedRect.left * widgetSize.width,
      normalizedRect.top * widgetSize.height,
      normalizedRect.right * widgetSize.width,
      normalizedRect.bottom * widgetSize.height,
    );
    final corners = [
      pixelRect.topLeft,
      pixelRect.topRight,
      pixelRect.bottomLeft,
      pixelRect.bottomRight,
    ];
    for (int i = 0; i < corners.length; i++) {
      if ((corners[i] - tapPixel).distance <= handleHitRadius) return i;
    }
    return -1;
  }

  @override
  bool shouldRepaint(covariant ManualBoxEditorPainter old) =>
      old.normalizedRect != normalizedRect;
}
```

- [ ] **Step 2: Verify**

```bash
flutter analyze lib/widgets/manual_box_editor.dart
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add lib/widgets/manual_box_editor.dart
git commit -m "feat: add ManualBoxEditorPainter with 4-corner drag handles"
```

---

## Task 6: Wire manual box editor into main.dart

**Files:**
- Modify: `lib/main.dart`

- [ ] **Step 1: Add import**

At top of `lib/main.dart`, add:

```dart
import 'widgets/manual_box_editor.dart';
```

- [ ] **Step 2: Add `_updateManualBoxCorner()` method**

Add after `_openManualBoxEditor()`:

```dart
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
```

- [ ] **Step 3: Add `_confirmManualBox()` and `_cancelManualBox()` methods**

```dart
void _confirmManualBox() {
  if (_manualBoxRect == null) return;
  final label = _yoloService.labels.isNotEmpty ? _yoloService.labels.first : 'objeto';
  final classId = 0;
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
```

- [ ] **Step 4: Extend pan gesture handlers to support handle dragging**

In `build()`, the `GestureDetector` currently has `onPanStart/Update/End` wired only to `_isRegionMode`. Extend them to also handle manual box editing. Replace:

```dart
onPanStart: _isRegionMode
    ? (details) { ... region drawing ... }
    : null,
onPanUpdate: _isRegionMode
    ? (details) { ... region update ... }
    : null,
onPanEnd: _isRegionMode
    ? (details) { ... region end ... }
    : null,
```

With:

```dart
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
```

- [ ] **Step 5: Add `ManualBoxEditorPainter` to the Stack**

In the `Stack`'s `children` list (inside `child: Stack(fit: StackFit.expand, children: [...])`), add after the `BoundingBoxPainter`:

```dart
if (_isManualBoxActive && _manualBoxRect != null)
  CustomPaint(
    painter: ManualBoxEditorPainter(_manualBoxRect!),
  ),
```

- [ ] **Step 6: Add Confirm/Cancel buttons**

After the existing edit-mode instruction overlay in the Stack, add:

```dart
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
```

- [ ] **Step 7: Update `InteractiveViewer` `panEnabled` to disable during manual box editing**

Find the `InteractiveViewer` from Task 3 and update `panEnabled`:

```dart
panEnabled: !_isRegionMode && !_isManualBoxActive,
```

- [ ] **Step 8: Update edit mode instruction overlay text**

Find the text `'Toque para adicionar/remover'` and change it to reflect the new behavior:

```dart
// BEFORE
'Toque para adicionar/remover'

// AFTER
'Toque na box para remover • Toque vazio para re-detectar'
```

- [ ] **Step 9: Verify end-to-end**

```bash
flutter analyze lib/
flutter run
```

Test the full flow:
1. Pick image → run inference → numbered circles appear
2. Pinch zoom → boxes stay aligned
3. Edit mode → tap existing box → disappears → undo button appears → tap undo → box returns
4. Edit mode → tap empty area where there's a missed object → model re-runs → auto-detected and numbered
5. Edit mode → tap empty area where nothing exists → manual box editor appears → drag corners to resize → Confirmar → box numbered and added → undo removes it
6. Edit mode → Cancelar on manual box → nothing added

- [ ] **Step 10: Commit**

```bash
git add lib/main.dart lib/widgets/manual_box_editor.dart
git commit -m "feat: wire manual box editor with drag handles and confirm/cancel"
```

---

## Task 7: Update widget test

**Files:**
- Modify: `test/widget_test.dart`

- [ ] **Step 1: Extend the smoke test**

Replace `test/widget_test.dart` content:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tryingg_flutter/main.dart';

void main() {
  testWidgets('App loads with initial empty state', (WidgetTester tester) async {
    await tester.pumpWidget(const MaterialApp(home: YoloApp()));
    await tester.pump();

    expect(find.byIcon(Icons.image_search), findsOneWidget);
    expect(find.text('Selecione uma imagem para começar'), findsOneWidget);
  });

  testWidgets('Undo button not visible before edit mode', (WidgetTester tester) async {
    await tester.pumpWidget(const MaterialApp(home: YoloApp()));
    await tester.pump();

    // No undo button should be present when no image is loaded
    expect(find.byIcon(Icons.undo), findsNothing);
  });
}
```

- [ ] **Step 2: Run tests**

```bash
flutter test
```
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add test/widget_test.dart
git commit -m "test: extend smoke test to cover undo button visibility"
```

---

## Verification Checklist

- [ ] `flutter analyze` — zero errors
- [ ] `flutter test` — all tests pass
- [ ] Pick image → inference → numbered circles with thin boxes visible
- [ ] Pinch to zoom in/out — boxes stay aligned with objects
- [ ] Region select → draw rectangle → inference runs on region
- [ ] Edit mode → tap existing box → removed → undo button lights up → undo restores it
- [ ] Edit mode → tap empty area with object → auto-detected and added → undo removes it
- [ ] Edit mode → tap truly empty area → manual box editor appears with 4 corner handles
- [ ] Drag corner handles → box resizes with minimum size constraint
- [ ] Confirm → box added with auto-label → undo removes it
- [ ] Cancel → no box added
- [ ] Zoom during edit mode works (pan/tap both function correctly)
