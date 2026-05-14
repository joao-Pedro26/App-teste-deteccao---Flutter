# Multi-Photo Session — Design Spec

**Date:** 2026-05-13  
**Status:** Approved

## Context

Currently the app processes one image at a time. When the user picks a new image, the previous results are discarded. For field inspection use cases (e.g. counting timber pieces across a stack that requires multiple photos), the fiscal needs to accumulate detection counts across several photos, navigate back to any previous photo, and edit detections if needed. This spec covers adding that capability without changing any existing detection or editing behavior.

---

## Data Model

### New class: `PhotoSession`

Lives in `lib/main.dart` alongside the existing `_EditAction` class.

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

### State changes in `_YoloAppState`

Replace the single-image state variables with a session list:

| Remove | Replace with |
|---|---|
| `File? _imageFile` | `List<PhotoSession> _photos = []` |
| `img.Image? _decodedImage` | `int _currentIndex = 0` |
| `List<Recognition> _results` | *(moved into PhotoSession)* |
| `List<_EditAction> _undoStack` | *(moved into PhotoSession)* |
| `List<Rect> _savedRegions` | *(moved into PhotoSession)* |
| `bool _awaitingRegionSelection` | *(moved into PhotoSession)* |

**State that stays global** (not per-photo): `_isEditMode`, `_isRegionMode`, `_draggingCircleIndex`, `_draggingOriginalDetection`, `_draggingCenterOverride`, `_significantDrag`, `_draggingRegion`, `_currentWidgetSize`, `_transformationController`.

Add convenience getter:
```dart
PhotoSession? get _current => _photos.isEmpty ? null : _photos[_currentIndex];
```

All existing references to `_imageFile`, `_results`, `_undoStack`, etc. are updated to read/write through `_current`.

---

## UI Changes

### Thumbnail strip

A horizontally scrollable `SizedBox(height: 48)` row inserted **between the image viewer and the detection card**, visible only when `_photos.isNotEmpty`.

Each thumbnail:
- Size: 40×32px
- Uses `Image.file` with `fit: BoxFit.cover`, clipped to rounded rect
- Badge (bottom-right): small green pill showing detection count for that photo
- Active photo: 2px emerald border (`#10B981`), inactive: 1px `#444`
- Tap → navigate to that photo
- Long press → show delete confirmation dialog

`+` button at the end of the strip (same 40×32 size, emerald fill) → shows a bottom sheet com "Galeria" e "Câmera", igual ao comportamento dos botões existentes.

### Navigation arrows

Two `IconButton`s overlaid on the image viewer (`Stack` already exists):
- Left arrow: visible when `_currentIndex > 0`
- Right arrow: visible when `_currentIndex < _photos.length - 1`
- Positioned at vertical center, 4px from each edge
- Semi-transparent emerald circle background (`opacity: 0.85`)
- Tapping changes `_currentIndex` and calls `setState`

### Total counter

In the existing detection card (`_buildDetectionCard`), add a second line below the per-photo summary:

```
Foto atual: 5 peças
Total da sessão: 12 peças  ← new, shown only when _photos.length > 1
```

### "Nova sessão" button

Add a text button or icon in the app bar (trailing area) labeled "Nova sessão". Tapping shows a confirmation dialog ("Limpar todas as fotos e começar do zero?"). On confirm: clears `_photos`, resets `_currentIndex` to 0.

---

## Behavior

| Ação | Resultado |
|---|---|
| Carrega/tira nova foto | `PhotoSession` criada, adicionada ao final de `_photos`, `_currentIndex` aponta pra ela |
| Toca seta `›` / `‹` | `_currentIndex` incrementa/decrementa, `_isEditMode` e `_isRegionMode` resetam para `false`, `_transformationController` reseta para identidade |
| Toca thumbnail | `_currentIndex` = índice tocado, mesmos resets acima |
| Toque longo no thumbnail | `showDialog` com "Remover esta foto?" → Cancelar / Remover |
| Remove foto | `_photos.removeAt(index)`, `_currentIndex` ajustado para `min(_currentIndex, _photos.length - 1)` |
| Edita boxes (modo atual) | Opera sobre `_current.results` e `_current.undoStack`, sem mudança de comportamento |
| Region selection | Opera sobre `_current.savedRegions` e `_current.awaitingRegionSelection` |
| "Nova sessão" | `_photos = []`, `_currentIndex = 0` |

---

## Files to Modify

- `lib/main.dart` — único arquivo afetado. Todas as mudanças ficam aqui:
  - Adicionar classe `PhotoSession`
  - Substituir variáveis de estado individuais por `_photos` + `_currentIndex`
  - Adicionar getter `_current`
  - Atualizar todos os acessos a `_imageFile`, `_results`, etc.
  - Inserir thumbnail strip no build
  - Inserir setas de navegação na Stack do image viewer
  - Atualizar detection card com total da sessão
  - Adicionar botão "Nova sessão" no app bar

- `lib/widgets/box_painter.dart` — sem mudanças (já recebe `detections` como parâmetro)
- `lib/yolo_service.dart` — sem mudanças

---

## Verification

1. **Fluxo básico:** abrir app → tirar/carregar foto → detecções aparecem normalmente → tira segunda foto → thumbnail strip aparece com 2 fotos → total da sessão atualizado
2. **Navegação:** setas e thumbnails navegam corretamente → estado de edição (boxes, undo) é independente por foto
3. **Edição:** ativar edit mode em foto 1, mover um box → navegar para foto 2 → voltar para foto 1 → mudança persiste
4. **Exclusão:** toque longo em thumbnail → diálogo aparece → cancelar não remove → confirmar remove → índice ajustado corretamente
5. **Nova sessão:** botão no app bar → diálogo → confirmar → app volta ao estado inicial (sem fotos)
6. **Caso extremo:** remover a única foto → app volta ao estado vazio (sem strip, sem setas)
