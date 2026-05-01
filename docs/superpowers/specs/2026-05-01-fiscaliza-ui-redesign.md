# Fiscaliza — UI Redesign Spec

**Date:** 2026-05-01  
**Status:** Approved

## Context

The app "YOLOv8 Detector" is being renamed to **"Fiscaliza"** and receiving a full visual redesign. The current UI uses an ad-hoc dark navy/blue palette with hardcoded colors scattered across the codebase, no centralized theme system, a heavy overlay banner that covers the image during edit mode, low-contrast bounding boxes in light environments, and numbered circles pinned to the top edge of boxes instead of centered inside them.

The goal is a minimal, professional interface: neutral 60-30-10 palette, accent color reserved exclusively for CTAs, single sans-serif font with weight/size hierarchy, 8px-grid spacing, and consistent 8px border radius. Both Light and Dark modes are required with a toggle in the AppBar.

---

## Design Decisions (approved via visual brainstorming)

| Decision | Choice |
|---|---|
| Theme | Light + Dark with in-session toggle (sun/moon icon in AppBar) |
| Accent color — CTAs only | Emerald `#10B981` |
| Bottom action layout | FAB pills (Galeria / Câmera) + contextual ghost toolbar above |
| App name | **Fiscaliza** |
| Process button label | **Detectar** |
| Edit mode indicator | Green border on image container + subtle bottom chip — no overlay covering image |
| Box visibility in Light mode | 2px red stroke + white outline halo on stroke and numbered circles |
| Number inside box | Circle **centered inside** box, size proportional to box (min 10px, max 22px), white ring border |

---

## Design System

### Color Palette — 60-30-10

| Role | Light | Dark |
|---|---|---|
| 60% Background | `#FFFFFF` | `#0F172A` Slate 950 |
| 30% Surfaces / Cards | `#F1F5F9` Slate 100 | `#1E293B` Slate 800 |
| 10% CTA accent | `#10B981` Emerald 500 | `#10B981` Emerald 500 |
| Text primary | `#0F172A` | `#F8FAFC` |
| Text secondary | `#64748B` | `#94A3B8` |
| Text muted | `#94A3B8` | `#475569` |
| Borders / dividers | `#E2E8F0` | `#334155` |
| Active button bg | `#ECFDF5` | `rgba(16,185,129,0.12)` |
| Active button text | `#065F46` | `#10B981` |
| Active button border | `#10B981` | `#10B981` |
| Error / delete | `#EF4444` | `#EF4444` |

`#10B981` appears **only** on FAB pill buttons, active toolbar button state, and the edit-mode image border. Nothing else uses this color.

### Typography — Inter (single font)

Add `google_fonts` to `pubspec.yaml`. Use `GoogleFonts.interTextTheme()` as the base.

| Usage | Size | Weight | Extra |
|---|---|---|---|
| AppBar title | 18sp | w700 | |
| Primary value (count) | 18sp | w700 | |
| Button label | 14sp | w600 | |
| Body text | 13sp | w400 | |
| Section labels | 11sp | w600 | `letterSpacing: 1.0`, uppercase |
| Secondary / muted | 12sp | w400 | |

No italic. Hierarchy through size and weight only.

### Spacing — 8px Grid

| Token | Value | Common uses |
|---|---|---|
| sp4 | 4px | tight gaps, icon padding |
| sp8 | 8px | gap between toolbar buttons |
| sp12 | 12px | image container margin, card vertical padding |
| sp16 | 16px | screen horizontal margin, card horizontal padding |
| sp24 | 24px | section gaps |
| sp32 | 32px | large vertical rhythm |

### Border Radius

| Component | Radius |
|---|---|
| Cards, panels, image container | 8px |
| FAB pill buttons | 50px (fully rounded) |
| Ghost toolbar buttons | 8px |
| Icon-only buttons | 8px |
| Edit mode chip | 20px |

No drop shadows. Cards use a `1px` border only. Image container uses `ClipRRect(borderRadius: 8px)`.

---

## File Structure

```
lib/
  theme/
    app_theme.dart          ← NEW: ThemeData light+dark, color constants, TextStyles
  main.dart                 ← MODIFY: use AppTheme, theme toggle, rename, restructure UI
  yolo_service.dart         ← unchanged
  widgets/
    box_painter.dart        ← MODIFY: centered circle, white outline for light mode
    manual_box_editor.dart  ← unchanged
```

### `lib/theme/app_theme.dart` (new file)

Exports:
- `AppTheme.light()` → `ThemeData`
- `AppTheme.dark()` → `ThemeData`
- Fixed color constants (same in both themes): `AppTheme.emerald`, `AppTheme.errorRed`, `AppTheme.activeLightBg`, `AppTheme.activeDarkBg`, `AppTheme.activeLightText`, `AppTheme.activeDarkText`
- Brightness-dependent colors (use `Theme.of(context).colorScheme.*`): background, surface, divider, text colors — these are configured inside `ThemeData` and accessed via `colorScheme`, not via `AppTheme.*` statics
- Static `TextStyle` getters (font/size/weight only, no color): `AppTheme.titleStyle`, `AppTheme.valueStyle`, `AppTheme.labelStyle`, `AppTheme.bodyStyle`, `AppTheme.secondaryStyle`, `AppTheme.buttonStyle`; color applied at call site via `style.copyWith(color: ...)`

All widget files import `app_theme.dart` and resolve colors via `Theme.of(context).colorScheme` or `AppTheme.*` fixed constants. Zero hardcoded hex color values remain in `main.dart` or `box_painter.dart`.

---

## Screen Specifications

### AppBar (all screens)

- Title: `"Fiscaliza"`, `AppTheme.titleStyle`
- Leading: none
- `elevation: 0`, `scrolledUnderElevation: 0`
- Background: `colorScheme.surface` (60% color)
- Bottom border: `PreferredSize` with 1px `Divider` using divider color
- Actions:
  - `ThemeToggleButton`: `IconButton` with `Icons.light_mode` / `Icons.dark_mode` alternating; calls `setState` toggling `ThemeMode` on `_YoloAppState`
  - `ModelStatusIndicator`: 8px dot (`#10B981` ready, `#EF4444` error) or 12px `CircularProgressIndicator` while loading; no change to existing logic

### Screen 1 — Empty State

Body: `Expanded` → `Column(mainAxisAlignment: center)`:
- `Icon(Icons.image_search_outlined, size: 64, color: textMuted)`
- `SizedBox(height: sp16)`
- `Text("Selecione uma imagem para começar", style: secondaryStyle, textAlign: center)`

Bottom: FAB row only (no toolbar, no detection card):
- `Padding(horizontal: sp16, bottom: sp16)` → `Row(gap: sp8)` → `[GalleryFab] [CameraFab]`

### Screen 2 — Results State

**Image area:**
- `Padding(all: sp12)` → `ClipRRect(radius: 8px)` → `InteractiveViewer` → `AspectRatio` → `GestureDetector` → `Stack`
- Stack children: `Image.file`, `BoundingBoxPainter`, `ManualBoxEditorPainter`, `_RegionSelectorPainter`, loading overlay, region delete buttons (unchanged logic)
- Edit-mode image border: wrap the `ClipRRect` in a `Container` with conditional `decoration: BoxDecoration(borderRadius: 8px, border: Border.all(color: AppTheme.emerald, width: 2))` when `_isEditMode`

**Edit mode chip** (replaces old full-width overlay banner):
- `Positioned(bottom: sp8, left: 0, right: 0)` inside the Stack
- `Center` → `Container(padding: h:sp12 v:sp6, decoration: borderRadius:20px, color: black70/white90, border: 1px divider)`
- `Text("✏  TOQUE PARA EDITAR", style: 10sp w600 muted, letterSpacing: 0.5)`
- Only visible when `_isEditMode && !_isManualBoxActive`

**Detection card** (visible when `_imageFile != null`):
- `Padding(h: sp16, v: sp4)` → `Container(padding: h:sp16 v:sp12, decoration: radius:8px border:1px dividerColor bg:surfaceCard)`
- `Text("DETECÇÕES", style: labelStyle)` — uppercase, muted
- `Text("246x madeira", style: valueStyle)`
- `Text("Total: 246 objeto(s)", style: secondaryStyle)`

**Contextual toolbar** (visible when `_imageFile != null && !_isManualBoxActive && !_awaitingRegionSelection`):
- `Padding(h: sp16, bottom: sp8)` → `Row(gap: sp8)`
- `SelectAreaButton(flex:1)`: ghost style; active (emerald) when `_isRegionMode`
- `EditButton(flex:1)`: ghost style; active (emerald) when `_isEditMode`
- `UndoIconButton`: 40×40 ghost icon button `Icons.undo`; visible only when `_isEditMode && _undoStack.isNotEmpty`
- Ghost style: `OutlinedButton` with border `1.5px dividerColor`, radius `8px`, height `40px`, text color secondary
- Active style: border `AppTheme.emerald`, bg `activeBg`, text `activeText`

**Process button** (visible when `_awaitingRegionSelection || _isRegionMode`):
- Full-width FAB pill style, emerald, label `"Detectar"` (no regions) or `"Detectar X Área(s)"` (with regions), `Icons.check`
- Replaces toolbar row when visible

**FAB row** (always visible except when `_isManualBoxActive`):
- `Padding(h: sp16, bottom: sp16)` → `Row(gap: sp8)`
- Each: `Expanded` → `ElevatedButton`, height `48px`, radius `50px`, bg `AppTheme.emerald`, white text `14sp w600`, icon + label

### Screen 3 — Manual Box Active

Toolbar replaced by:
- `ConfirmButton(flex:1)`: emerald active style, `Icons.check`, label `"Confirmar"`
- `CancelButton(flex:1)`: red border (`#EF4444`) + `rgba(239,68,68,0.10)` bg, `Icons.close`, label `"Cancelar"`

FAB row hidden while `_isManualBoxActive`.

### Screen 4 — Region Selection Mode

- `SelectAreaButton` renders in active state
- Dragged region outline: `DashedBorder` or `Paint` with `color: AppTheme.emerald` (replaces current amber/yellow)
- Region delete buttons: `Icons.close`, 20×20 circle, `#EF4444` background

---

## BoundingBoxPainter Changes (`lib/widgets/box_painter.dart`)

### Signature change

```dart
// Before
BoundingBoxPainter({required this.results, required this.imageSize, ...})

// After — add isDarkMode
BoundingBoxPainter({required this.results, required this.imageSize, required this.isDarkMode, ...})
```

Caller in `main.dart` passes `isDarkMode: Theme.of(context).brightness == Brightness.dark`.

### Centered numbered circle

Current code already centers the circle (`cx, cy` = box center). Keep that. Update the sizing and rendering:

```
radius = (min(boxW, boxH) * 0.30).clamp(5.0, 11.0)   // 30% of shorter side, 5–11px
fontSize = (radius * 0.9).clamp(4.0, 9.0)
```

Dark mode rendering (inside `_drawNumberedCircle`):
1. Draw filled circle: `color = boxColor, style = fill`
2. Draw stroke ring: `color = Colors.white, width = 1.5px, style = stroke`
3. Draw number: centered `TextPainter`, white, `fontSize`

Light mode rendering (additional steps before step 1):
1. Draw white halo circle: `radius + 2px`, `color = Colors.white, style = fill`
2. Then steps 1–3 above (the colored circle on top of the white halo)

### Box stroke — light mode

Inside `_drawRegularBox` and `_drawOBB`, when `!isDarkMode`:
1. Paint white outline first: `strokeWidth = 5px, color = Colors.white.withOpacity(0.7), style = stroke` — drawn before the colored stroke
2. Then colored stroke: `strokeWidth = 2px`

Dark mode: colored stroke only, `strokeWidth = 2px` (unchanged behavior).

---

## Verification Checklist

1. `flutter run` — Light/Dark toggle in AppBar switches both themes cleanly
2. Empty state: only image placeholder + 2 FAB pills visible, no toolbar or card
3. Pick image → results appear: detection card + ghost toolbar (Selecionar Área + Editar) + FABs
4. Tap Editar → button turns emerald, image gets green border, chip "✏ TOQUE PARA EDITAR" appears at bottom of image without covering content
5. Light mode + detections: boxes have visible white outline halo, numbered circles have white halo ring
6. Numbered circles are centered inside their box, sized proportionally (larger boxes → larger circle, smaller boxes → smaller circle, min 5px radius)
7. Undo button appears only when in edit mode with undo stack non-empty; disappears when stack is empty
8. Tap "Detectar" in region mode → label shows "Detectar" or "Detectar X Área(s)" (never "Processar Imagem")
9. Manual box active: Confirm/Cancel replace toolbar; FABs hidden
10. `flutter analyze` — zero errors
11. `flutter test` — existing smoke tests pass
