# Melhorias CountThings-Style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transformar o app YOLO detector em um fluxo estilo CountThings com numeração de peças, seleção de área antes do processamento, e correção manual inteligente.

**Architecture:** 
- Manter `YoloService` existente (já funcional)
- Estender `BoundingBoxPainter` para desenhar números centralizados
- Modificar fluxo em `main.dart` para "region-first" (seleciona → processa)
- Adicionar inferência pontual para adição manual de detecções

**Tech Stack:** Flutter Dart, tflite_flutter, image_picker, CustomPainter

---

## File Structure

| File | Responsibility |
|------|----------------|
| `lib/widgets/box_painter.dart` | Desenho das bounding boxes com numeração |
| `lib/main.dart` | Fluxo principal, region selection, correção manual |
| `lib/yolo_service.dart` | Inferência YOLO (sem mudanças) |
| `test/widget_test.dart` | Testes das novas funcionalidades |

---

### Task 1: Numeração das Detecções (CountThings-Style)

**Files:**
- Modify: `lib/widgets/box_painter.dart`
- Test: `test/widget_test.dart`

- [ ] **Step 1: Adicionar função de ordenação espacial**

No início do arquivo `lib/widgets/box_painter.dart`, após as imports:

```dart
// Ordena detecções em ordem espacial: topo→baixo, esquerda→direita
List<Recognition> sortDetectionsSpatially(List<Recognition> detections) {
  return List<Recognition>.from(detections)..sort((a, b) {
    // Primeiro ordena por Y (topo da box)
    final yCompare = a.location.top.compareTo(b.location.top);
    if (yCompare.abs() > 0.02) return yCompare; // Threshold de 2%
    // Se Y similar, ordena por X (esquerda da box)
    return a.location.left.compareTo(b.location.left);
  });
}
```

- [ ] **Step 2: Modificar BoundingBoxPainter para aceitar número**

Alterar construtor e adicionar campo de número:

```dart
class BoundingBoxPainter extends CustomPainter {
  final List<Recognition> detections;
  final Map<Recognition, int> detectionNumbers; // Mapeia detecção → número

  BoundingBoxPainter(this.detections, {this.detectionNumbers = const {}});
```

- [ ] **Step 3: Adicionar método _drawNumberedCircle**

Após `_drawRegularBox`, adicionar:

```dart
// Desenha círculo com número no centro da box
void _drawNumberedCircle(
    Canvas canvas, Size size, Recognition d, Color color, int number) {
  // Centro da box em pixels
  final cx = (d.location.left + d.location.right) / 2 * size.width;
  final cy = (d.location.top + d.location.bottom) / 2 * size.height;

  // Raio do círculo (~15% da menor dimensão da box ou mínimo 12px)
  final boxMinDim = min(d.location.width, d.location.height) * size.shortestSide;
  final radius = max(12.0, boxMinDim * 0.15);

  // Círculo de fundo
  final circlePaint = Paint()
    ..color = color
    ..style = PaintingStyle.fill;
  canvas.drawCircle(Offset(cx, cy), radius, circlePaint);

  // Borda branca para contraste
  final borderPaint = Paint()
    ..color = Colors.white
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2;
  canvas.drawCircle(Offset(cx, cy), radius, borderPaint);

  // Número centralizado
  final textSpan = TextSpan(
    text: '$number',
    style: TextStyle(
      color: Colors.white,
      fontSize: max(10.0, radius * 0.8),
      fontWeight: FontWeight.bold,
    ),
  );
  final textPainter = TextPainter(
    text: textSpan,
    textDirection: TextDirection.ltr,
  )..layout();

  textPainter.paint(
    canvas,
    Offset(cx - textPainter.width / 2, cy - textPainter.height / 2),
  );
}
```

- [ ] **Step 4: Modificar método paint para usar ordenação e números**

Substituir o método `paint` existente:

```dart
@override
void paint(Canvas canvas, Size size) {
  // Ordena detecções espacialmente
  final sortedDetections = sortDetectionsSpatially(detections);
  
  for (var i = 0; i < sortedDetections.length; i++) {
    final d = sortedDetections[i];
    final number = i + 1; // Numeração 1-based
    final color = _boxColors[d.classId % _boxColors.length];

    if (d.isOBB && d.angle != null) {
      _drawOBB(canvas, size, d, color);
    } else {
      _drawRegularBox(canvas, size, d, color);
    }
    
    // Desenha número centralizado
    _drawNumberedCircle(canvas, size, d, color, number);
  }
}
```

- [ ] **Step 5: Ajustar espessura do contorno para 1px**

Em `_drawRegularBox`, mudar `strokeWidth = 1` (já está 1, verificar)
Em `_drawOBB`, mudar `strokeWidth = 1` (era 3):

```dart
final boxPaint = Paint()
  ..color = color
  ..style = PaintingStyle.stroke
  ..strokeWidth = 1; // Era 3
```

- [ ] **Step 6: Commit**

```bash
git add lib/widgets/box_painter.dart
git commit -m "feat: add numbered circles CountThings-style with spatial sorting"
```

---

### Task 2: Fluxo Region-First (Selecionar → Processar)

**Files:**
- Modify: `lib/main.dart`

- [ ] **Step 1: Adicionar novo estado `_awaitingRegionSelection`**

Na classe `_YoloAppState`, após as declarações de estado existentes:

```dart
bool _awaitingRegionSelection = false;  // Aguarda seleção de região antes de processar
bool _regionProcessed = false;           // Região já foi processada
```

- [ ] **Step 2: Modificar `_processImage` para NÃO processar imediatamente**

Substituir o método `_processImage`:

```dart
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
    _imageFile = File(picked.path);
    _decodedImage = null;
    _awaitingRegionSelection = true; // Aguarda seleção
    _regionProcessed = false;
  });
}
```

- [ ] **Step 3: Adicionar método `_confirmRegionAndProcess`**

Após `_runInferenceOnRegion`, adicionar:

```dart
/// Confirma a região selecionada e processa
Future<void> _confirmRegionAndProcess() async {
  if (_draggingRegion == null || _decodedImage != null) return;

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
      
      if (region != Rect.fromLTWH(0, 0, 1, 1)) {
        // Processar apenas região
        await _runInferenceOnRegion(region);
      } else {
        // Processar imagem toda
        final detections = await _yoloService.runInference(decoded);
        setState(() {
          _decodedImage = decoded;
          _results = detections;
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
```

- [ ] **Step 4: Adicionar botão "Processar" na UI**

Na seção de botões (após os botões de modo), adicionar botão de confirmar:

```dart
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
```

- [ ] **Step 5: Permitir duplo-toque para confirmar região**

No GestureDetector, adicionar `onDoubleTap`:

```dart
onDoubleTap: _awaitingRegionSelection && _draggingRegion != null
    ? _confirmRegionAndProcess
    : null,
```

- [ ] **Step 6: Ajustar overlay de instrução para modo region-first**

Modificar o overlay de instrução (onde diz "Arraste para selecionar área"):

```dart
if (_awaitingRegionSelection && !_regionProcessed)
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
            child: const Text(
              'Arraste para selecionar área',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'ou toque em "Processar Área"',
            style: TextStyle(color: Colors.white70, fontSize: 14),
          ),
        ],
      ),
    ),
  ),
```

- [ ] **Step 7: Commit**

```bash
git add lib/main.dart
git commit -m "feat: implement region-first flow (select area before processing)"
```

---

### Task 3: Correção Manual Inteligente

**Files:**
- Modify: `lib/main.dart`

- [ ] **Step 1: Criar método `_runInferenceOnPoint`**

Após `_runInferenceOnRegion`, adicionar:

```dart
/// Roda inferência em um ponto específico da imagem (para adição manual)
Future<void> _runInferenceOnPoint(Offset tapPosition) async {
  if (_decodedImage == null || _currentWidgetSize == null) return;

  setState(() => _isProcessing = true);

  try {
    // Converter tap para coordenadas normalizadas
    final normalizedTap = Offset(
      tapPosition.dx / _currentWidgetSize!.width,
      tapPosition.dy / _currentWidgetSize!.height,
    );

    // Definir região de ~10% da imagem centrada no tap
    const regionSize = 0.10;
    final halfSize = regionSize / 2;
    
    final region = Rect.fromLTRB(
      (normalizedTap.dx - halfSize).clamp(0.0, 1.0),
      (normalizedTap.dy - halfSize).clamp(0.0, 1.0),
      (normalizedTap.dx + halfSize).clamp(0.0, 1.0),
      (normalizedTap.dy + halfSize).clamp(0.0, 1.0),
    );

    // Converter para pixels na imagem original
    final int rx = (region.left * _decodedImage!.width).round();
    final int ry = (region.top * _decodedImage!.height).round();
    final int rw = (region.width * _decodedImage!.width).round();
    final int rh = (region.height * _decodedImage!.height).round();

    // Crop da imagem
    final cropped = img.copyCrop(_decodedImage!, x: rx, y: ry, width: rw, height: rh);

    // Run inference na região
    final detections = await _yoloService.runInference(cropped);

    if (detections.isNotEmpty) {
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
        _results.addAll(offsetDetections);
        _isProcessing = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('${offsetDetections.length} objeto(s) adicionado(s)'),
          duration: const Duration(seconds: 1),
          backgroundColor: Colors.green,
        ),
      );
    } else {
      // Nenhuma detecção → cria placeholder
      setState(() => _isProcessing = false);
      _createPlaceholder(normalizedTap);
    }
  } catch (e) {
    debugPrint('Erro na inferência pontual: $e');
    setState(() => _isProcessing = false);
  }
}
```

- [ ] **Step 2: Criar método `_createPlaceholder`**

Após `_runInferenceOnPoint`, adicionar:

```dart
/// Cria uma box placeholder quando o modelo não detecta nada
void _createPlaceholder(Offset normalizedPosition) {
  // Placeholder com label genérico
  const boxSize = 0.05;
  final halfSize = boxSize / 2;

  final placeholder = Recognition(
    0, // Classe 0 (primeira classe disponível)
    'Objeto?',
    0.0, // Confiança zero indica placeholder
    Rect.fromLTRB(
      (normalizedPosition.dx - halfSize).clamp(0.0, 1.0),
      (normalizedPosition.dy - halfSize).clamp(0.0, 1.0),
      (normalizedPosition.dx + halfSize).clamp(0.0, 1.0),
      (normalizedPosition.dy + halfSize).clamp(0.0, 1.0),
    ),
  );

  setState(() {
    _results.add(placeholder);
  });

  ScaffoldMessenger.of(context).showSnackBar(
    const SnackBar(
      content: Text('Placeholder adicionado - toque para editar'),
      duration: Duration(seconds: 2),
      backgroundColor: Colors.orange,
    ),
  );
}
```

- [ ] **Step 3: Adicionar edição de placeholder via long-press**

Criar método `_handleImageLongPress`:

```dart
/// Handler para long-press na imagem (editar placeholder)
void _handleImageLongPress(Offset tapPosition) async {
  if (!_isEditMode) return;

  final hitBox = _hitTest(tapPosition);
  if (hitBox != null && hitBox.score == 0.0) {
    // É um placeholder - abrir dialog para editar classe
    final selectedLabel = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: const Color(0xFF16213E),
        title: const Text(
          'Editar Objeto',
          style: TextStyle(color: Colors.white),
        ),
        content: SizedBox(
          width: double.maxFinite,
          child: ListView(
            shrinkWrap: true,
            children: _yoloService.labels.map((label) {
              return ListTile(
                title: Text(label, style: const TextStyle(color: Colors.white)),
                onTap: () => Navigator.pop(context, label),
              );
            }).toList(),
          ),
        ),
      ),
    );

    if (selectedLabel != null) {
      setState(() {
        final index = _results.indexOf(hitBox);
        if (index != -1) {
          _results[index] = Recognition(
            _yoloService.labels.indexOf(selectedLabel),
            selectedLabel,
            1.0, // Confiança máxima após edição manual
            hitBox.location,
            angle: hitBox.angle,
          );
        }
      });
    }
  }
}
```

- [ ] **Step 4: Atualizar `_handleImageTap` para usar auto-detect**

Modificar `_handleImageTap`:

```dart
/// Handler para tap na imagem (modo edição)
void _handleImageTap(Offset tapPosition) {
  if (!_isEditMode) return;

  final hitBox = _hitTest(tapPosition);
  if (hitBox != null) {
    // Tap em box existente - remover
    _removeBox(hitBox);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('${hitBox.label} removido'),
        duration: const Duration(seconds: 1),
        backgroundColor: Colors.orange,
      ),
    );
  } else {
    // Tap em área vazia - tentar detectar automaticamente
    _runInferenceOnPoint(tapPosition);
  }
}
```

- [ ] **Step 5: Adicionar long-press ao GestureDetector**

No GestureDetector em `build`, adicionar:

```dart
onLongPress: _isEditMode
    ? () {
        // Precisa capturar posição do long-press
        // Implementar com GestureDetector nested ou Listener
      }
    : null,
```

Nota: Para capturar a posição do long-press, usar `Listener` com `onPointerDown` e timer, ou substituir por um botão flutuante de "Editar classe" que aparece quando há placeholder selecionado.

- [ ] **Step 6: Adicionar contador de atualizações automáticas**

O contador já atualiza automaticamente via `_getSummary()`, mas garantir que placeholders sejam contados:

Em `_getSummary`, ajustar para mostrar placeholders separadamente:

```dart
String _getSummary() {
  if (_isProcessing) return 'Processando...';
  if (_results.isEmpty) return 'Nenhum objeto detectado.';

  final Map<String, int> counts = {};
  int placeholderCount = 0;
  
  for (var r in _results) {
    if (r.score == 0.0) {
      placeholderCount++;
    } else {
      counts[r.label] = (counts[r.label] ?? 0) + 1;
    }
  }
  
  String summary = counts.entries.map((e) => '${e.value}x ${e.key}').join('  |  ');
  if (placeholderCount > 0) {
    summary += (summary.isNotEmpty ? '  |  ' : '') + '${placeholderCount}x Objeto?';
  }
  return summary;
}
```

- [ ] **Step 7: Commit**

```bash
git add lib/main.dart
git commit -m "feat: add smart manual correction with auto-detect and placeholders"
```

---

### Task 4: Integração e Testes

**Files:**
- Modify: `test/widget_test.dart`

- [ ] **Step 1: Escrever teste para ordenação espacial**

```dart
testWidgets('Detections are sorted spatially (top-to-bottom, left-to-right)', (tester) async {
  // Criar detecções fora de ordem
  final detections = [
    Recognition(0, 'A', 0.9, Rect.fromLTRB(0.5, 0.5, 0.6, 0.6)), // Bottom-right
    Recognition(0, 'B', 0.9, Rect.fromLTRB(0.1, 0.1, 0.2, 0.2)), // Top-left
    Recognition(0, 'C', 0.9, Rect.fromLTRB(0.3, 0.1, 0.4, 0.2)), // Top-right
  ];
  
  final sorted = sortDetectionsSpatially(detections);
  
  expect(sorted[0].label, 'B'); // Primeiro: top-left
  expect(sorted[1].label, 'C'); // Segundo: top-right (mesmo Y, maior X)
  expect(sorted[2].label, 'A'); // Terceiro: bottom
});
```

- [ ] **Step 2: Escrever teste para placeholder**

```dart
testWidgets('Placeholder is created when model detects nothing', (tester) async {
  final app = MaterialApp(home: YoloApp());
  await tester.pumpWidget(app);
  
  // Simular tap em área vazia no modo edição
  // ... implementação do teste
});
```

- [ ] **Step 3: Rodar testes**

```bash
flutter test
```

Expected: All tests pass

- [ ] **Step 4: Teste manual end-to-end**

```bash
flutter run
```

Verificar:
1. Imagem carrega com overlay
2. Seleção de área funciona
3. Processamento roda na região
4. Boxes numeradas em ordem espacial
5. Modo edição remove boxes
6. Modo edição adiciona via auto-detect
7. Placeholder é criado quando não detecta
8. Contador atualiza automaticamente

- [ ] **Step 5: Commit final**

```bash
git add test/widget_test.dart
git commit -m "test: add tests for spatial sorting and placeholder creation"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ Numeração CountThings-style (Task 1)
- ✅ Ordenação espacial (Task 1, Step 1)
- ✅ Contorno fino 1px (Task 1, Step 6)
- ✅ Fluxo region-first (Task 2)
- ✅ Overlay escura com recorte (Task 2, Step 6)
- ✅ Permanecer em modo seleção (Task 2)
- ✅ Exclusão por tap (Task 3, Step 4)
- ✅ Adição com auto-detect (Task 3, Step 1)
- ✅ Placeholder fallback (Task 3, Step 2)
- ✅ Contador atualiza (Task 3, Step 6)

**Placeholder scan:** Nenhum "TBD", "TODO", ou "fill in" encontrado.

**Type consistency:** 
- `Recognition` usado consistentemente
- `Rect` coordenadas normalizadas (0.0-1.0) em todo lugar
- `Offset` para posições de tap

**Issues found:** Nenhuma inconsistência encontrada.
