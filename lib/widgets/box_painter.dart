import 'dart:math';
import 'package:flutter/material.dart';
import '../yolo_service.dart';

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

// Paleta de cores por classe (cicla automaticamente)
const List<Color> _boxColors = [
  Color(0xFFE53935),
  Color(0xFF1E88E5),
  Color(0xFF43A047),
  Color(0xFFFB8C00),
  Color(0xFF8E24AA),
  Color(0xFF00ACC1),
  Color(0xFFFFB300),
  Color(0xFF6D4C41),
  Color(0xFF00897B),
  Color(0xFFD81B60),
];

class BoundingBoxPainter extends CustomPainter {
  final List<Recognition> detections;

  BoundingBoxPainter(this.detections);

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

  // ── Box reto (modelos regulares) ─────────────────────────────────────────
  void _drawRegularBox(
      Canvas canvas, Size size, Recognition d, Color color) {
    final boxPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1;

    final rect = Rect.fromLTRB(
      d.location.left   * size.width,
      d.location.top    * size.height,
      d.location.right  * size.width,
      d.location.bottom * size.height,
    );

    canvas.drawRect(rect, boxPaint);
    // _drawLabel(canvas, rect.left, rect.top, d, color);
  }

  // ── Box rotacionado (modelos OBB) ─────────────────────────────────────────
  // O ângulo θ do YOLOv8-OBB é definido em relação ao eixo X,
  // positivo no sentido horário, em radianos.
  void _drawOBB(Canvas canvas, Size size, Recognition d, Color color) {
    final boxPaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1; // Contorno fino

    // Centro do box em pixels
    final cx = (d.location.left + d.location.right)  / 2 * size.width;
    final cy = (d.location.top  + d.location.bottom) / 2 * size.height;
    final w  = d.location.width  * size.width;
    final h  = d.location.height * size.height;

    final theta = d.angle!; // radianos

    // Calcula os 4 vértices do retângulo rotacionado
    // a partir do centro + dimensões + ângulo
    final cosA = cos(theta);
    final sinA = sin(theta);

    // Half-sizes
    final hw = w / 2;
    final hh = h / 2;

    // Cantos em coordenadas locais (antes da rotação)
    final corners = [
      Offset(-hw, -hh),
      Offset( hw, -hh),
      Offset( hw,  hh),
      Offset(-hw,  hh),
    ];

    // Rotaciona e translada cada canto
    final rotated = corners.map((c) => Offset(
      cx + c.dx * cosA - c.dy * sinA,
      cy + c.dx * sinA + c.dy * cosA,
    )).toList();

    // Desenha o polígono com os 4 vértices
    final path = Path()
      ..moveTo(rotated[0].dx, rotated[0].dy)
      ..lineTo(rotated[1].dx, rotated[1].dy)
      ..lineTo(rotated[2].dx, rotated[2].dy)
      ..lineTo(rotated[3].dx, rotated[3].dy)
      ..close();

    canvas.drawPath(path, boxPaint);

    // Linha indicando a direção/orientação (topo do box)
    final directionPaint = Paint()
      ..color = color
      ..strokeWidth = 3.0
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(rotated[0], rotated[1], directionPaint);

    // Label no canto superior-esquerdo rotacionado
    // _drawLabel(canvas, rotated[0].dx, rotated[0].dy, d, color);
  }

  // ── Label com fundo colorido ──────────────────────────────────────────────
  // void _drawLabel(
  //     Canvas canvas, double x, double y, Recognition d, Color color) {
  //   final angleStr = d.angle != null
  //       ? ' ${(d.angle! * 180 / pi).toStringAsFixed(0)}°'
  //       : '';
  //   final labelText =
  //       '${d.label} ${(d.score * 100).toStringAsFixed(0)}%$angleStr';

  //   final textPainter = TextPainter(
  //     text: TextSpan(
  //       text: labelText,
  //       style: const TextStyle(
  //         color: Colors.white,
  //         fontSize: 12,
  //         fontWeight: FontWeight.bold,
  //       ),
  //     ),
  //     textDirection: TextDirection.ltr,
  //   )..layout();

  //   final labelW = textPainter.width + 8;
  //   final labelH = textPainter.height + 4;

  //   // Posiciona acima do ponto, garantindo que não saia da tela
  //   double labelTop = y - labelH;
  //   if (labelTop < 0) labelTop = y;

  //   final labelRect = Rect.fromLTWH(x, labelTop, labelW, labelH);

  //   canvas.drawRect(labelRect, Paint()..color = color);
  //   textPainter.paint(canvas, Offset(labelRect.left + 4, labelRect.top + 2));
  // }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) =>
      oldDelegate.detections != detections;
}
