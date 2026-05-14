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
  Color(0xFF43A047),
];

class BoundingBoxPainter extends CustomPainter {
  final List<Recognition> detections;
  final bool isDarkMode;

  BoundingBoxPainter(this.detections, {this.isDarkMode = false});

  // Desenha círculo com número no centro da box
  void _drawNumberedCircle(
      Canvas canvas, Size size, Recognition d, Color color, int number) {
    // Centro da box em pixels
    final cx = (d.location.left + d.location.right) / 2 * size.width;
    final cy = (d.location.top + d.location.bottom) / 2 * size.height;

    // Raio: 30% da menor dimensão, mas nunca ultrapassa a borda com menos de 3px de margem
    final boxWidth = d.location.width * size.width;
    final boxHeight = d.location.height * size.height;
    final halfMin = min(boxWidth, boxHeight) / 2;
    const padding = 3.0;
    final double maxAllowed = max(4.0, halfMin - padding);
    final double radius = (min(boxWidth, boxHeight) * 0.30).clamp(4.0, min(11.0, maxAllowed));
    final fontSize = (radius * 0.9).clamp(4.0, 9.0);

    // Círculo de fundo
    final circlePaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;
    canvas.drawCircle(Offset(cx, cy), radius, circlePaint);

    // Borda branca para contraste
    final borderPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.5;
    canvas.drawCircle(Offset(cx, cy), radius, borderPaint);

    // Número centralizado — fonte proporcional ao raio
    final textSpan = TextSpan(
      text: '$number',
      style: TextStyle(
        color: Colors.white,
        fontSize: fontSize,
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
    final rect = Rect.fromLTRB(
      d.location.left   * size.width,
      d.location.top    * size.height,
      d.location.right  * size.width,
      d.location.bottom * size.height,
    );

  

    final boxPaint = Paint()
      ..color = color.withValues(alpha: 0.75)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    canvas.drawRect(rect, boxPaint);
    // _drawLabel(canvas, rect.left, rect.top, d, color);
  }

  // ── Box rotacionado (modelos OBB) ─────────────────────────────────────────
  // O ângulo θ do YOLOv8-OBB é definido em relação ao eixo X,
  // positivo no sentido horário, em radianos.
  void _drawOBB(Canvas canvas, Size size, Recognition d, Color color) {
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


    final boxPaint = Paint()
      ..color = color.withValues(alpha: 0.75)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    canvas.drawPath(path, boxPaint);

    // Linha indicando a direção/orientação (topo do box)
    final directionPaint = Paint()
      ..color = color.withValues(alpha: 0.75)
      ..strokeWidth = 1.5
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
