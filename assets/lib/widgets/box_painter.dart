import 'package:flutter/material.dart';
import '../yolo_service.dart';

// Paleta de cores por classe (cicla automaticamente)
const List<Color> _boxColors = [
  Color(0xFFE53935), // vermelho
  Color(0xFF1E88E5), // azul
  Color(0xFF43A047), // verde
  Color(0xFFFB8C00), // laranja
  Color(0xFF8E24AA), // roxo
  Color(0xFF00ACC1), // ciano
  Color(0xFFFFB300), // amarelo
  Color(0xFF6D4C41), // marrom
  Color(0xFF00897B), // teal
  Color(0xFFD81B60), // pink
];

class BoundingBoxPainter extends CustomPainter {
  final List<Recognition> detections;

  BoundingBoxPainter(this.detections);

  @override
  void paint(Canvas canvas, Size size) {
    for (var d in detections) {
      final color = _boxColors[d.classId % _boxColors.length];

      final boxPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;

      // Converte de coordenadas normalizadas (0-1) para pixels do widget
      final rect = Rect.fromLTRB(
        d.location.left   * size.width,
        d.location.top    * size.height,
        d.location.right  * size.width,
        d.location.bottom * size.height,
      );

      // Desenha a bounding box
      canvas.drawRect(rect, boxPaint);

      // --- Label com fundo colorido ---
      final String labelText =
          '${d.label} ${(d.score * 100).toStringAsFixed(0)}%';

      final textSpan = TextSpan(
        text: labelText,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final labelWidth  = textPainter.width + 8;
      final labelHeight = textPainter.height + 4;

      // Posiciona o label acima da box, mas nunca fora da tela
      double labelTop = rect.top - labelHeight;
      if (labelTop < 0) labelTop = rect.top;

      final labelRect = Rect.fromLTWH(
        rect.left,
        labelTop,
        labelWidth,
        labelHeight,
      );

      // Fundo do label
      canvas.drawRect(
        labelRect,
        Paint()..color = color,
      );

      // Texto do label
      textPainter.paint(
        canvas,
        Offset(labelRect.left + 4, labelRect.top + 2),
      );
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) =>
      oldDelegate.detections != detections;
}
