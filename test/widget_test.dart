import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tryingg_flutter/main.dart';

void _noopToggle() {}

void main() {
  testWidgets('App loads with initial empty state', (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(
      home: YoloApp(onToggleTheme: _noopToggle, isDarkMode: false),
    ));
    await tester.pump();

    expect(find.byIcon(Icons.image_search), findsOneWidget);
    expect(find.text('Selecione uma imagem para começar'), findsOneWidget);
  });

  testWidgets('Undo button not visible before edit mode', (WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(
      home: YoloApp(onToggleTheme: _noopToggle, isDarkMode: false),
    ));
    await tester.pump();

    expect(find.byIcon(Icons.undo), findsNothing);
  });
}
