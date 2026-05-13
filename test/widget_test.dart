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
