import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'yolo_service.dart';
import 'widgets/box_painter.dart';

void main() => runApp(const MaterialApp(
      home: YoloApp(),
      debugShowCheckedModeBanner: false,
    ));

class YoloApp extends StatefulWidget {
  const YoloApp({super.key});

  @override
  State<YoloApp> createState() => _YoloAppState();
}

class _YoloAppState extends State<YoloApp> {
  final YoloService _yoloService = YoloService();
  File? _imageFile;
  img.Image? _decodedImage; // Guarda a imagem decodificada para obter dimensões reais
  List<Recognition> _results = [];
  bool _isProcessing = false;
  bool _modelReady = false;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await _yoloService.init();
    setState(() => _modelReady = true);
  }

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
      _isProcessing = true;
      _results = [];
      _imageFile = File(picked.path);
      _decodedImage = null;
    });

    try {
      final bytes = await _imageFile!.readAsBytes();
      final decoded = img.decodeImage(bytes);

      if (decoded != null) {
        final detections = await _yoloService.runInference(decoded);
        setState(() {
          _decodedImage = decoded;
          _results = detections;
          _isProcessing = false;
        });       
      } else {
        setState(() => _isProcessing = false);
      }
    } catch (e) {
      debugPrint('Erro no processamento: $e');
      setState(() => _isProcessing = false);
    }
  }

  /// Monta o texto de resumo com contagem por classe
  String _getSummary() {
    if (_isProcessing) return 'Processando...';
    if (_results.isEmpty) return 'Nenhum objeto detectado.';

    final Map<String, int> counts = {};
    for (var r in _results) {
      counts[r.label] = (counts[r.label] ?? 0) + 1;
    }
    return counts.entries.map((e) => '${e.value}x ${e.key}').join('  |  ');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        title: const Text('YOLOv8 Detector'),
        backgroundColor: const Color(0xFF16213E),
        foregroundColor: Colors.white,
        elevation: 0,
        actions: [
          if (!_modelReady)
            const Padding(
              padding: EdgeInsets.all(12.0),
              child: SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Colors.white,
                ),
              ),
            ),
          if (_modelReady)
            const Padding(
              padding: EdgeInsets.all(14.0),
              child: Icon(Icons.check_circle, color: Colors.greenAccent, size: 22),
            ),
        ],
      ),
      body: Column(
        children: [
          // --- Área da imagem ---
          Expanded(
            child: _imageFile == null
                ? const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.image_search, size: 72, color: Colors.white24),
                        SizedBox(height: 16),
                        Text(
                          'Selecione uma imagem para começar',
                          style: TextStyle(color: Colors.white38, fontSize: 16),
                        ),
                      ],
                    ),
                  )
                : Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      // LayoutBuilder para obter o tamanho real do widget
                      child: LayoutBuilder(
                        builder: (context, constraints) {
                          // Calcula o aspect ratio real da imagem original
                          // para que as bounding boxes se alinhem corretamente
                          double aspectRatio = 1.0;
                          if (_decodedImage != null) {
                            aspectRatio =
                                _decodedImage!.width / _decodedImage!.height;
                          }

                          return AspectRatio(
                            aspectRatio: aspectRatio,
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                // Imagem de fundo
                                Image.file(
                                  _imageFile!,
                                  fit: BoxFit.fill, // Fill para alinhar com as boxes
                                ),
                                // Bounding Boxes
                                if (_results.isNotEmpty)
                                  CustomPaint(
                                    painter: BoundingBoxPainter(_results),
                                  ),
                                // Loading overlay
                                if (_isProcessing)
                                  Container(
                                    color: Colors.black45,
                                    child: const Center(
                                      child: Column(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          CircularProgressIndicator(
                                              color: Colors.white),
                                          SizedBox(height: 12),
                                          Text('Detectando...',
                                              style: TextStyle(
                                                  color: Colors.white)),
                                        ],
                                      ),
                                    ),
                                  ),
                              ],
                            ),
                          );
                        },
                      ),
                    ),
                  ),
          ),

          // --- Painel de resultados ---
          Container(
            width: double.infinity,
            margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(
              color: const Color(0xFF16213E),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.blueAccent.withOpacity(0.4)), 
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Detecções',
                  style: TextStyle(
                      color: Colors.white54,
                      fontSize: 12,
                      fontWeight: FontWeight.w500),
                ),
                const SizedBox(height: 4),
                Text(
                  _getSummary(),
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                if (_results.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: Text(
                      'Total: ${_results.length} objeto(s)',
                      style: const TextStyle(
                          color: Colors.blueAccent, fontSize: 13),
                    ),
                  ),
              ],
            ),
          ),

          // --- Botões ---
          Padding(
            padding: const EdgeInsets.only(bottom: 36, top: 8),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _ActionButton(
                  icon: Icons.photo_library_rounded,
                  label: 'Galeria',
                  onTap: () => _processImage(ImageSource.gallery),
                ),
                _ActionButton(
                  icon: Icons.camera_alt_rounded,
                  label: 'Câmera',
                  onTap: () => _processImage(ImageSource.camera),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: onTap,
      icon: Icon(icon),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.blueAccent,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    );
  }
}
