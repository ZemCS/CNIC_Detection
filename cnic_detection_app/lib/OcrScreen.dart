import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

class CameraScanScreen extends StatefulWidget {
  const CameraScanScreen({super.key});

  @override
  _CameraScanScreenState createState() => _CameraScanScreenState();
}

enum CaptureState { front, back }

class _CameraScanScreenState extends State<CameraScanScreen>
    with SingleTickerProviderStateMixin {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;
  late AnimationController _scanAnimation;
  bool _isLoading = false;
  CaptureState _captureState = CaptureState.front;
  Uint8List? _frontImageBytes;
  Uint8List? _backImageBytes;

  final String _apiUrl = 'http://192.168.18.112:5000/cnic';

  @override
  void initState() {
    super.initState();
    _initializeControllerFuture = _initializeController();
    _scanAnimation = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat();
  }

  Future<void> _initializeController() async {
    final cameras = await availableCameras();
    final camera = cameras.first;
    _controller = CameraController(
      camera,
      ResolutionPreset.high,
      enableAudio: false,
    );
    await _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    _scanAnimation.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      body: Stack(
        children: [
          FutureBuilder<void>(
            future: _initializeControllerFuture,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.done) {
                return Stack(
                  children: [CameraPreview(_controller), _buildScanOverlay()],
                );
              } else if (snapshot.hasError) {
                return Center(child: Text('Camera error: ${snapshot.error}'));
              } else {
                return const Center(child: CircularProgressIndicator());
              }
            },
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              height: 120,
              margin: const EdgeInsets.only(bottom: 20),
              width: double.infinity,
              color: Theme.of(context).scaffoldBackgroundColor,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    _captureState == CaptureState.front
                        ? 'Capture FRONT of CNIC'
                        : 'Capture BACK of CNIC',
                    style: const TextStyle(
                      color: Colors.black,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  FloatingActionButton(
                    onPressed: _captureAndProcess,
                    child: const Icon(Icons.camera),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScanOverlay() {
    return Positioned.fill(
      child: AnimatedBuilder(
        animation: _scanAnimation,
        builder: (context, child) {
          return CustomPaint(painter: _ScanPainter(_scanAnimation.value));
        },
      ),
    );
  }

  Future<List<XFile>> _captureBurst({int count = 1}) async {
    final List<XFile> shots = [];
    for (int i = 0; i < count; i++) {
      shots.add(await _controller.takePicture());
      await Future.delayed(const Duration(milliseconds: 5));
    }
    return shots;
  }

  Future<void> _captureAndProcess() async {
    if (_isLoading) return;
    setState(() => _isLoading = true);

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 16),
            Text('Processing ${_captureState.name} side...'),
          ],
        ),
      ),
    );

    try {
      await _initializeControllerFuture;
      final burst = await _captureBurst(count: 5);
      double bestScore = -1.0;
      img.Image? bestImage;
      for (final file in burst) {
        final bytes = await File(file.path).readAsBytes();
        final frame = img.decodeImage(bytes)!;
        final score = _varianceOfLaplacian(frame);
        if (score > bestScore) {
          bestScore = score;
          bestImage = frame;
        }
      }
      if (bestImage == null) throw Exception('No clear frame');
      bestImage = img.adjustColor(bestImage, contrast: 1.2);
      final jpg = img.encodeJpg(bestImage, quality: 100);

      if (_captureState == CaptureState.front) {
        _frontImageBytes = Uint8List.fromList(jpg);
        setState(() => _captureState = CaptureState.back);
        if (mounted && Navigator.canPop(context)) {
          Navigator.of(context).pop(); // Pop processing dialog
        }
      } else {
        _backImageBytes = Uint8List.fromList(jpg);
        if (mounted && Navigator.canPop(context)) {
          Navigator.of(context).pop(); // Pop processing dialog before API call
        }
        await _sendToApi();
      }
    } catch (e) {
      if (mounted && Navigator.canPop(context)) {
        Navigator.of(context).pop();
      }
      debugPrint('Error: $e');
      _showErrorDialog('Capture error: $e');
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _sendToApi() async {
    if (_frontImageBytes == null || _backImageBytes == null) {
      if (mounted && Navigator.canPop(context)) {
        Navigator.of(context).pop();
      }
      _showErrorDialog('Both images must be captured');
      // Reset state to front
      if (mounted) {
        setState(() {
          _captureState = CaptureState.front;
          _frontImageBytes = null;
          _backImageBytes = null;
          _isLoading = false;
        });
      }
      return;
    }

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => const AlertDialog(
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Sending to server...'),
          ],
        ),
      ),
    );

    try {
      final uri = Uri.parse(_apiUrl);
      final request = http.MultipartRequest('POST', uri)
        ..files.add(
          http.MultipartFile.fromBytes(
            'file_front',
            _frontImageBytes!,
            filename: 'front.jpg',
          ),
        )
        ..files.add(
          http.MultipartFile.fromBytes(
            'file_back',
            _backImageBytes!,
            filename: 'back.jpg',
          ),
        );

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      debugPrint('Response status: ${response.statusCode}');
      debugPrint('Response body: $responseBody');

      if (mounted && Navigator.canPop(context)) {
        Navigator.of(context).pop(); // Pop 'Sending to server...' dialog
      }

      if (response.statusCode >= 200 && response.statusCode < 300) {
        // Successful response, navigate to ResponseScreen
        if (mounted) {
          await Navigator.of(context).push(
            MaterialPageRoute(
              builder: (_) => ResponseScreen(
                response: responseBody,
                onCaptureAgain: () {
                  // No state reset here, handled below
                },
              ),
            ),
          );
        }
      } else {
        // Error response, show dialog
        if (mounted) {
          _showErrorDialog(responseBody);
        }
      }
    } catch (e) {
      if (mounted && Navigator.canPop(context)) {
        Navigator.of(context).pop();
      }
      debugPrint('API Error: $e');
      if (mounted) {
        _showErrorDialog('API Error: $e');
      }
    } finally {
      // Reset state to front after API call (success or failure)
      if (mounted) {
        setState(() {
          _captureState = CaptureState.front;
          _frontImageBytes = null;
          _backImageBytes = null;
          _isLoading = false;
        });
      }
    }
  }

  void _showErrorDialog(String message) {
    // Parse error message if it's JSON
    String formattedMessage = message;
    try {
      final json = jsonDecode(message) as Map<String, dynamic>;
      if (json.containsKey('error')) {
        formattedMessage = json['error'] as String;
        // Capitalize first letter for readability
        if (formattedMessage.isNotEmpty) {
          formattedMessage = formattedMessage[0].toUpperCase() + formattedMessage.substring(1);
        }
      }
    } catch (_) {
      // If not JSON, use the message as is
    }

    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text(
          'Error',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        content: Text(
          formattedMessage,
          style: const TextStyle(
            fontSize: 16,
            color: Colors.black,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text(
              'OK',
              style: TextStyle(fontSize: 16),
            ),
          ),
        ],
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12.0),
        ),
      ),
    );
  }

  double _varianceOfLaplacian(img.Image src) {
    final gray = img.grayscale(src);
    final width = gray.width;
    final height = gray.height;
    const kernel = [0, 1, 0, 1, -4, 1, 0, 1, 0];
    final lap = List<double>.filled(width * height, 0);

    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        double sum = 0;
        for (int ky = -1; ky <= 1; ky++) {
          for (int kx = -1; kx <= 1; kx++) {
            final pixel = gray.getPixel(x + kx, y + ky).r.toDouble();
            sum += kernel[(ky + 1) * 3 + (kx + 1)] * pixel;
          }
        }
        lap[y * width + x] = sum;
      }
    }

    final mean = lap.reduce((a, b) => a + b) / lap.length;
    final variance =
        lap.map((v) => math.pow(v - mean, 2)).reduce((a, b) => a + b) /
        lap.length;
    return variance.toDouble();
  }
}

class _ScanPainter extends CustomPainter {
  final double progress;

  _ScanPainter(this.progress);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.green.withOpacity(0.7);
    final y = size.height * progress;
    canvas.drawLine(Offset(0, y), Offset(size.width, y), paint);
  }

  @override
  bool shouldRepaint(covariant _ScanPainter oldDelegate) =>
      oldDelegate.progress != progress;
}

class ResponseScreen extends StatelessWidget {
  final String response;
  final VoidCallback onCaptureAgain;

  const ResponseScreen({
    super.key,
    required this.response,
    required this.onCaptureAgain,
  });

  @override
  Widget build(BuildContext context) {
    // Parse the JSON response
    Map<String, dynamic> data;
    try {
      data = jsonDecode(response) as Map<String, dynamic>;
    } catch (e) {
      return Scaffold(
        appBar: AppBar(title: const Text('API Response')),
        body: Center(
          child: Text('Error parsing response: $e'),
        ),
      );
    }

    final front = data['front'] as Map<String, dynamic>? ?? {};
    final back = data['back'] as Map<String, dynamic>? ?? {};
    final isCnic = data['front']?['isCNIC?']?.toString() ?? 'False';

    if (isCnic == 'False') {
      return Scaffold(
        appBar: AppBar(
          title: const Text(
            'CNIC Information',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          centerTitle: true,
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                const Text(
                  'The images input were not of a valid CNIC',
                  style: TextStyle(
                    fontSize: 16,
                    color: Colors.black,
                  ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: () {
                    onCaptureAgain();
                    Navigator.of(context).pop();
                  },
                  child: const Text('Capture Again'),
                ),
              ],
            ),
          ),
        ),
      );
    }

    // Define the fields to display
    final fields = [
      {'label': 'Name', 'value': front['Name'] ?? 'N/A'},
      {'label': 'Father Name', 'value': front['Father Name'] ?? 'N/A'},
      {'label': 'Gender', 'value': front['Gender'] ?? 'N/A'},
      {'label': 'Country of Stay', 'value': front['Country of Stay'] ?? 'N/A'},
      {
        'label': 'Identity Number',
        'value': front['Identity Number'] ?? back['Identity Number'] ?? 'N/A'
      },
      {'label': 'Date of Birth', 'value': front['Date of Birth'] ?? 'N/A'},
      {'label': 'Date of Issue', 'value': front['Date of Issue'] ?? 'N/A'},
      {'label': 'Date of Expiry', 'value': front['Date of Expiry'] ?? 'N/A'},
    ];

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'CNIC Information',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: ListView(
            shrinkWrap: true,
            children: [
              ...fields.map((field) => Padding(
                    padding: const EdgeInsets.only(bottom: 16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12.0,
                            vertical: 6.0,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.grey[200],
                            borderRadius: BorderRadius.circular(8.0),
                          ),
                          child: Text(
                            field['label']!,
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Colors.black,
                            ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                        const SizedBox(height: 4.0),
                        Text(
                          field['value']!,
                          style: const TextStyle(
                            fontSize: 16,
                            color: Colors.black,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  )),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  onCaptureAgain();
                  Navigator.of(context).pop();
                },
                child: const Text('Capture Again'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}