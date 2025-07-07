// PreviewScreen.dart
import 'dart:typed_data';
import 'package:flutter/material.dart';

class PreviewScreen extends StatelessWidget {
  final Uint8List frontImageBytes;
  final Uint8List backImageBytes;

  const PreviewScreen({
    super.key,
    required this.frontImageBytes,
    required this.backImageBytes,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('CNIC Preview')),
      body: Column(
        children: [
          const Text('Front Side', style: TextStyle(fontSize: 20)),
          Expanded(
            child: Image.memory(frontImageBytes),
          ),
          const Divider(),
          const Text('Back Side', style: TextStyle(fontSize: 20)),
          Expanded(
            child: Image.memory(backImageBytes),
          ),
        ],
      ),
    );
  }
}