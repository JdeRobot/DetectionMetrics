---
layout: home
title: Compatibility
permalink: /v2/compatibility/

sidebar:
  nav: "main_v2"
---

# Image semantic segmentation
- Datasets:
    - [Rellis3D](https://www.unmannedlab.org/research/RELLIS-3D)
    - [GOOSE](https://goose-dataset.de/)
    - Custom GAIA format
- Models:
    - PyTorch ([TorchScript](https://pytorch.org/docs/stable/jit.html) format):
        - Input shape: `(batch, channels, height, width)`
        - Output shape: `(batch, classes, height, width)`
        - JSON configuration file format:

        ```json
        {
            "normalization": {
                "mean": [<r>, <g>, <b>],
                "std": [<r>, <g>, <b>]
            }
        }
        ```
    - Tensorflow ([SavedModel](https://www.tensorflow.org/guide/saved_mode`) format):
        - Input shape: `(batch, height, width, channels)`
        - Output shape: `(batch, height, width, classes)`
        - JSON configuration file format:

        ```json
        {
            "image_size": [<height>, <width>]
        }
        ```
    - ONNX: coming soon
- Metrics:
    - Intersection over Union (IoU)

# LiDAR semantic segmentation
Coming soon.

# Object detection
Coming soon.