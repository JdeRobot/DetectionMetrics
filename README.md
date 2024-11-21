<a href="https://mmg-ai.com/en/"><img src="https://jderobot.github.io/assets/images/logo.png" width="100 " align="right" /></a>

# Detection Metrics


#### Docs [here](https://html-preview.github.io/?url=https://github.com/JdeRobot/DetectionMetrics/blob/dph/v2/py_docs/_build/html/index.html)

Detection Metrics is a set of tools to evaluate semantic segmentation and object detection models. With a current focus on semantic segmentation in unstructured outdoor environments, it is built in such a way that it can be easily extended to new tasks, datasets or deep learning frameworks.

# What's supported in Detection Metrics
## Image semantic segmentation
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
                "mean":[<r>, <g>, <b>],
                "std": : [<r>, <g>, <b>]
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

## LiDAR semantic segmentation
Coming soon.

## Object detection
Coming soon.


# Installation
### Using venv
Create your virtual environment:
```
mkdir .venv
python3 -m venv .venv
```

Activate your environment and install as pip package:
```
source .venv/bin/activate
pip install -e .
```

### Using poetry

Install Poetry (if not done before):
```
python3 -m pip install --user pipx
pipx install poetry
```

Install dependencies and activate poetry environment (you can get out of the Poetry shell by running `exit`):
```
poetry install
poetry shell
```

### Common
Install your deep learning framework of preference in your environment. We have tested:
- CUDA Version: `12.6`
- `torch==2.4.1`
- `torchvision==0.19.1`
- `tensorflow[and-cuda]==2.17.1`

And it's done! You can check the `examples` directory for inspiration and run some of the scripts provided either by activating the created environment using `poetry shell` or directly running `poetry run python examples/<some_python_script.py>`.
