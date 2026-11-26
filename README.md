# Detective Rice!

<img src="Golden%20rice%20detective%20icon.png" alt="Detective Rice logo" width="124">

Detective Rice! is a computer vision project that identifies a single uncooked, dehusked rice grain from six Bangladeshi varieties. The Streamlit app displays the three highest-scoring varieties in English and Bangla, along with the 224 × 224 preview used to prepare the model input.

The classifier uses a trained EfficientNet-B0 checkpoint and runs locally on CPU with either PyTorch or ONNX Runtime. The repository includes both model files and 12 labeled test photos, so you can try the app without training a model or supplying your own image.

## Quick start

Use Python 3.11 or 3.12. Clone the repository and enter the project directory.

```bash
git clone https://github.com/AhsanCS64/Detective-Rice.git
cd Detective-Rice
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-pytorch.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS and Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Install PyTorch and TorchVision using the command for your platform, then launch the app.

- **Linux** — `.venv/bin/python -m pip install -r requirements-pytorch.txt`
- **macOS** — `.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1`

```bash
.venv/bin/python -m streamlit run app.py
```

**ONNX Runtime** is the default inference engine and runs with the packages in `requirements.txt`. Install the optional PyTorch dependencies if you want to switch to **PyTorch** in the app.

## Using the app

1. Choose **PyTorch** or **ONNX Runtime**.
2. Select a **Sample photo** or upload a JPEG, PNG, or WebP image. For your own photo, use a sharp close-up of one dehusked grain on a matte black background.
3. Review the three ranked predictions and the model-input preview. For a sample photo, the app also shows the recorded test-set label.

The bundled samples include two images per variety from the held-out test split. See [sample labels and provenance](Testing%20Samples/labels.csv) for their original dataset paths.

## Supported varieties

The table follows the model's output order.

| Output index | Variety | Dataset class |
| ---: | --- | --- |
| 0 | BRRI dhan49 | `BD49` |
| 1 | BRRI dhan52 | `BD52` |
| 2 | BRRI dhan67 | `BRRI67` |
| 3 | BRRI dhan74 | `BRRI74` |
| 4 | Binadhan-7 | `Binadhan7` |
| 5 | Binadhan-20 | `Binadhan20` |

## Model and evaluation

The training run selected the epoch 3 EfficientNet-B0 checkpoint by validation macro F1. Both exports were compared on the same 495 held-out microscope images.

| Metric | Result |
| --- | ---: |
| Validation macro F1 | 0.9799 |
| Held-out top-1 accuracy, both runtimes | 98.79% (489 of 495) |
| PyTorch and ONNX top-1 agreement | 495 of 495 |
| PyTorch and ONNX ordered top-3 agreement | 495 of 495 |
| Maximum absolute logit difference | 0.003293 |

The shared preprocessing pipeline corrects EXIF orientation, converts the image to RGB, and resizes it proportionally with centered black padding. It then scales and normalizes the pixels. The ONNX export accepts float32 tensors shaped `[N, 3, 224, 224]` and returns six logits per image. The app applies softmax to those logits to display the scores.

The reported accuracy applies to the held-out microscope images. Performance on fresh phone photos is unmeasured. The displayed scores are relative to the six supported varieties, and the app will still return a ranking for an unsupported variety or an out-of-scope image.

[Evaluation report](parity_report.json) · [Per-image predictions](parity_predictions.csv) · [Kaggle run and outputs](https://www.kaggle.com/code/hmashanhabibkushal/detective-rice/output?scriptVersionId=374163214)

## Repository contents

| Path | Purpose |
| --- | --- |
| [`app.py`](app.py) | Streamlit interface |
| [`rice_inference.py`](rice_inference.py) | Shared preprocessing and model loaders |
| [`detective-rice.ipynb`](detective-rice.ipynb) | Project notebook |
| [`rice_variety_model.pth`](rice_variety_model.pth) | PyTorch checkpoint with model weights and metadata |
| [`rice_variety_model.onnx`](rice_variety_model.onnx) | ONNX export with a dynamic batch dimension |
| [`model_metadata.json`](model_metadata.json) | Class order, normalization settings, and input and output specifications |
| [`Testing Samples/`](Testing%20Samples) | Labeled test photos used by the in-app sample picker |

## Dataset attribution

The project uses **Rice Varieties in Bangladesh** by Md Tahsin and co-authors. Refer to the dataset record for its license and attribution terms.

[Original dataset record](https://doi.org/10.17632/2fgv99854n.1) · [Kaggle dataset mirror](https://www.kaggle.com/datasets/fahadbd/rice-varieties-in-bangladesh)
