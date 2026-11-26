# Detective Rice 😼

The bundle includes the trained EfficientNet-B0 model in two formats, a shared preprocessing implementation, a local Streamlit app, and parity reports. The checkpoint is the Kaggle run's epoch 3 model, selected by validation macro F1.

## Run locally on Windows

Use 64-bit Python 3.11 or 3.12. Unzip the bundle, open PowerShell in the extracted folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pytorch.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use `-3.11` in the first line if that is your installed version. On macOS, use `python3 -m venv .venv`, install `torch==2.7.1 torchvision==0.22.1` from PyPI with `.venv/bin/python -m pip install`, then install `requirements.txt` and launch Streamlit using `.venv/bin/python`. On Linux the CPU requirements file works with the same `.venv/bin/python` executable. The app defaults to PyTorch and includes an ONNX Runtime switch.

## Model files

- `rice_variety_model.pth` contains a CPU state dictionary under `state_dict` and configuration under `metadata`. Load with `weights_only=True`.
- `rice_variety_model.onnx` accepts float32 `images` with shape `[N,3,224,224]` and returns float32 `logits` with shape `[N,6]`. Batch size is dynamic. Apply softmax across the six logits for display.
- `model_metadata.json` records output order, the data type and tensor layout, and the exact image preprocessing settings. It is also embedded in both model files.
- `parity_report.json` and `parity_predictions.csv` record the comparison on the original, fixed held-out split. `sha256.json` records file checksums.

The model expects a single, uncooked, dehusked grain. The original training and held-out data were microscope photos. Fresh phone-photo accuracy remains unmeasured.
