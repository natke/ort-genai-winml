# ORT GenAI Samples

End-to-end AI chat examples using [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai).

## Files

| File | Description |
|---|---|
| `model-chat.py` | Interactive chat with an ONNX model using ORT GenAI |
| `common.py` | Shared utilities (config, search options, chat templates, guidance, EP registration) |
| `winml.py` | Windows ML integration for automatic execution provider discovery and registration |

## Prerequisites

- Python 3.10+
- An ONNX model folder containing `genai_config.json` and the model files

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 2. Install required packages

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ --extra-index-url https://pypi.org/simple onnxruntime-winml 
pip install onnxruntime-genai-winml==0.11.2
pip install wasdk-Microsoft.Windows.AI.MachineLearning[all]
pip install winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap
```

## Usage

### Use Windows ML to register execution providers

```bash
python model-chat.py -m <model_path> --use_winml
```

Optionally combine with `--ep_path` to provide a specific provider library path:

```bash
python model-chat.py -m <model_path> --use_winml --ep_path "C:\path\to\onnxruntime_providers_openvino_plugin.dll"
```

### Example: Chat with Qwen 2.5 1.5B on OpenVINO GPU

```bash
python model-chat.py  --use_winml -m qwen2.5-1.5b-openvino-gpu:2
```


