# ORT GenAI Samples

End-to-end AI chat examples using [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai).

## Files

| File | Description |
|---|---|
| `model-chat.py` | Interactive chat with an ONNX model using ORT GenAI |
| `common.py` | Shared utilities (config, search options, chat templates, guidance, EP registration) |
| `winml.py` | Windows ML integration for automatic execution provider discovery and registration |
| `test_ort_ep.py` | Standalone ORT test: register an EP via WinML and load a model with `onnxruntime` |

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
pip install onnxruntime==1.23.2
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
python model-chat.py  --use_winml -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2
```

### Verbose ORT logging

To enable verbose ONNX Runtime session logs (e.g. EP node assignment, graph partitioning), use the `--ort_log_level` flag:

```bash
python model-chat.py -m <model_path> --use_winml --ort_log_level VERBOSE
```

Log levels: `VERBOSE`, `INFO`, `WARNING`, `ERROR`, `FATAL`

Use `--log_file` to capture ORT logs to a file without interfering with the interactive prompt:

```bash
python model-chat.py -m <model_path> --use_winml --ort_log_level VERBOSE --log_file ort_verbose.log
```

> **Note:** Do not use PowerShell `2>` redirection with verbose logging — it blocks `input()` due to pipe buffer issues.

This is separate from the script-level flags:
- `-v` / `--verbose` — print script status messages and timing
- `-d` / `--debug` — dump ORT GenAI model input/output tensors

### Directly registering an EP (without WinML)

You can register an EP DLL directly with `--ep_path`. The EP type is auto-detected from the DLL name:

```bash
python model-chat.py -m <model_path> --ep_path "C:\path\to\onnxruntime_providers_openvino_plugin.dll"
```

Or explicitly specify the EP:

```bash
python model-chat.py -e openvino -m <model_path> --ep_path "C:\path\to\onnxruntime_providers_openvino_plugin.dll"
```

### Comparing OpenVINO EP versions with verbose logging

The `msix/OpenVINO-EP/x64/Release/` folder contains OpenVINO EP packages for testing:

| File | Description |
|---|---|
| `OpenVINOEP.1.8.61.0.msix` | OpenVINO EP 1.8.61 (working on NVIDIA GPU systems) |
| `OpenVINOEP.1.8.63.0.msix` | OpenVINO EP 1.8.63 (broken on NVIDIA GPU systems) |

Check the currently installed version:

```powershell
Get-AppxPackage *openvino* | Select-Object Name, Version, PackageFullName
```

#### Run with OpenVINO EP 1.8.61 (working):

Uninstall the current version and install 1.8.61:

```powershell
Get-AppxPackage *openvino* | Remove-AppxPackage
Add-AppxPackage -Path .\msix\OpenVINO-EP\x64\Release\OpenVINOEP.1.8.61.0.msix
python model-chat.py -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2 --use_winml --providers OpenVINOExecutionProvider --ort_log_level VERBOSE --log_file ort-ov-1.8.61.log
```

Or, use `--ep_path` to point directly to the DLL and register directly with ORT:

```powershell
Get-AppxPackage *openvino* | Remove-AppxPackage
Add-AppxPackage -Path .\msix\OpenVINO-EP\x64\Release\OpenVINOEP.1.8.61.0.msix
python model-chat.py -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2 --ep_path "C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_1.8.61.0_x64__8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_openvino_plugin.dll" --ort_log_level VERBOSE --log_file ort-ov-1.8.61.log
```

#### Run with OpenVINO EP 1.8.63 (broken on NVIDIA GPU systems):

Uninstall and install 1.8.63:

```powershell
Get-AppxPackage *openvino* | Remove-AppxPackage
Add-AppxPackage -Path .\msix\OpenVINO-EP\x64\Release\OpenVINOEP.1.8.63.0.msix
python model-chat.py -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2 --use_winml --providers OpenVINOExecutionProvider --ort_log_level VERBOSE --log_file ort-ov-1.8.63.log
```

Or, use `--ep_path` to point directly to the DLL and register directly with ORT:

```powershell
Get-AppxPackage *openvino* | Remove-AppxPackage
Add-AppxPackage -Path .\msix\OpenVINO-EP\x64\Release\OpenVINOEP.1.8.63.0.msix
python model-chat.py -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2 --ep_path "C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_1.8.63.0_x64__8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_openvino_plugin.dll" --ort_log_level VERBOSE --log_file ort-ov-1.8.63.log
```

Read the logs:

```powershell
Get-Content ort-ov-1.8.61.log
Get-Content ort-ov-1.8.63.log
```

### Standalone ORT EP test (test_ort_ep.py)

`test_ort_ep.py` registers an EP via WinML into the pip `onnxruntime` package (not ORT GenAI) and creates an `InferenceSession` to verify the EP loads and is active.

> **Note:** `onnxruntime-genai-winml` statically links its own ORT runtime. The pip `onnxruntime` package is a separate runtime, so results here may differ from `model-chat.py`.

#### Basic usage

```powershell
python test_ort_ep.py -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2
```

#### Specify device type

```powershell
python test_ort_ep.py -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2 --device_type GPU
```

#### Registration check only (skip session creation)

```powershell
python test_ort_ep.py -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2 --skip_session
```

#### Verbose ORT logging

```powershell
python test_ort_ep.py -m $env:USERPROFILE\.foundry\cache\Microsoft\qwen2.5-1.5b-instruct-openvino-gpu-2\v2 --log_level 0
```

#### Specify a different ONNX file

```powershell
python test_ort_ep.py -m <model_path> --onnx_file model.onnx
```


