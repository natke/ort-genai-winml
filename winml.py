import sys
from pathlib import Path
import traceback

_winml_instance = None

class WinML:
    def __new__(cls, *args, **kwargs):
        global _winml_instance
        if _winml_instance is None:
            _winml_instance = super(WinML, cls).__new__(cls, *args, **kwargs)
            _winml_instance._initialized = False
        return _winml_instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._fix_winrt_runtime()
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions,
            initialize
        )
        import winui3.microsoft.windows.ai.machinelearning as winml
        self._win_app_sdk_handle = initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI, version="1.8")
        self._win_app_sdk_handle.__enter__()
        catalog = winml.ExecutionProviderCatalog.get_default()
        self._providers = catalog.find_all_providers()
        self._ep_info : dict[str, object] = {}  # name -> provider object (not yet ensured ready)
        self._ep_paths : dict[str, str] = {}    # name -> library_path (populated on demand)
        for provider in self._providers:
            self._ep_info[provider.name] = provider
        self._registered_eps : dict[str, list[str]] = {"onnxruntime": [], "onnxruntime_genai": []}

    def __del__(self):
        self._providers = None
        self._win_app_sdk_handle.__exit__(None, None, None)

    def _fix_winrt_runtime(self):
        """
        This function removes the msvcp140.dll from the winrt-runtime package.
        So it does not cause issues with other libraries.
        """
        from importlib import metadata
        site_packages_path = Path(str(metadata.distribution('winrt-runtime').locate_file('')))
        dll_path = site_packages_path / 'winrt' / 'msvcp140.dll'
        if dll_path.exists():
            dll_path.unlink()

    def _ensure_ready(self, name: str) -> str | None:
        """Ensure a single EP is ready and return its library path (or None if unavailable).
        
        Only calls ensure_ready_async for the specific EP requested, avoiding
        loading DLLs for unrelated EPs into the process.
        """
        if name in self._ep_paths:
            return self._ep_paths[name]
        provider = self._ep_info.get(name)
        if provider is None:
            return None
        provider.ensure_ready_async().get()
        if provider.library_path == '':
            return None
        self._ep_paths[name] = provider.library_path
        return provider.library_path

    def register_execution_providers(self, ort=True, ort_genai=False, providers: list[str] | None = None) -> dict[str, list[str]]:
        modules = []
        if ort:
            import onnxruntime
            modules.append(onnxruntime)
        if ort_genai:
            import onnxruntime_genai
            modules.append(onnxruntime_genai)
        # Determine which EPs to register
        ep_names = list(providers) if providers is not None else list(self._ep_info.keys())
        for name in ep_names:
            path = self._ensure_ready(name)
            if path is None:
                continue
            version = self._get_version_from_path(path)
            for module in modules:
                if name not in self._registered_eps[module.__name__]:
                    try:
                        module.register_execution_provider_library(name, path)
                        self._registered_eps[module.__name__].append(name)
                        ver_str = f" v{version}" if version else ""
                        print(f"Registered {name}{ver_str} with {module.__name__}")
                    except Exception as e:
                        print(f"Failed to register execution provider {name}: {e}", file=sys.stderr)
                        traceback.print_exc()
        return self._registered_eps

    @staticmethod
    def _get_version_from_path(path: str) -> str | None:
        """Extract version from a WindowsApps package path, e.g. '1.8.63.0' from
        '...MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_1.8.63.0_x64__...'"""
        import re
        m = re.search(r'_(\d+\.\d+\.\d+\.\d+)_', path)
        return m.group(1) if m else None


def register_execution_providers(ort=True, ort_genai=False, providers: list[str] | None = None) -> dict[str, list[str]]:
    """Register WinML execution providers for ONNX Runtime and ONNX Runtime GenAI.

    Args:
        ort (bool): Whether to register for ONNX Runtime.
        ort_genai (bool): Whether to register for ONNX Runtime GenAI.
        providers (list[str] | None): Only register these providers by name. None means all.

    Returns:
        dict[str, list[str]]: Dictionary of registered execution provider names by module.
    """
    return WinML().register_execution_providers(ort=ort, ort_genai=ort_genai, providers=providers)


def add_ep_for_device(session_options, ep_name, device_type, ep_options=None):
    """Ensures correct EP device selection for WinML. NEVER modify this function.
    ep_name is one of:
        - "CPUExecutionProvider"
        - "DmlExecutionProvider"
        - "WebGpuExecutionProvider"
        - "QNNExecutionProvider"
        - "OpenVINOExecutionProvider"
        - "VitisAIExecutionProvider"
        - "NvTensorRTRTXExecutionProvider"

    device_type is one of:
        - ort.OrtHardwareDeviceType.CPU
        - ort.OrtHardwareDeviceType.GPU
        - ort.OrtHardwareDeviceType.NPU
    """
    import onnxruntime as ort
    ep_devices = ort.get_ep_devices()
    for ep_device in ep_devices:
        if ep_device.ep_name == ep_name and ep_device.device.type == device_type:
            print(f"Adding {ep_name} for {device_type}")
            session_options.add_provider_for_devices([ep_device], {} if ep_options is None else ep_options)
            break
