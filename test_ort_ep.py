# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Standalone ORT test: register the OpenVINO EP via WinML and load a model.

This uses the pip `onnxruntime` package. Note that `onnxruntime-genai-winml`
statically links its own ORT runtime — the pip `onnxruntime` is a *separate*
ORT build, so the WinML EP plugin DLL may not be ABI-compatible with it.

This script tests both registration approaches:
  1. register_execution_provider_library() — the traditional approach
  2. get_ep_devices() / add_provider_for_devices() — the newer device-based API

It also diagnoses the EP 1.8.63 ov_device mismatch bug: EP 1.8.63 reports
ov_device='GPU.0' instead of 'GPU', which causes onnxruntime-genai-winml
0.11.2's model.cpp exact-match to fail. Use --device_type GPU to check.

Usage:
    python test_ort_ep.py -m <model_folder>
    python test_ort_ep.py -m <model_folder> --device_type GPU
    python test_ort_ep.py -m <model_folder> --skip_session
    python test_ort_ep.py -m <model_folder> --onnx_file model.onnx
"""

import argparse
import glob
import os
import sys
import time


def find_onnx_model(model_path: str, onnx_file: str | None = None) -> str:
    """Locate the .onnx file inside a model folder."""
    model_path = os.path.expandvars(os.path.expanduser(model_path))
    if onnx_file:
        candidate = os.path.join(model_path, onnx_file)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Specified ONNX file not found: {candidate}")

    # Search for .onnx files
    candidates = glob.glob(os.path.join(model_path, "*.onnx"))
    if not candidates:
        # Try one level deeper (e.g. model_path/decoder/model.onnx)
        candidates = glob.glob(os.path.join(model_path, "**", "*.onnx"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .onnx files found in {model_path}")
    if len(candidates) > 1:
        print(f"Multiple ONNX files found, using first: {candidates}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Test OpenVINO EP registration and model loading with standalone ORT")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to model folder containing .onnx file(s)")
    parser.add_argument("--onnx_file", type=str, default=None, help="Specific .onnx filename within model_path (default: auto-detect)")
    parser.add_argument("--device_type", type=str, default=None, help="EP device_type option (e.g. GPU, GPU.0, CPU)")
    parser.add_argument("--ep", type=str, default="OpenVINOExecutionProvider", help="EP name to test (default: OpenVINOExecutionProvider)")
    parser.add_argument("--providers", type=str, nargs="+", default=None, help="WinML providers to register (default: only the --ep value)")
    parser.add_argument("--log_level", type=int, default=2, help="ORT session log severity (0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR)")
    parser.add_argument("--skip_session", action="store_true", help="Only register the EP, skip creating an InferenceSession")
    args = parser.parse_args()

    ep_name = args.ep
    providers_to_register = args.providers or [ep_name]

    # ── Step 1: Print ORT version info ─────────────────────────────────────
    import onnxruntime as ort
    print("=" * 60)
    print("Step 1: ORT version info")
    print("=" * 60)
    print(f"  onnxruntime version:  {ort.__version__}")
    print(f"  onnxruntime location: {os.path.dirname(ort.__file__)}")
    try:
        import onnxruntime_genai as og
        print(f"  onnxruntime-genai:    {og.__version__} (statically links its own ORT)")
    except ImportError:
        print(f"  onnxruntime-genai:    not installed")
    print(f"  Built-in providers:   {ort.get_available_providers()}")

    # ── Step 2: Register EP via WinML ──────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Step 2: Register {ep_name} via WinML")
    print("=" * 60)
    try:
        import winml
        result = winml.register_execution_providers(
            ort=True, ort_genai=False, providers=providers_to_register
        )
        print(f"Registration result: {result}")
    except Exception as e:
        print(f"FAILED to register EP: {e}")
        sys.exit(1)

    if ep_name not in result.get("onnxruntime", []):
        print(f"WARNING: {ep_name} was not registered with onnxruntime")
        print("Available WinML providers:", list(winml.WinML()._ep_info.keys()))
        sys.exit(1)

    # Show the EP DLL path
    w = winml.WinML()
    ep_path = w._ep_paths.get(ep_name, "unknown")
    print(f"EP library path: {ep_path}")

    # ── Step 3: Check EP devices (newer API) ──────────────────────────────
    print()
    print("=" * 60)
    print("Step 3: Check EP devices via get_ep_devices()")
    print("=" * 60)
    ep_device_match = None
    ep_devices_list = []  # collect for diagnosis
    try:
        ep_devices = ort.get_ep_devices()
        if ep_devices:
            for epd in ep_devices:
                marker = ""
                if epd.ep_name == ep_name:
                    marker = " <-- target EP"
                    if ep_device_match is None:
                        ep_device_match = epd  # take first match
                    if args.device_type and str(epd.device.type).endswith(args.device_type.upper()):
                        ep_device_match = epd
                ep_meta = getattr(epd, 'ep_metadata', {})
                print(f"  {epd.ep_name} on {epd.device.type}  ep_metadata={ep_meta}{marker}")
                ep_devices_list.append(epd)
        else:
            print("  (no EP devices returned)")
    except AttributeError:
        print("  ort.get_ep_devices() not available in this ORT version")
    except Exception as e:
        print(f"  get_ep_devices() failed: {e}")

    # ── Diagnosis: check ov_device vs device_type match ────────────────────
    #
    # EP 1.8.63 changed ov_device from "GPU" to "GPU.0". This breaks
    # onnxruntime-genai-winml 0.11.2's model.cpp exact string match.
    # See BUG_ANALYSIS.md for full details.
    #
    if ep_devices_list:
        print()
        print("  ── ov_device diagnosis ──")
        target_devices = [epd for epd in ep_devices_list if epd.ep_name == ep_name]
        if target_devices:
            for epd in target_devices:
                ep_meta = getattr(epd, 'ep_metadata', {})
                ov_device = ep_meta.get('ov_device', '(not set)')
                ep_version = ep_meta.get('version', '(unknown)')
                hw_type = str(epd.device.type).rsplit('.', 1)[-1]
                print(f"  {ep_name}: ov_device='{ov_device}', plugin={ep_version}, hardware={hw_type}")
                if args.device_type:
                    exact_match = (ov_device == args.device_type)
                    substr_match = (args.device_type in ov_device) if ov_device != '(not set)' else False
                    status = "MATCH" if exact_match else "MISMATCH"
                    print(f"    device_type='{args.device_type}' vs ov_device='{ov_device}': exact={status}")
                    if not exact_match and substr_match:
                        print(f"    ⚠ AFFECTED by EP 1.8.63 bug: ov_device '{ov_device}' contains '{args.device_type}' but is not an exact match.")
                        print(f"      onnxruntime-genai-winml 0.11.2 will FAIL (exact match in model.cpp).")
                        print(f"      onnxruntime-genai-winml 0.12.0 will PASS (substring match in interface.cpp).")
                    elif exact_match:
                        print(f"    ✓ ov_device exactly matches — both 0.11.2 and 0.12.0 will work.")
        else:
            print(f"  No devices found for {ep_name}")
        if not args.device_type:
            print(f"  (Use --device_type GPU to check for ov_device mismatch)")

    if args.skip_session:
        print("\n--skip_session specified, stopping here.")
        return

    # ── Step 4: Find model ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 4: Locate ONNX model")
    print("=" * 60)
    try:
        onnx_path = find_onnx_model(args.model_path, args.onnx_file)
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"Model: {onnx_path} ({size_mb:.1f} MB)")
    except FileNotFoundError as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # ── Step 5: Create InferenceSession ────────────────────────────────────
    #
    # Try two approaches:
    #   A) add_provider_for_devices() — the device-based API (if EP showed up in get_ep_devices)
    #   B) Standard providers=[...] list — falls back to this if A is not available
    #
    print()
    print("=" * 60)
    print(f"Step 5: Create InferenceSession with {ep_name}")
    print("=" * 60)

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = args.log_level
    sess_options.log_verbosity_level = 10

    provider_options = {}
    if args.device_type:
        provider_options["device_type"] = args.device_type
        print(f"  device_type = {args.device_type}")

    session = None

    # Approach A: add_provider_for_devices
    if ep_device_match is not None:
        print(f"\n  [A] Trying add_provider_for_devices({ep_device_match.ep_name}, {ep_device_match.device.type})...")
        try:
            sess_options_a = ort.SessionOptions()
            sess_options_a.log_severity_level = args.log_level
            sess_options_a.log_verbosity_level = 10
            sess_options_a.add_provider_for_devices(
                [ep_device_match],
                provider_options if provider_options else {}
            )
            t0 = time.time()
            session = ort.InferenceSession(onnx_path, sess_options_a)
            elapsed = time.time() - t0
            print(f"  [A] Session created in {elapsed:.2f}s")
        except Exception as e:
            print(f"  [A] FAILED: {e}")
            session = None

    # Approach B: standard providers list
    if session is None:
        providers_list = [(ep_name, provider_options)] if provider_options else [ep_name]
        providers_list.append("CPUExecutionProvider")
        print(f"\n  [B] Trying InferenceSession(providers={[p if isinstance(p,str) else p[0] for p in providers_list]})...")
        try:
            t0 = time.time()
            session = ort.InferenceSession(onnx_path, sess_options, providers=providers_list)
            elapsed = time.time() - t0
            print(f"  [B] Session created in {elapsed:.2f}s")
        except Exception as e:
            print(f"  [B] FAILED: {e}")
            print()
            print("=" * 60)
            print("DIAGNOSIS")
            print("=" * 60)
            print(f"The EP plugin DLL was registered but could not be loaded by")
            print(f"pip onnxruntime {ort.__version__}.")
            print()
            print(f"This is likely an ABI mismatch: the OpenVINO EP plugin DLL")
            print(f"was built for the ORT version statically linked inside")
            print(f"onnxruntime-genai-winml, which is a different ORT build than")
            print(f"pip onnxruntime {ort.__version__}.")
            print()
            print(f"To test the EP with the correct ORT runtime, use model-chat.py:")
            print(f"  python model-chat.py -m <model_path> --use_winml --providers {ep_name}")
            sys.exit(1)

    # ── Step 6: Session details ────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Step 6: Session details")
    print("=" * 60)
    active_providers = session.get_providers()
    print(f"Active providers: {active_providers}")
    if ep_name in active_providers:
        print(f"  ✓ {ep_name} is active")
    else:
        print(f"  ✗ {ep_name} NOT active (fell back to: {active_providers})")

    print(f"\nInputs ({len(session.get_inputs())}):")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type} {inp.shape}")

    print(f"\nOutputs ({len(session.get_outputs())}):")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type} {out.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
