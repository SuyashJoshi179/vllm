# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVTX instrumentation plugin for the speculative-decoding-under-TP profiling
harness (Task 9).

This module patches a handful of hot-path methods at runtime to emit NVTX
ranges so that an Nsight Systems trace can be attributed to decode phases:

    VERIFY        - GPUModelRunner.execute_model, decode-only target forward
    PREFILL_STEP  - GPUModelRunner.execute_model, when the scheduler_output
                    carries prefill work for at least one request
    SAMPLE        - GPUModelRunner.sample_tokens
    DRAFT         - each speculative proposer's propose()

DRAFT ranges nest inside SAMPLE (drafting happens during sample_tokens) and
both are siblings of VERIFY/PREFILL_STEP. VERIFY does NOT contain DRAFT.

No existing vLLM source file is modified; everything here is monkeypatching
applied from the `vllm.general_plugins` entry point (see register() below).
"""

import functools
import importlib
import os

import torch

_WRAPPED_ATTR = "_specdec_nvtx_wrapped"
_RPC_INSTALLED_ATTR = "_specdec_nvtx_rpcs_installed"

# Proposer classes that expose a `propose()` method to wrap with DRAFT.
# SpecDecodeBaseProposer is the shared base for the LLM-based drafters
# (EAGLE/MTP, incl. Nemotron); subclasses that do not override propose()
# inherit the wrapped implementation for free. Step3p5MTPProposer overrides
# propose() itself, so it needs its own entry.
_DRAFT_TARGETS = (
    ("vllm.v1.spec_decode.ngram_proposer", "NgramProposer"),
    ("vllm.v1.spec_decode.medusa", "MedusaProposer"),
    ("vllm.v1.spec_decode.llm_base_proposer", "SpecDecodeBaseProposer"),
    ("vllm.v1.spec_decode.ngram_proposer_gpu", "NgramProposerGPU"),
    ("vllm.v1.spec_decode.suffix_decoding", "SuffixDecodingProposer"),
    ("vllm.v1.spec_decode.step3p5", "Step3p5MTPProposer"),
)

# Module-level snapshot of the most recent register() call, so report() can
# be called standalone (e.g. via the worker RPC) without re-running patching.
_state: dict = {"patched": [], "errors": []}


def _has_prefill(scheduler_output) -> bool:
    """True if `scheduler_output` schedules prefill work for any request.

    A decode-step request has exactly `1 + len(draft_token_ids)` scheduled
    tokens: the previous step's sampled/bonus token plus however many draft
    tokens are being verified. vLLM itself uses this identity internally
    (see gpu_model_runner.py's `num_scheduled_tokens[req_idx] == draft_len
    + 1` check) to distinguish decode requests from chunked-prefill ones.
    Anything scheduling more tokens than that for a request is prefill
    (first prompt chunk or a chunked-prefill continuation).
    """
    try:
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
            draft_len = len(spec_tokens.get(req_id, ()))
            if num_tokens > draft_len + 1:
                return True
        return False
    except AttributeError:
        # Structure changed under us; fail safe to VERIFY rather than guess.
        return False


def _make_execute_model_wrapper(orig):
    @functools.wraps(orig)
    def wrapper(self, scheduler_output, *args, **kwargs):
        range_name = "PREFILL_STEP" if _has_prefill(scheduler_output) else "VERIFY"
        torch.cuda.nvtx.range_push(range_name)
        try:
            return orig(self, scheduler_output, *args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    setattr(wrapper, _WRAPPED_ATTR, True)
    return wrapper


def _make_static_range_wrapper(orig, range_name: str):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        torch.cuda.nvtx.range_push(range_name)
        try:
            return orig(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    setattr(wrapper, _WRAPPED_ATTR, True)
    return wrapper


def _patch_method(cls, method_name, make_wrapper, label, patched, errors) -> bool:
    """Patch `cls.method_name` in place using `make_wrapper(orig) -> new_fn`.

    Idempotent: looks at cls.__dict__ directly (not inherited attributes) so
    a subclass that doesn't override the method is left alone -- patching
    the base class already covers it, and this avoids double-wrapping the
    same underlying function object through two different class names.
    """
    try:
        orig = cls.__dict__.get(method_name)
        if orig is None:
            if hasattr(cls, method_name):
                # Inherited; the base class patch (if any) already covers it.
                return False
            errors.append(f"{label}: no attribute {method_name!r}")
            return False
        if getattr(orig, _WRAPPED_ATTR, False):
            patched.append(f"{label} (already patched)")
            return True
        setattr(cls, method_name, make_wrapper(orig))
        patched.append(label)
        return True
    except Exception as exc:  # noqa: BLE001 - report, don't crash the caller
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        return False


def install_worker_rpcs(worker_cls) -> None:
    """Attach collective_rpc-callable helpers to the vLLM Worker class.

    - start_nsys_capture() / stop_nsys_capture(): bracket the Nsight Systems
      capture window from the harness driver process.
    - specdec_nvtx_report(): lets the harness confirm, per rank, that
      instrumentation actually landed.
    """
    if getattr(worker_cls, _RPC_INSTALLED_ATTR, False):
        return

    def start_nsys_capture(self) -> None:
        torch.cuda.profiler.start()

    def stop_nsys_capture(self) -> None:
        torch.cuda.profiler.stop()

    def specdec_nvtx_report(self) -> dict:
        return report()

    worker_cls.start_nsys_capture = start_nsys_capture
    worker_cls.stop_nsys_capture = stop_nsys_capture
    worker_cls.specdec_nvtx_report = specdec_nvtx_report
    setattr(worker_cls, _RPC_INSTALLED_ATTR, True)


def report() -> dict:
    """Return the current instrumentation status for this process."""
    patched = list(_state["patched"])
    errors = list(_state["errors"])
    ok = any(p.startswith("DRAFT:") for p in patched)
    return {
        "pid": os.getpid(),
        "patched": patched,
        "errors": errors,
        "ok": ok,
    }


def register() -> dict:
    """Entry point for the `vllm.general_plugins` group."""
    patched: list = []
    errors: list = []

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _patch_method(
        GPUModelRunner,
        "execute_model",
        _make_execute_model_wrapper,
        "VERIFY/PREFILL_STEP:GPUModelRunner.execute_model",
        patched,
        errors,
    )
    _patch_method(
        GPUModelRunner,
        "sample_tokens",
        lambda orig: _make_static_range_wrapper(orig, "SAMPLE"),
        "SAMPLE:GPUModelRunner.sample_tokens",
        patched,
        errors,
    )

    for module_name, class_name in _DRAFT_TARGETS:
        label = f"DRAFT:{class_name}.propose"
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: import failed: {type(exc).__name__}: {exc}")
            continue
        _patch_method(
            cls,
            "propose",
            lambda orig: _make_static_range_wrapper(orig, "DRAFT"),
            label,
            patched,
            errors,
        )

    try:
        from vllm.v1.worker.gpu_worker import Worker

        install_worker_rpcs(Worker)
        patched.append("RPC:Worker.{start_nsys_capture,stop_nsys_capture,specdec_nvtx_report}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"install_worker_rpcs: {type(exc).__name__}: {exc}")

    _state["patched"] = patched
    _state["errors"] = errors

    result = report()

    require_draft = os.environ.get("SPECDEC_NVTX_REQUIRE_DRAFT", "1") != "0"
    if require_draft and not result["ok"]:
        raise RuntimeError(
            "specdec_nvtx: no DRAFT range was installed on any proposer "
            "(set SPECDEC_NVTX_REQUIRE_DRAFT=0 to allow speculation-off "
            f"runs). Report: {result}"
        )

    return result
