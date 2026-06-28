# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import atexit
import gc
import os
import time
import json
import threading
from contextlib import asynccontextmanager, contextmanager

import pypdfium2 as pdfium
from loguru import logger
from tqdm import tqdm

from .utils import enable_custom_logits_processors, set_default_gpu_memory_utilization, set_default_batch_size, \
    set_lmdeploy_backend, mod_kwargs_by_device_type
from .model_output_to_middle_json import (
    append_page_blocks_to_middle_json,
    finalize_middle_json,
    init_middle_json,
)
from mineru.backend.utils.runtime_utils import exclude_progress_bar_idle_time
from ...data.data_reader_writer import DataWriter
from mineru.utils.pdf_image_tools import (
    aio_load_images_from_pdf_bytes_range,
    load_images_from_pdf_doc,
)
from ...utils.check_sys_env import is_mac_os_version_supported
from ...utils.config_reader import get_device, get_processing_window_size

from ...utils.enum_class import ImageType
from ...utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)
from ...utils.models_download_utils import auto_download_and_get_model_root_path

from mineru_vl_utils import MinerUClient
from packaging import version


def _get_vlm_quantization() -> str:
    try:
        from mineru.utils.memory_config import get_memory_optimization_config
        config = get_memory_optimization_config()
        return config.quantization
    except Exception:
        return ""


def _load_mlx_model_quantized(model_path: str, bits: int = 8):
    try:
        from mlx_vlm import load as mlx_load
        from mlx_vlm.utils import quantize_model
        from mlx_vlm.prompt import simple_prompt
    except ImportError:
        logger.warning("mlx-vlm not available for quantized loading, falling back to default loader")
        from mineru_vl_utils.mlx_compat import load_mlx_model
        return load_mlx_model(model_path)

    try:
        from pathlib import Path
        model_path_obj = Path(model_path) if not isinstance(model_path, Path) else model_path

        quantized_dir = model_path_obj.parent / f"{model_path_obj.name}_int{bits}"
        if quantized_dir.exists():
            logger.info(f"Found pre-quantized MLX model at {quantized_dir}")
            model, processor = mlx_load(str(quantized_dir))
            return model, processor

        logger.info(f"Quantizing MLX model to {bits}-bit on-the-fly (this may take a moment)...")
        model, processor, _ = mlx_load(str(model_path_obj))
        model, config = quantize_model(model, None, bits, group_size=64)

        try:
            from mlx_vlm.utils import save_model
            save_model(str(quantized_dir), model, processor, config)
            logger.info(f"Saved quantized model to {quantized_dir} for future use")
            model, processor = mlx_load(str(quantized_dir))
        except Exception as save_err:
            logger.debug(f"Could not save quantized model (using in-memory): {save_err}")

        return model, processor
    except Exception as e:
        logger.warning(f"MLX quantization failed ({e}), falling back to default loader")
        from mineru_vl_utils.mlx_compat import load_mlx_model
        return load_mlx_model(model_path)


class ModelSingleton:
    _instance = None
    _models = {}
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        backend: str,
        model_path: str | None,
        server_url: str | None,
        **kwargs,
    ) -> MinerUClient:
        key = (backend, model_path, server_url)
        with self._lock:
            if key not in self._models:
                start_time = time.time()
                model = None
                processor = None
                vllm_llm = None
                lmdeploy_engine = None
                vllm_async_llm = None
                batch_size = kwargs.get("batch_size", 0)  # for transformers backend only
                max_concurrency = kwargs.get("max_concurrency", 100)  # for http-client backend only
                http_timeout = kwargs.get("http_timeout", 600)  # for http-client backend only
                server_headers = kwargs.get("server_headers", None)  # for http-client backend only
                max_retries = kwargs.get("max_retries", 3)  # for http-client backend only
                retry_backoff_factor = kwargs.get("retry_backoff_factor", 0.5)  # for http-client backend only
                # 从kwargs中移除这些参数，避免传递给不相关的初始化函数
                for param in ["batch_size", "max_concurrency", "http_timeout", "server_headers", "max_retries", "retry_backoff_factor"]:
                    if param in kwargs:
                        del kwargs[param]
                if backend not in ["http-client"] and not model_path:
                    model_path = auto_download_and_get_model_root_path("/","vlm")
                if backend == "transformers":
                    try:
                        from transformers import (
                            AutoProcessor,
                            Qwen2VLForConditionalGeneration,
                        )
                        from transformers import __version__ as transformers_version
                    except ImportError:
                        raise ImportError("Please install transformers to use the transformers backend.")

                    if version.parse(transformers_version) >= version.parse("4.56.0"):
                        dtype_key = "dtype"
                    else:
                        dtype_key = "torch_dtype"
                    device = get_device()
                    quantization = _get_vlm_quantization()
                    load_kwargs = {dtype_key: "auto"}
                    if quantization == "int8":
                        logger.info("Loading transformers model with INT8 quantization (bitsandbytes) for memory optimization")
                        try:
                            from transformers import BitsAndBytesConfig
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_skip_modules=["vision", "visual", "merger"],
                            )
                            load_kwargs.pop(dtype_key, None)
                            load_kwargs["device_map"] = "auto"
                        except ImportError:
                            logger.warning("bitsandbytes not available, falling back to full precision. Install with: pip install bitsandbytes")
                    elif quantization == "int4":
                        logger.info("Loading transformers model with INT4 quantization (bitsandbytes nf4) for memory optimization")
                        try:
                            from transformers import BitsAndBytesConfig
                            import torch as _torch
                            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=_torch.bfloat16,
                                llm_int8_skip_modules=["vision", "visual", "merger"],
                            )
                            load_kwargs.pop(dtype_key, None)
                            load_kwargs["device_map"] = "auto"
                        except ImportError:
                            logger.warning("bitsandbytes not available, falling back to full precision. Install with: pip install bitsandbytes")
                    else:
                        load_kwargs["device_map"] = {"": device}
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        **load_kwargs,
                    )
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        use_fast=True,
                    )
                    if batch_size == 0:
                        batch_size = set_default_batch_size()
                elif backend == "mlx-engine":
                    mlx_supported = is_mac_os_version_supported()
                    if not mlx_supported:
                        raise EnvironmentError("mlx-engine backend is only supported on macOS 13.5+ with Apple Silicon.")
                    from mineru_vl_utils.mlx_compat import load_mlx_model
                    quantization = _get_vlm_quantization()
                    if quantization == "int8":
                        logger.info("Loading MLX model with INT8 quantization for memory optimization")
                        model, processor = _load_mlx_model_quantized(model_path, bits=8)
                    elif quantization == "int4":
                        logger.info("Loading MLX model with INT4 quantization for memory optimization")
                        model, processor = _load_mlx_model_quantized(model_path, bits=4)
                    else:
                        model, processor = load_mlx_model(model_path)
                else:
                    if os.getenv('OMP_NUM_THREADS') is None:
                        os.environ["OMP_NUM_THREADS"] = "1"

                    if backend == "vllm-engine":
                        try:
                            import vllm
                        except ImportError:
                            raise ImportError("Please install vllm to use the vllm-engine backend.")

                        kwargs = mod_kwargs_by_device_type(kwargs, vllm_mode="sync_engine")

                        if "compilation_config" in kwargs:
                            if isinstance(kwargs["compilation_config"], str):
                                try:
                                    kwargs["compilation_config"] = json.loads(kwargs["compilation_config"])
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse compilation_config as JSON: {kwargs['compilation_config']}")
                                    del kwargs["compilation_config"]
                        if "gpu_memory_utilization" not in kwargs:
                            kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                        if "model" not in kwargs:
                            kwargs["model"] = model_path
                        vlm_quant = _get_vlm_quantization()
                        if vlm_quant and "quantization" not in kwargs:
                            if vlm_quant == "int8":
                                kwargs["quantization"] = "bitsandbytes"
                                logger.info("Using vLLM bitsandbytes INT8 quantization")
                            elif vlm_quant == "int4":
                                kwargs["quantization"] = "awq"
                                logger.info("Using vLLM AWQ INT4 quantization")
                        if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                            from mineru_vl_utils import MinerULogitsProcessor
                            kwargs["logits_processors"] = [MinerULogitsProcessor]
                        # 使用kwargs为 vllm初始化参数
                        vllm_llm = vllm.LLM(**kwargs)
                    elif backend == "vllm-async-engine":
                        try:
                            from vllm.engine.arg_utils import AsyncEngineArgs
                            from vllm.v1.engine.async_llm import AsyncLLM
                            from vllm.config import CompilationConfig
                        except ImportError:
                            raise ImportError("Please install vllm to use the vllm-async-engine backend.")

                        kwargs = mod_kwargs_by_device_type(kwargs, vllm_mode="async_engine")

                        if "compilation_config" in kwargs:
                            if isinstance(kwargs["compilation_config"], dict):
                                # 如果是字典，转换为 CompilationConfig 对象
                                kwargs["compilation_config"] = CompilationConfig(**kwargs["compilation_config"])
                            elif isinstance(kwargs["compilation_config"], str):
                                # 如果是 JSON 字符串，先解析再转换
                                try:
                                    config_dict = json.loads(kwargs["compilation_config"])
                                    kwargs["compilation_config"] = CompilationConfig(**config_dict)
                                except (json.JSONDecodeError, TypeError) as e:
                                    logger.warning(
                                        f"Failed to parse compilation_config: {kwargs['compilation_config']}, error: {e}")
                                    del kwargs["compilation_config"]
                        if "gpu_memory_utilization" not in kwargs:
                            kwargs["gpu_memory_utilization"] = set_default_gpu_memory_utilization()
                        if "model" not in kwargs:
                            kwargs["model"] = model_path
                        vlm_quant = _get_vlm_quantization()
                        if vlm_quant and "quantization" not in kwargs:
                            if vlm_quant == "int8":
                                kwargs["quantization"] = "bitsandbytes"
                                logger.info("Using vLLM bitsandbytes INT8 quantization (async)")
                            elif vlm_quant == "int4":
                                kwargs["quantization"] = "awq"
                                logger.info("Using vLLM AWQ INT4 quantization (async)")
                        if enable_custom_logits_processors() and ("logits_processors" not in kwargs):
                            from mineru_vl_utils import MinerULogitsProcessor
                            kwargs["logits_processors"] = [MinerULogitsProcessor]
                        # 使用kwargs为 vllm初始化参数
                        vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))
                    elif backend == "lmdeploy-engine":
                        try:
                            from lmdeploy import PytorchEngineConfig, TurbomindEngineConfig
                            from lmdeploy.serve.vl_async_engine import VLAsyncEngine
                        except ImportError:
                            raise ImportError("Please install lmdeploy to use the lmdeploy-engine backend.")
                        if "cache_max_entry_count" not in kwargs:
                            kwargs["cache_max_entry_count"] = 0.5

                        device_type = os.getenv("MINERU_LMDEPLOY_DEVICE", "")
                        if device_type == "":
                            if "lmdeploy_device" in kwargs:
                                device_type = kwargs.pop("lmdeploy_device")
                                if device_type not in ["cuda", "ascend", "maca", "camb"]:
                                    raise ValueError(f"Unsupported lmdeploy device type: {device_type}")
                            else:
                                device_type = "cuda"
                        lm_backend = os.getenv("MINERU_LMDEPLOY_BACKEND", "")
                        if lm_backend == "":
                            if "lmdeploy_backend" in kwargs:
                                lm_backend = kwargs.pop("lmdeploy_backend")
                                if lm_backend not in ["pytorch", "turbomind"]:
                                    raise ValueError(f"Unsupported lmdeploy backend: {lm_backend}")
                            else:
                                lm_backend = set_lmdeploy_backend(device_type)
                        logger.info(f"lmdeploy device is: {device_type}, lmdeploy backend is: {lm_backend}")

                        if lm_backend == "pytorch":
                            kwargs["device_type"] = device_type
                            backend_config = PytorchEngineConfig(**kwargs)
                        elif lm_backend == "turbomind":
                            backend_config = TurbomindEngineConfig(**kwargs)
                        else:
                            raise ValueError(f"Unsupported lmdeploy backend: {lm_backend}")

                        log_level = 'ERROR'
                        from lmdeploy.utils import get_logger
                        lm_logger = get_logger('lmdeploy')
                        lm_logger.setLevel(log_level)
                        if os.getenv('TM_LOG_LEVEL') is None:
                            os.environ['TM_LOG_LEVEL'] = log_level

                        lmdeploy_engine = VLAsyncEngine(
                            model_path,
                            backend=lm_backend,
                            backend_config=backend_config,
                        )
                predictor = MinerUClient(
                    backend=backend,
                    model=model,
                    processor=processor,
                    lmdeploy_engine=lmdeploy_engine,
                    vllm_llm=vllm_llm,
                    vllm_async_llm=vllm_async_llm,
                    server_url=server_url,
                    batch_size=batch_size,
                    max_concurrency=max_concurrency,
                    http_timeout=http_timeout,
                    server_headers=server_headers,
                    max_retries=max_retries,
                    retry_backoff_factor=retry_backoff_factor,
                    enable_table_formula_eq_wrap=True,
                    image_analysis=True,
                    enable_cross_page_table_merge=True,
                )
                predictor._mineru_runtime_handles = {
                    "backend": backend,
                    "model": model,
                    "processor": processor,
                    "vllm_llm": vllm_llm,
                    "vllm_async_llm": vllm_async_llm,
                    "lmdeploy_engine": lmdeploy_engine,
                }
                _maybe_enable_serial_execution(predictor, backend)
                self._models[key] = predictor
                elapsed = round(time.time() - start_time, 2)
                logger.info(f"get {backend} predictor cost: {elapsed}s")
        return self._models[key]

    def shutdown(self) -> None:
        with self._lock:
            predictors = list(self._models.values())
            self._models.clear()

        for predictor in predictors:
            _shutdown_predictor_runtime(predictor)

        gc.collect()


async def _get_model_async(
    backend: str,
    model_path: str | None,
    server_url: str | None,
    **kwargs,
) -> MinerUClient:
    return await asyncio.to_thread(
        ModelSingleton().get_model,
        backend,
        model_path,
        server_url,
        **kwargs,
    )


def _iter_shutdown_candidates(predictor: MinerUClient):
    runtime_handles = getattr(predictor, "_mineru_runtime_handles", {})
    client = getattr(predictor, "client", None)

    seen_ids = set()

    def _yield_candidate(candidate):
        if candidate is None:
            return
        candidate_id = id(candidate)
        if candidate_id in seen_ids:
            return
        seen_ids.add(candidate_id)
        yield candidate

    for key in ("vllm_llm", "vllm_async_llm", "lmdeploy_engine", "model"):
        yield from _yield_candidate(runtime_handles.get(key))

    if client is not None:
        for key in ("vllm_llm", "vllm_async_llm", "lmdeploy_engine", "model"):
            yield from _yield_candidate(getattr(client, key, None))


def _call_nested_shutdown(target, method_path: str, label: str) -> bool:
    current = target
    for attr in method_path.split("."):
        current = getattr(current, attr, None)
        if current is None:
            return False

    if not callable(current):
        return False

    try:
        current()
        logger.debug(f"Shutdown {label} via `{method_path}`")
        return True
    except TypeError:
        logger.debug(f"Skip unsupported shutdown call {label}.{method_path}")
        return False
    except Exception as exc:
        logger.debug(f"Failed to shutdown {label} via `{method_path}`: {exc}")
        return False


def _shutdown_runtime_handle(handle) -> None:
    for method_path in (
        "shutdown",
        "close",
        "stop",
        "terminate",
        "destroy",
        "engine.shutdown",
        "engine.close",
        "engine_core.shutdown",
        "engine_core.close",
        "llm_engine.shutdown",
        "llm_engine.close",
        "llm_engine.model_executor.shutdown",
        "llm_engine.model_executor.close",
        "model_executor.shutdown",
        "model_executor.close",
    ):
        if _call_nested_shutdown(handle, method_path, type(handle).__name__):
            return


def _clear_predictor_references(predictor: MinerUClient) -> None:
    runtime_handles = getattr(predictor, "_mineru_runtime_handles", {})
    for key in tuple(runtime_handles.keys()):
        runtime_handles[key] = None

    client = getattr(predictor, "client", None)
    if client is not None:
        for attr in ("vllm_llm", "vllm_async_llm", "lmdeploy_engine", "model", "processor"):
            if hasattr(client, attr):
                setattr(client, attr, None)


def _shutdown_predictor_runtime(predictor: MinerUClient) -> None:
    for handle in _iter_shutdown_candidates(predictor):
        _shutdown_runtime_handle(handle)
    _clear_predictor_references(predictor)


def shutdown_cached_models() -> None:
    ModelSingleton().shutdown()


atexit.register(shutdown_cached_models)


def _predictor_uses_mlx(predictor: MinerUClient, backend: str | None = None) -> bool:
    if backend == "mlx-engine":
        return True
    client = getattr(predictor, "client", None)
    return type(client).__module__.endswith(".mlx_client")


def _maybe_enable_serial_execution(
    predictor: MinerUClient,
    backend: str | None = None,
) -> MinerUClient:
    if _predictor_uses_mlx(predictor, backend) and not hasattr(
        predictor, "_mineru_execution_lock"
    ):
        predictor._mineru_execution_lock = threading.Lock()
    return predictor


@contextmanager
def predictor_execution_guard(predictor: MinerUClient):
    lock = getattr(predictor, "_mineru_execution_lock", None)
    if lock is None:
        yield
        return
    with lock:
        yield


@asynccontextmanager
async def aio_predictor_execution_guard(predictor: MinerUClient):
    lock = getattr(predictor, "_mineru_execution_lock", None)
    if lock is None:
        yield
        return
    await asyncio.to_thread(lock.acquire)
    try:
        yield
    finally:
        lock.release()


def _close_images(images_list):
    for image_dict in images_list or []:
        pil_img = image_dict.get("img_pil")
        if pil_img is not None:
            try:
                pil_img.close()
            except Exception:
                pass


def doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    client_side_output_generation = bool(
        kwargs.pop("client_side_output_generation", False)
    )
    if predictor is None:
        predictor = ModelSingleton().get_model(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json()
    results = []
    doc_closed = False
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        total_windows = (
            (page_count + effective_window_size - 1) // effective_window_size
            if effective_window_size
            else 0
        )
        logger.info(
            f'VLM processing-window run. page_count={page_count}, '
            f'window_size={configured_window_size}, total_windows={total_windows}'
        )

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = load_images_from_pdf_doc(
                    pdf_doc,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                    pdf_bytes=pdf_bytes,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    logger.info(
                        f'VLM processing window {window_index + 1}/{total_windows}: '
                        f'pages {window_start + 1}-{window_end + 1}/{page_count} '
                        f'({len(images_pil_list)} pages)'
                    )
                    with predictor_execution_guard(predictor):
                        window_results = predictor.batch_two_step_extract(
                            images=images_pil_list,
                            image_analysis=image_analysis,
                        )
                    results.extend(window_results)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_page_blocks_to_middle_json(
                        middle_json,
                        window_results,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_start_index=window_start,
                        progress_bar=progress_bar,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"processing-window infer finished, cost: {infer_time}, "
                f"speed: {round(len(results) / infer_time, 3)} page/s"
            )
        if not client_side_output_generation:
            finalize_middle_json(middle_json["pdf_info"])
        close_pdfium_document(pdf_doc)
        doc_closed = True
        return middle_json, results
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)


async def aio_doc_analyze(
    pdf_bytes,
    image_writer: DataWriter | None,
    predictor: MinerUClient | None = None,
    backend="transformers",
    model_path: str | None = None,
    server_url: str | None = None,
    image_analysis: bool = True,
    **kwargs,
):
    client_side_output_generation = bool(
        kwargs.pop("client_side_output_generation", False)
    )
    if predictor is None:
        predictor = await _get_model_async(backend, model_path, server_url, **kwargs)
    predictor = _maybe_enable_serial_execution(predictor, backend)

    pdf_doc = open_pdfium_document(pdfium.PdfDocument, pdf_bytes)
    middle_json = init_middle_json()
    results = []
    doc_closed = False
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
        configured_window_size = get_processing_window_size(default=64)
        effective_window_size = min(page_count, configured_window_size) if page_count else 0
        total_windows = (
            (page_count + effective_window_size - 1) // effective_window_size
            if effective_window_size
            else 0
        )
        logger.info(
            f'VLM processing-window run. page_count={page_count}, '
            f'window_size={configured_window_size}, total_windows={total_windows}'
        )

        infer_start = time.time()
        progress_bar = None
        last_append_end_time = None
        try:
            for window_index, window_start in enumerate(range(0, page_count, effective_window_size or 1)):
                window_end = min(page_count - 1, window_start + effective_window_size - 1)
                images_list = await aio_load_images_from_pdf_bytes_range(
                    pdf_bytes,
                    start_page_id=window_start,
                    end_page_id=window_end,
                    image_type=ImageType.PIL,
                )
                try:
                    images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
                    logger.info(
                        f'VLM processing window {window_index + 1}/{total_windows}: '
                        f'pages {window_start + 1}-{window_end + 1}/{page_count} '
                        f'({len(images_pil_list)} pages)'
                    )
                    async with aio_predictor_execution_guard(predictor):
                        window_results = await predictor.aio_batch_two_step_extract(
                            images=images_pil_list,
                            image_analysis=image_analysis,
                        )
                    results.extend(window_results)
                    if progress_bar is None:
                        progress_bar = tqdm(total=page_count, desc="Processing pages")
                    else:
                        exclude_progress_bar_idle_time(
                            progress_bar,
                            last_append_end_time,
                            now=time.time(),
                        )
                    append_page_blocks_to_middle_json(
                        middle_json,
                        window_results,
                        images_list,
                        pdf_doc,
                        image_writer,
                        page_start_index=window_start,
                        progress_bar=progress_bar,
                    )
                    last_append_end_time = time.time()
                finally:
                    _close_images(images_list)
        finally:
            if progress_bar is not None:
                progress_bar.close()
        infer_time = round(time.time() - infer_start, 2)
        if infer_time > 0 and page_count > 0:
            logger.debug(
                f"processing-window infer finished, cost: {infer_time}, "
                f"speed: {round(len(results) / infer_time, 3)} page/s"
            )
        if not client_side_output_generation:
            await asyncio.to_thread(finalize_middle_json, middle_json["pdf_info"])
        close_pdfium_document(pdf_doc)
        doc_closed = True
        return middle_json, results
    finally:
        if not doc_closed:
            close_pdfium_document(pdf_doc)
