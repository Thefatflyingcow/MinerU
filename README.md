# MinerU — Low Memory Fork

This is a fork of [MinerU](https://github.com/opendatalab/MinerU), a document parsing tool that converts PDF, DOCX, PPTX, XLSX, and images into Markdown and JSON using VLM + OCR engines. Go check out the original project for full documentation — this README only covers what this fork changes.

## What This Fork Does

I made MinerU use less RAM. The original requires 16GB minimum, 32GB recommended. On my M1 Pro MacBook with 16GB unified memory, that's tight — macOS eats 4-5GB, leaving ~10GB for MinerU, and the VLM model alone is 2.4GB in BF16.

So I built an adaptive memory optimization system that auto-detects your hardware at startup and tunes everything to fit. The goal was simple: **make it not OOM on a 16GB Mac**. I don't really have a use case for it — it just seemed like a fun project.

## What Changed

Six optimizations, all auto-detected based on available memory:

**1. VLM INT8 Quantization** — The MinerU2.5-Pro VLM (a fine-tuned Qwen2-VL-2B) loads at 8-bit instead of 16-bit. Works across MLX (macOS), transformers (bitsandbytes), and vLLM backends. Cuts VLM memory roughly in half.

**2. Adaptive Processing Window** — Originally holds 64 pages in memory at once. On low-memory systems, drops to 2-8 pages. Less image data sitting in RAM.

**3. Adaptive PDF DPI** — Renders pages at 144 DPI instead of 200 on low-memory systems. 48% less image data per page. The original code had `144` commented out right next to `200` — they clearly considered it.

**4. LRU Model Eviction** — MinerU caches all its pipeline models (layout, OCR, formula, table) forever. On constrained systems, the singleton now evicts the least-recently-used model when loading a new one, then calls `clean_memory()`.

**5. Adaptive Backend Routing** — If you're on a low-memory system using the hybrid backend and you feed it a text-only PDF, it routes to the pipeline backend instead (no VLM needed). Uses the existing `pdf_classify()` function that was already in the codebase.

**6. Lower GPU Memory Utilization** — vLLM/MLX grabs 50% of VRAM for KV cache by default. On unified memory, that's half your RAM. Dropped to 30% on low-memory systems.

## Testing (Be Honest Here)

This is a proof-of-concept. Here's what actually happened:

**What I tested:**
- Memory profiling and auto-config detection — works, correctly identifies M1 Pro 16GB
- Pipeline parsing of synthetic PDFs (generated with reportlab) — 4 test PDFs, all completed
- Peak memory comparison on a 50-page PDF — 1,376MB optimized vs 1,775MB unoptimized (22% less)
- Image RAM at 144 vs 200 DPI — 45.9MB vs 88.5MB for 8 pages (48% less)
- All 7 environment variable overrides — verified

**What I didn't test:**
- VLM INT8 quantization end-to-end — hit a transformers/MLX version conflict in the test environment. The code paths exist and pass unit tests, but I never ran actual VLM parsing with quantization enabled
- Real-world PDFs — everything was synthetic. No scanned documents, no academic papers, no handwriting
- Actual quality comparison — the "0.06% accuracy drop" number is from Qwen2-VL-2B's published INT8 benchmarks, not from running MinerU2.5-Pro. The real impact on parsing quality is unknown
- Speed — INT8 can be faster or slower depending on backend. The smaller window size makes it slower (more batches). Not properly benchmarked

**Bottom line:** The infrastructure works and the memory savings are real for the pipeline backend. The VLM quantization is code-complete but unvalidated. Use at your own risk.

## How to Use

Everything is automatic. Just install and run MinerU as normal — it detects your memory and configures itself. If you want to override:

| Env Var | Values | What it does |
|---|---|---|
| `MINERU_VLM_QUANTIZATION` | `int8`, `int4`, `none` | VLM quantization level |
| `MINERU_PROCESSING_WINDOW_SIZE` | `1`-`64` | Pages held in memory |
| `MINERU_PDF_IMAGE_DPI` | `72`-`200` | PDF render resolution |
| `MINERU_VLM_BATCH_SIZE` | `1`-`8` | VLM inference batch size |
| `MINERU_MODEL_EVICTION` | `true`, `false` | Enable LRU model eviction |
| `MINERU_MODEL_EVICTION_BUDGET_GB` | float | Memory budget for models |
| `MINERU_GPU_MEMORY_UTILIZATION` | `0.1`-`0.95` | GPU memory fraction |
| `MINERU_ADAPTIVE_BACKEND` | `true`, `false` | Route text PDFs to pipeline |

To disable all optimizations and get original behavior:
```bash
export MINERU_VLM_QUANTIZATION=none
export MINERU_PROCESSING_WINDOW_SIZE=64
export MINERU_PDF_IMAGE_DPI=200
export MINERU_MODEL_EVICTION=false
export MINERU_ADAPTIVE_BACKEND=false
```

## Files Changed

| File | What |
|---|---|
| `mineru/utils/memory_profiler.py` | New — detects system memory, chip type, unified memory |
| `mineru/utils/memory_config.py` | New — resolves optimal settings from memory profile |
| `mineru/utils/config_reader.py` | Adaptive processing window size |
| `mineru/utils/pdf_image_tools.py` | Adaptive DPI |
| `mineru/utils/pdf_reader.py` | Uses adaptive DPI as default |
| `mineru/backend/vlm/vlm_analyze.py` | INT8 quantization for MLX, transformers, vLLM |
| `mineru/backend/vlm/utils.py` | Adaptive GPU memory utilization and batch size |
| `mineru/backend/pipeline/model_init.py` | LRU model eviction in AtomModelSingleton |
| `mineru/cli/common.py` | Adaptive backend selection in do_parse |

## Credits

- Original project: [opendatalab/MinerU](https://github.com/opendatalab/MinerU)
- VLM quantization research: [RedHat VLM quantization](https://developers.redhat.com/articles/2025/04/01/enable-faster-vision-language-models-quantization), [Qwen2-VL INT8 benchmarks](https://openlm.ai/qwen2-vl)
- MLX quantization: [Apple WWDC25 MLX session](https://developer.apple.com/videos/play/wwdc2025/298)
