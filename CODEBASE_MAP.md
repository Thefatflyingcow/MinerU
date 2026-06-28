# MinerU — Full Codebase Map

**MinerU** is a high-accuracy document parsing engine (by OpenDataLab) that converts PDF, DOCX, PPTX, XLSX, and images into structured **Markdown / JSON**. It has a **VLM + OCR dual engine** architecture supporting 109 languages. Repo: 183 MB, Python 3.10–3.13, Apache 2.0 + commercial terms.

---

## 1. Top-Level Layout

```
MinerU/
├── mineru/              # Main package (the whole product)
│   ├── backend/         # 4 processing engines: pipeline, vlm, hybrid, office
│   ├── cli/             # 17 files: CLI, FastAPI, Router, Gradio, VLM servers
│   ├── model/           # ML models: layout, ocr, mfr, table, vlm, docx, pptx, xlsx
│   ├── data/            # IO: file/S3/HTTP readers+writers
│   ├── utils/           # 35 support modules (PDF, bbox, table, OCR, config...)
│   ├── resources/       # Packaged assets (gradio css/js, fasttext model, dicts)
│   └── version.py
├── tests/               # Single e2e test + coverage helpers
├── demo/                # demo.py + sample pdfs/office_docs
├── docker/              # compose.yaml + global/ and china/ (10+ accelerator variants)
├── docs/                # MkDocs Material site (en/zh, i18n)
├── projects/            # Archived community gallery (redirects to awesome-mineru)
├── pyproject.toml       # 8 console scripts, optional extras: vlm/vllm/lmdeploy/mlx/s3/pipeline/gradio/core/all
├── mineru.template.json # User config template (S3, latex, llm-aided, models-dir)
└── update_version.py    # CI version bumper (git tag → version.py)
```

---

## 2. The 8 Entry-Point Commands

| Script | Module | What it is |
|---|---|---|
| `mineru` | `cli/client.py:main` | Main CLI — **HTTP client** that spawns or talks to `mineru-api` |
| `mineru-api` | `cli/fast_api.py:main` | FastAPI service (the actual parser) |
| `mineru-router` | `cli/router.py:main` | Load-balancing proxy over N `mineru-api` workers (GPU-pinnable) |
| `mineru-gradio` | `cli/gradio_app.py:main` | Bilingual (zh/en) web UI |
| `mineru-vllm-server` | `cli/vlm_server.py:vllm_server` | vLLM inference server for the VLM |
| `mineru-lmdeploy-server` | `cli/vlm_server.py:lmdeploy_server` | LMDeploy inference server |
| `mineru-openai-server` | `cli/vlm_server.py:openai_server` | Auto-picks vLLM/LMDeploy → OpenAI-compatible endpoint |
| `mineru-models-download` | `cli/models_download.py:download_models` | Downloads pipeline/VLM models from HF/ModelScope |

**Key insight:** `mineru` (client.py) does NOT parse in-process — it submits over HTTP to `mineru-api`. All real parsing lives behind the FastAPI surface.

---

## 3. HTTP API Surface (identical on `mineru-api` and `mineru-router`)

| Method | Path | Status | Purpose |
|---|---|---|---|
| POST | `/file_parse` | 200 | Sync parse → ZIP/JSON + `X-MinerU-Task-*` headers |
| POST | `/tasks` | 202 | Async submit → `{task_id, status_url, result_url, queued_ahead}` |
| GET | `/tasks/{id}` | 200 | Status poll |
| GET | `/tasks/{id}/result` | 200/202/409 | ZIP or JSON result |
| GET | `/health` | 200/503 | Health + queue stats (+ `servers[]` on router) |

Request = multipart form with `files` + 17 form fields (`backend, effort, parse_method, lang_list, formula_enable, table_enable, image_analysis, server_url, start/end_page_id, return_* flags...`). Defined once in `cli/api_request.py` and shared by both services.

---

## 4. The 4 Processing Backends (`mineru/backend/`)

The central orchestrator is **`do_parse` / `aio_do_parse`** at `mineru/cli/common.py:668` / `:760`. It normalizes the backend name, peels off office docs first, then dispatches:

### 4.1 `pipeline/` — Traditional small-model OCR pipeline
Layout detector → formula recognizer → OCR det+rec → table recognition → paragraph split. Best for CPU/low-VRAM and scanned docs.
- Entry: `doc_analyze_streaming` (`pipeline/pipeline_analyze.py:157`)
- Models used: PP-DocLayoutV2, UniMERNet/PP-FormulaNet, PytorchPaddleOCR, SLANet-Plus (wireless tables), UNet (wired tables)
- Own `MagicModel`, own `para_split`, own `union_make` serializer

### 4.2 `vlm/` — Pure Vision-Language-Model parsing
Sends page images to the MinerU2.5-Pro VLM which directly returns structured blocks. Supports transformers/vllm/lmdeploy/mlx/http-client engines.
- Entry: `doc_analyze` / `aio_doc_analyze` (`vlm/vlm_analyze.py:423` / `:523`)
- `ModelSingleton` (`:43`) caches `MinerUClient` (from external `mineru_vl_utils`) per backend
- Calls `predictor.batch_two_step_extract` (layout-then-content)

### 4.3 `hybrid/` — Pipeline layout + VLM content (the DEFAULT backend)
Combines pipeline small models (layout + formula + OCR-det) with VLM content extraction. Two effort levels:
- **`medium`** (default, fast): pipeline layout bboxes fed to VLM via `batch_extract_with_layout` (VLM skips its own layout)
- **`high`**: VLM does full `batch_two_step_extract`, pipeline adds inline-formula + OCR-det sidecars
- Entry: `doc_analyze` / `aio_doc_analyze` (`hybrid/hybrid_analyze.py:889` / `:1097`)
- **Reuses VLM's `union_make`** for serialization; reuses both VLM's `ModelSingleton` and pipeline's `HybridModelSingleton`

### 4.4 `office/` — Native DOCX/PPTX/XLSX parsing (no ML)
Reads Office Open XML directly via python-docx/python-pptx/openpyxl/mammoth. Runs first in `do_parse`.
- Entry: `office_docx_analyze` / `office_pptx_analyze` / `office_xlsx_analyze` (`office/{docx,pptx,xlsx}_analyze.py:11`)
- Writes `para_blocks` directly (no finalize step); own `union_make` in `office/mkcontent/`
- Handles equations (OMML → LaTeX via `oMath2Latex`), charts (OOXML → HTML/SVG), hyperlinks, rich text, TOC

### Backend relationship
```
                       do_parse / aio_do_parse (cli/common.py:668/760)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  _process_office_doc        _process_pipeline       _process_vlm / _process_hybrid
  (always first, by suffix)  (backend=="pipeline")   (backend startswith vlm-/hybrid-)
        │                           │                           │
        ▼                           ▼                           ▼
  office.*_analyze        pipeline.doc_analyze_     vlm.doc_analyze  OR  hybrid.doc_analyze
                           streaming                 
        │                           │                           │
        └───────────┬───────────────┴───────────────────────────┘
                    ▼  (all produce a middle_json with _backend tag)
          _process_output (common.py:259) → selects union_make by process_mode
            pipeline → pipeline_middle_json_mkcontent.union_make
            vlm/hybrid → vlm_middle_json_mkcontent.union_make
            docx/pptx/xlsx → office_middle_json_mkcontent.union_make
                    ▼
          writes: *.md, *_content_list.json, *_content_list_v2.json,
                  *_middle.json, *_model.json, *_layout.pdf, *_span.pdf
```

### The `middle_json` intermediate representation (canonical across all backends)
```python
{
  "pdf_info": [ {preproc_blocks, para_blocks, discarded_blocks, page_idx, page_size}, ... ],
  "_backend": "pipeline" | "vlm" | "hybrid" | "office",
  "_version_name": <version>,
  # hybrid adds: "_effort", "_ocr_enable"
}
```
`preproc_blocks` → (finalize step) → `para_blocks` → `union_make` → Markdown/content_list.

---

## 5. The Model Layer (`mineru/model/`)

| Category | Key class (file:line) | Model / Engine | Runtime |
|---|---|---|---|
| **layout** | `PPDocLayoutV2LayoutModel` (`layout/pp_doclayoutv2.py:888`) | PP-DocLayoutV2 (RT-DETR + HGNetV2 + reading-order head), 25 classes | torch + transformers |
| **ocr** | `PytorchPaddleOCR` (`ocr/pytorch_paddle.py:50`) | PP-OCRv6/v5/v3 (PyTorch port) + seal OCR | torch (via `utils/pytorchocr`) |
| **mfr (default)** | `UnimernetModel` (`mfr/unimernet/Unimernet.py:26`) | UniMERNet (Swin + mBART) → LaTeX | torch + transformers |
| **mfr (alt)** | `FormulaRecognizer` (`mfr/pp_formulanet_plus_m/predict_formula.py:23`) | PP-FormulaNet-Plus-M | torch (via `pytorchocr`) |
| **table cls** | `PaddleTableClsModel` (`table/cls/paddle_table_cls.py:16`) | PP-LCNet wired/wireless classifier | onnxruntime |
| **table orient** | `MineruTableOrientationClsModel` (`table/cls/mineru_table_ori_cls.py:25`) | OCR-score heuristic (0/90/270°) | — |
| **table wireless** | `PaddleTableModel` (`table/rec/slanet_plus/main.py:153`) | SLANet-Plus → HTML | onnxruntime |
| **table wired** | `UnetTableModel` (`table/rec/unet_table/main.py:267`) | UNet structure → HTML (falls back to wireless if better) | onnxruntime |
| **vlm** | wrappers (`vlm/vllm_server.py`, `lmdeploy_server.py`) | MinerU2.5-Pro-2605-1.2B (Qwen2VL family) | vllm/lmdeploy/transformers/mlx/http |
| **docx** | `DocxConverter` (`docx/docx_converter.py:43`) | parser (no ML) | python-docx + mammoth + lxml |
| **pptx** | `PptxConverter` (`pptx/pptx_converter.py:88`) | parser (no ML) + XY-cut reading order | python-pptx + lxml |
| **xlsx** | `XlsxConverter` (`xlsx/xlsx_converter.py:167`) | parser (no ML) + flood-fill table detection | openpyxl + zipfile |
| **utils/pytorchocr** | `BaseOCRV20`, `BaseModel`, `TextSystem` | Shared OCR/MFR torch infrastructure | torch |

- **Model registry**: `mineru/utils/enum_class.py:96` (`ModelPath`) — all HF/ModelScope repo + relative paths
- **Model download**: `mineru/utils/models_download_utils.auto_download_and_get_model_root_path` — from `OpenDataLab/PDF-Extract-Kit-1.0` (ModelScope) or `opendatalab/PDF-Extract-Kit-1.0` (HF); VLM from `opendatalab/MinerU2.5-Pro-2605-1.2B`
- **MFR backend toggle**: env `MINERU_FORMULA_CH_SUPPORT` switches unimernet ↔ pp_formulanet
- **Office equations**: OMML → LaTeX via `oMath2Latex` (`model/docx/tools/math/omml.py:200`), shared by docx/pptx/xlsx

---

## 6. The Data/IO Layer (`mineru/data/`)

- **`data_reader_writer/`** (public, range-aware): `FileBasedDataReader/Writer`, `MultiBucketS3DataReader/Writer`, `S3DataReader/Writer`, `DummyDataWriter`. S3 is lazy-loaded (needs `mineru[s3]`).
- **`io/`** (transport primitives): `HttpReader/Writer`, `S3Reader/Writer` (with `Range=` byte-range reads), `IOReader/IOWriter` ABCs.
- **`data/utils/`**: `S3Config` & `PageInfo` pydantic schemas, `parse_s3path`/`parse_s3_range_params`, exceptions (`FileNotExisted`, `InvalidConfig`, `InvalidParams`, `EmptyData`, `CUDA_NOT_AVAILABLE`).
- Input sources: **local file** (default), **S3** (`s3://` paths), **HTTP** (via `io/`), **base64** (content-level, not transport).

---

## 7. The Utils Layer (`mineru/utils/` — 35 files, grouped)

| Group | Key modules |
|---|---|
| **PDF processing** | `pdf_reader` (pypdfium2→PIL), `pdf_text_tool` (char/line extraction + dedup), `pdf_image_tools` (multiprocess render pool + crops), `pdf_classify` (txt vs ocr heuristic, 815 lines), `pdf_page_id`, `pdfium_guard` (thread-safe pdfium + broken-page repair) |
| **Layout/bbox** | `bbox_utils`, `boxbase` (geometry primitives: IoU, distance, overlap), `draw_bbox` (debug visualizations), `cut_image`, `span_block_fix`, `span_pre_proc` (span→char attribution, 22.7 KB) |
| **Table** | `table_merge` (cross-page table merging, 41.5 KB), `table_continuation` (续/continued markers) |
| **OCR/language** | `ocr_utils` (513 lines, det pre/post-processing), `ocr_language` (lang normalization), `language` (fasttext detection), `char_utils` (hyphen/full-width), `guess_suffix_or_lang` (Magika + OOXML inspection) |
| **Config/env** | `config_reader` (`~/mineru.json` + env toggles + `get_device`), `os_env_config`, `check_sys_env`, `enum_class` (`BlockType`, `ContentType`, `ContentTypeV2`, `MakeMode`, `ModelPath`), `cli_parser` |
| **Model utils** | `model_utils` (layout result manipulation + VRAM care), `models_download_utils` (source resolution + caching), `magic_model_utils` & `visual_magic_model_utils` (visual block regrouping) |
| **Office** | `docx_formatting` (pydantic style schema), `office_rich_text` (run + hyperlink formatting) |
| **Other** | `hash_utils` (MD5/SHA), `llm_aided` (LLM-driven title hierarchy via OpenAI client), `title_level_postprocess` (finalize dispatcher), `engine_utils` (`get_vlm_engine` auto-selection by OS) |

---

## 8. Shared Backend Utilities (`mineru/backend/utils/`)

| File | Used by | Purpose |
|---|---|---|
| `para_block_utils.py` | vlm, hybrid | `build_para_blocks_from_preproc`, `merge_para_text_blocks` (cross-page merge) |
| `runtime_utils.py` | all 3 PDF backends | `cross_page_table_merge` (gated by `MINERU_TABLE_MERGE_ENABLE`) |
| `html_image_utils.py` | all 4 backends | `replace_inline_table_images`, `save_base64_image` |
| `formula_number.py` | pipeline, hybrid | formula-number optimization |
| `markdown_utils.py` | vlm, pipeline, office | markdown escaping |
| `ocr_det_utils.py` | office (pptx charts) | OCR fallback |
| `office_chart.py` | office (pptx/xlsx) | OOXML chart → HTML/SVG (989 lines) |
| `office_image.py` | office | WMF/EMF vector image handling |

---

## 9. Infrastructure

### Tests (`tests/`)
- **One e2e test**: `tests/unittest/test_e2e.py::test_pipeline_with_two_config` — runs the pipeline twice (txt + ocr modes) on `tests/unittest/pdfs/test.pdf`, asserts image/table/equation/text content via `fuzzywuzzy` + `BeautifulSoup`.
- Coverage: `pytest --cov=mineru --cov-report html`; `get_coverage.py` enforces ≥0.2% floor; `clean_coverage.py` wipes `htmlcov/`.

### Demo (`demo/`)
- `demo/demo.py` — async HTTP API client demo: auto-spawns `mineru-api`, submits `demo/pdfs/` + `demo/office_docs/`, extracts to `demo/api_output/`.

### Docker (`docker/`)
- `compose.yaml` — 4 profile-gated services (`openai-server`, `api`, `router`, `gradio`), all GPU-enabled.
- `global/Dockerfile` — Nvidia vLLM base, HF models.
- `china/` — 10 accelerator variants (Nvidia, Ascend NPU, Cambricon MLU, Hygon DCU, Enflame GCU, Kunlun XPU, METAX MACA, MooreThreads MUSA, Iluvatar Corex, T-Head PPU) using ModelScope + China mirrors.

### Docs (`docs/` + `mkdocs.yml`)
- MkDocs Material, i18n (en/zh), covers quick_start, usage, CLI params, reference, FAQ, demo.
- Chinese-only: 13 acceleration-card guides + 11 plugin integration guides (incl. Dify, RAGFlow, FastGPT, n8n, Cherry Studio, Coze, etc.).

### CI/CD (`.github/workflows/`)
- `python-package.yml` — release pipeline (tag `*released` → bump version → build wheels for Py 3.10–3.13 → GitHub Release + PyPI).
- `cli.yml` — test CI (push to master/dev → `coverage run` e2e → coverage gate, 240 min timeout).
- `mkdocs.yml` — docs deploy to GitHub Pages.
- `cla.yml` — CLA Assistant; `rerun.yml` — auto-retry failed CI up to 3×.

---

## 10. How Everything Connects (end-to-end flow)

```
User → mineru CLI / Gradio / HTTP client
         │ (HTTP multipart)
         ▼
   mineru-api (fast_api.py) ──or──► mineru-router ──► N × mineru-api workers (GPU-pinned)
         │
         ▼ run_parse_job (fast_api.py:822)
   do_parse / aio_do_parse (common.py:668/760)
         │
         ├─ Office file? → office backend → model/{docx,pptx,xlsx} converters (no ML)
         │
         └─ PDF/image? → select backend:
              ├─ pipeline → model/layout + model/ocr + model/mfr + model/table (small models)
              ├─ vlm      → MinerUClient (mineru_vl_utils) → MinerU2.5-Pro VLM
              └─ hybrid   → pipeline layout/MFR/OCR + VLM content (medium/high effort)
         │
         ▼ (each produces middle_json)
   _process_output → union_make → *.md, *_content_list.json, *_middle.json, *_model.json, *.pdf
         │
         ▼ (returned as ZIP or JSON)
   Client extracts results (optionally regenerates client-side, runs visualization)
```

**Config backbone**: `~/mineru.json` (read by `config_reader.py`) + env vars (`MINERU_*`) + `mineru.template.json` template. Device selection: `cuda/mps/npu/gcu/musa/mlu/sdaa/cpu` via `MINERU_DEVICE_MODE`. Model source: `auto`/`huggingface`/`modelscope` via `MINERU_MODEL_SOURCE`.
