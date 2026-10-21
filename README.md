# NeuroRAG-FT: Python RAG Research Assistant

A modular mini-project prototype for document-grounded question answering, with a Streamlit UI, persistent ChromaDB vector index, Sentence Transformers embeddings, source/page provenance, retrieval evaluation, and an optional LoRA training scaffold.

## Important status notes

- The default generator is **mock mode**. It confirms that the UI and retrieval flow are wired, but it does not generate a real answer.
- For real local generation, configure Ollama and select the `ollama` backend in the UI.
- The LoRA script is optional and is not automatically called by the app. Fine-tuning requires a carefully prepared dataset and a compatible model/runtime.
- Scanned PDFs may need OCR. Tables, equations, and multi-column layouts can extract incorrectly.
- The provided evaluation utilities measure retrieval against manually annotated chunk IDs. They do not establish answer correctness or citation entailment.
- No experimental performance claims are included. Run the system and record actual results before writing the report.

## Setup

Python 3.10 or 3.11 is a reasonable starting point; check package compatibility for your platform.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The first run may download embedding-model weights. Index a permitted PDF, TXT, or Markdown file in the Documents tab, then ask a question.

### Real local generation with Ollama

1. Install Ollama from https://ollama.com/.
2. Start the Ollama service.
3. Pull a model, for example `ollama pull llama3.1` (hardware requirements vary).
4. In the app sidebar, select backend `ollama` and enter the model name.
5. Test a small query and inspect the returned sources.

## Evaluation

Create a JSONL file with one object per line:

```json
{"question":"What method is used?","relevant_chunk_ids":["actual_chunk_id_1"]}
```

The `relevant_chunk_ids` must be annotated from the indexed corpus. Placeholder IDs in `sample_eval.jsonl` must be replaced. The Evaluation tab reports Recall@K and MRR, which evaluate retrieval only.

For a stronger study, compare:
1. Base model without retrieval.
2. Base model with RAG.
3. Fine-tuned model without retrieval.
4. Fine-tuned model with RAG.

Use the same held-out questions, decoding settings, and scoring rubric. Score factual correctness, citation support, abstention, and latency manually or with a documented evaluation protocol. Do not claim a system is better until results support that claim.

## Optional LoRA training scaffold

Prepare a JSONL file with `{"text":"..."}` records. Each training record should be a high-quality prompt/answer example or a clearly defined causal-LM example. Do not train on the test set.

```bash
python finetune_lora.py --model gpt2 --train-file train.jsonl --output-dir ./adapter-output
```

This uses GPT-2 as a compact teaching example. Other models may require different LoRA target module names, more memory, and a different runtime. Inspect the model architecture before training. The script does not automatically integrate the adapter into the app.

## Tests

```bash
pytest -q
```

Tests cover chunking validation, metric calculations, JSONL validation, and prompt source labels. They do not test end-to-end model quality.

## Project structure

- `app.py`: Streamlit user interface
- `rag_pipeline.py`: ingestion, chunking, embedding, ChromaDB, retrieval, prompt, generator interface
- `evaluation.py`: Recall@K, MRR, JSONL loading
- `finetune_lora.py`: optional PEFT/LoRA training scaffold
- `tests/`: core unit tests

## Research/report checklist

- Cite the course policy, assigned paper, and all external papers/tools actually used.
- Verify paper-specific claims against the source PDF.
- Report dataset/corpus size, chunk size/overlap, embedding model/version, top-k, generator/model version, and hardware.
- Include measured results, uncertainty where possible, and failure examples.
- Discuss privacy, document permissions, prompt injection, hallucinations, OCR/table extraction, and reproducibility.
