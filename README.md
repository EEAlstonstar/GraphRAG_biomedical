# Graph Counselor: Unified Reasoning & Dynamic Retrieval

An optimized fork of [Graph Counselor (ACL 2025)](https://arxiv.org/abs/2506.03939) for efficient graph-based question answering, with extended retrieval capabilities and a comprehensive evaluation suite.

![Architecture Overview](assets/main.PNG)

---

## Introduction

This repository builds upon the original **Graph Counselor** framework and introduces three major directions of improvement:

1. **Architecture simplification** — The multi-agent pipeline is refactored into a unified reasoning loop, reducing API overhead and improving context coherence.
2. **Retrieval enhancement** — Static few-shot examples are replaced by a dynamic retrieval system; a new semantic neighbor search tool enables targeted multi-hop reasoning within large neighborhoods.
3. **Evaluation extension** — The evaluation suite is expanded beyond Exact Match to include token-level F1, HITS@k, and reasoning-step efficiency analysis.

The system supports multiple graph domains (biomedical, DBLP, Amazon, GoodReads, Legal, MAPLE) and multiple backend LLMs (GPT, Qwen, Llama, Mistral, Gemma, ERNIE).

---

## Key Features

### Unified Reasoning Architecture

The original Planning Agent and Thought Agent are merged into a single reasoning step. The model generates plan, thought, and action in one forward pass, which reduces LLM API calls by over 60% and eliminates the context fragmentation inherent in cascaded multi-agent designs.

### Dynamic Few-Shot Retrieval

Static, hard-coded examples are replaced by a retrieval-based system:

- A few-shot bank (`few_shot_bank.json`) stores annotated reasoning traces.
- For each incoming question, the top-k semantically similar traces are retrieved using `sentence-transformers/all-mpnet-base-v2` and a FAISS inner-product index.
- Retrieved traces replace static examples in the agent prompt at inference time.

### Semantic Neighbor Search (NeighborSearch)

A new graph interaction tool is introduced alongside the existing four:

| Tool | Signature | Description |
|------|-----------|-------------|
| Retrieve | `Retrieve[keyword]` | Retrieve the most relevant node by keyword |
| Feature | `Feature[Node, feature]` | Read a named feature of a node |
| Degree | `Degree[Node, neighbor_type]` | Count neighbors of a given type |
| Neighbor | `Neighbor[Node, neighbor_type]` | List all neighbors of a given type |
| **NeighborSearch** | `NeighborSearch[Node, neighbor_type, query]` | Semantically search within a node's neighbors and return the top-3 most relevant to the query |

`NeighborSearch` is designed for high-degree nodes (e.g., a Disease node connected to hundreds of Gene nodes). Rather than listing all neighbors and relying on the LLM to filter, the agent retrieves only the most query-relevant ones directly. This reduces scratchpad length and the number of required reasoning steps.

The implementation stores all graph node embeddings at initialization and performs inner-product similarity against the neighbor subset at query time, with no additional indexing overhead per call.

---

## Installation

Dependencies are managed with Conda.

```bash
conda create -n graphcounselor python=3.8.1
conda activate graphcounselor

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pytorch -c nvidia faiss-gpu=1.7.4
conda install -c conda-forge langchain==0.1.0 langchain-core==0.1.7 langchain-community==0.0.9
conda install -c conda-forge openai==1.6.1 scikit-learn==1.3.2 sentence-transformers==2.2.2
conda install -c conda-forge transformers==4.36.2 datasets==2.16.1
conda install jsonlines tiktoken networkx IPython
pip install evaluate absl-py rouge_score
```

Alternatively, use the provided environment file:

```bash
conda env create -f environment.yml
conda activate graphcounselor
```

---

## Quick Start

### 1. Download Data

Download the processed graph data from Google Drive:

[Graph Data (Google Drive)](https://drive.google.com/drive/folders/1DJIgRZ3G-TOf7h0-Xub5_sE4slBUEqy9?usp=share_link)

Place the extracted files under `data/processed_data/`. The expected layout is:

```
data/processed_data/
├── biomedical/
│   ├── graph.json
│   ├── data.json
│   └── few_shot_bank.json
├── dblp/
├── amazon/
└── ...
```

### 2. Run the Agent

```bash
bash scripts/run_Graph-Counselor.sh
```

The script starts the vLLM server and launches the agent loop. Key command-line arguments for `run.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `dblp` | Target graph dataset |
| `--llm_version` | `gpt-3.5-turbo` | LLM backend |
| `--llm_way` | `transformer` | Inference backend (`transformer` or `vllm`) |
| `--reflexion_strategy` | `None` | Reflection mode (`None`, `Reflexion`, `Last_attempt`, `Last_attempt_and_Reflexion`) |
| `--compound_strategy` | `None` | Compound function mode (`None`, `compound`, `plan_compound`, `plan`) |
| `--use_dynamic_fewshot` | `True` | Enable dynamic few-shot retrieval |
| `--dynamic_k` | `3` | Number of few-shot examples to retrieve per question |
| `--max_steps` | `15` | Maximum reasoning steps per question |

### 3. Evaluate Results

**Standard evaluation** (EM, BLEU, ROUGE, LLM-as-judge):

```bash
bash eval.sh
```

**Enhanced evaluation** (EM, Token F1, HITS@k, step efficiency):

```bash
python eval_enhanced.py \
    --result_file results/<model>/<dataset>/output.jsonl \
    --logs_dir    results/<model>/<dataset>/logs \
    --dataset     biomedical \
    --output      results/<model>/<dataset>/report.csv
```

The enhanced script reports:

- **Exact Match (EM)** — strict string equality after normalization
- **Token F1** — token-overlap F1, appropriate for list-valued answers
- **HITS@1 / @3 / @5** — whether the ground truth appears in the top-k predicted items
- **Average reasoning steps** — broken down by correct vs. incorrect predictions
- **NeighborSearch call count** — usage frequency of the new tool per question

Per-question results are written to a CSV file for detailed error analysis.

---

## Project Structure

```
GraphRAG_biomedical/
├── Graph-Counselor/
│   ├── code/
│   │   ├── run.py                          # Main entry point
│   │   ├── GraphAgent.py                   # Base agent (GPT / HuggingFace)
│   │   ├── GraphAgent_vllm.py              # vLLM inference agent
│   │   ├── GraphReflectAgent.py            # Agent with reflexion
│   │   ├── GraphReflectAgent_vllm.py       # vLLM reflexion agent
│   │   ├── GraphAgent_Plan_Reflect_vllm.py # Plan + reflexion agent
│   │   ├── graph_prompts.py                # All prompt templates
│   │   ├── graph_fewshots.py               # Static few-shot examples
│   │   └── tools/
│   │       ├── retriever.py                # Node retriever (FAISS)
│   │       ├── graph_funcs.py              # Graph operations
│   │       └── dynamic_retriever.py        # Dynamic few-shot retriever
│   └── scripts/
│       ├── run_Graph-Counselor.sh
│       └── download.py
├── data/
│   └── processed_data/
│       ├── biomedical/
│       ├── dblp/
│       ├── amazon/
│       ├── goodreads/
│       ├── legal/
│       └── maple/
├── eval_enhanced.py                        # Extended evaluation script
├── eval_Llama.py                           # LLM-as-judge evaluation
├── eval_Qwen.py
├── eval.sh
├── environment.yml
└── README.md
```

---

## Supported Datasets

| Dataset | Node Types | Edge Types | Domain |
|---------|-----------|------------|--------|
| biomedical | 11 (Disease, Gene, Compound, ...) | 24 | Biomedical knowledge graph |
| DBLP | 3 (Paper, Author, Venue) | 4 | Academic citation network |
| Amazon | 2 (Item, Brand) | 5 | Product co-purchase graph |
| GoodReads | 4 (Book, Author, Publisher, Series) | 4 | Book recommendation graph |
| Legal | 4 (Opinion, Cluster, Docket, Court) | 4 | Legal citation network |
| MAPLE | 3 (Paper, Author, Venue) | 4 | Multi-domain academic graph |

---

## Supported Models

| Category | Models |
|----------|--------|
| OpenAI API | `gpt-3.5-turbo`, `gpt-4`, `gpt-3.5-turbo-1106`, `gpt-3.5-turbo-16k` |
| Meta Llama | `Llama-2-13b-chat-hf`, `Meta-Llama-3.1-70B-Instruct` |
| Mistral | `Mixtral-8x7B-Instruct-v0.1`, `Mistral-Nemo-Instruct-2407` |
| Alibaba Qwen | `Qwen2.5-7B-Instruct`, `Qwen2.5-72B-Instruct` |
| Google Gemma | `gemma-2-9b-it` |
| Baidu ERNIE | `ERNIE-Speed-8K`, `ERNIE-Speed-128K`, `ERNIE-Lite-8K`, `ERNIE-Tiny-8K` |

---

## Citation

If you use this repository, please cite the original Graph Counselor paper:

```bibtex
@article{gao2025graphcounselor,
  title={Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy},
  author={Junqi Gao and Xiang Zou and Ying Ai and Dong Li and Yichen Niu and Biqing Qi and Jianxing Liu},
  journal={arXiv preprint arXiv:2506.03939},
  year={2025},
  url={https://arxiv.org/abs/2506.03939}
}
```

---

## Acknowledgements

This project extends the original Graph Counselor (ACL 2025) by Gao et al. We thank the original authors for their foundational work on multi-agent graph exploration.

## Contributors

- YixinLiu
- xinyangsally
