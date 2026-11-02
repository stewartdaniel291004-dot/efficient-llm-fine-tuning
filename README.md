# Task 9 — Efficient LLM Fine-Tuning Using QLoRA

**Elevvo Pathways NLP Internship · Task 9**

This project fine-tunes a small instruction model for dialogue summarization on SAMSum using 4-bit QLoRA, then compares the base and adapted models with ROUGE.

## Project Setup

| Item | Configuration |
|---|---|
| Dataset | `knkarthick/samsum` (SAMSum) |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Runtime | Google Colab, Tesla T4 (14.56 GiB reported) |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | FP16 |
| Adapter | LoRA, rank 8, alpha 16, dropout 0.05 |
| LoRA target selection | `all-linear` |
| Training subset | 256 conversations |
| Validation subset | 64 conversations |
| Held-out subset prepared | 128 conversations |
| Evaluation subset | 32 conversations |
| Epochs | 1 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | 2e-4 |

The official dataset contains 14,731 training, 818 validation, and 819 test rows. The quick-run experiment uses subsets to fit a practical Colab workflow.

## Results

The executed notebook generated the following baseline and fine-tuned results on the same 32 evaluation conversations:

| Metric | Base Model | QLoRA Fine-Tuned |
|---|---:|---:|
| ROUGE-1 | 0.2933 | **0.4295** |
| ROUGE-2 | 0.0797 | **0.1699** |
| ROUGE-L | 0.2149 | **0.3276** |
| ROUGE-Lsum | 0.2160 | **0.3291** |

All four ROUGE scores were higher for the adapted model in this quick evaluation. These results are preliminary because they are based on 32 examples only; they should not be presented as definitive generalization results.

### Baseline Reproducibility Note

An earlier separately reported baseline was:

- ROUGE-1: 0.2929
- ROUGE-2: 0.0802
- ROUGE-L: 0.2151
- ROUGE-Lsum: 0.2162

The baseline regenerated in the uploaded executed notebook was:

- ROUGE-1: 0.2933
- ROUGE-2: 0.0797
- ROUGE-L: 0.2149
- ROUGE-Lsum: 0.2160

The values are close but not identical. This README uses the baseline regenerated in the same notebook run as the paired comparator. Preserve the earlier `baseline_metrics.json`; do not overwrite it or mix those earlier values into this comparison.

## Training Observations

| Measurement | Recorded Value |
|---|---:|
| Training runtime | 202.78 seconds |
| Training loss | 1.7241 |
| Validation loss | 1.5928 |
| Trainable parameters | 4,399,104 |
| Total model parameters | 498,431,872 |
| Trainable parameter share | 0.8826% |
| Peak allocated GPU memory during training | 2.19 GiB |
| Peak reserved GPU memory during training | 4.24 GiB |
| Base generation time | 1.34 minutes |
| Fine-tuned generation time | 1.4 minutes |

## Method

1. Load and inspect the official SAMSum train, validation, and test splits.
2. Tokenize the chat-formatted instruction and summary target separately.
3. Mask prompt tokens with `-100` so the supervised loss is applied to summary tokens.
4. Load the Qwen2.5-0.5B-Instruct base model in 4-bit NF4.
5. Attach LoRA adapters to `all-linear` layers and train for one epoch.
6. Save the adapter separately from the frozen base model.
7. Reload the quantized base model and saved adapter for inference.
8. Generate summaries using the same test conversations and decoding settings as the base model.
9. Compare ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum.

## Repository Contents

- `Task_9_Efficient_LLM_Fine_Tuning_QLoRA_Final.ipynb` — cleaned, self-contained notebook.
- `README.md` — project overview, method, configuration, and recorded quick-run results.

When the notebook is executed in Colab, it creates:

- `artifacts/qlora_adapter/` — PEFT adapter and tokenizer files.
- `results/base_rouge.json`
- `results/finetuned_rouge.json`
- `results/evaluation_ids.csv`
- `results/evaluation_manifest.json`
- `results/model_comparison_predictions.csv`
- `results/training_log.csv`
- `results/training_log.json`
- `results/rouge_comparison.csv`
- `results/rouge_comparison.json`
- `results/training_loss.png`
- `results/rouge_comparison.png`
- `results/gpu_memory.png`
- `results/summary_lengths.png`

## Run in Google Colab

1. Open the notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the dependency installation cell.
4. Restart the runtime once after installation.
5. Run all notebook cells from the beginning.
6. Download the generated `artifacts/` and `results/` folders for your repository.

The notebook does not include a Hugging Face token because the dataset and model are public. Authentication is optional unless Hub rate limits or private resources require it.

## Limitations

- The quick evaluation uses 32 conversations; full official test-set evaluation is still recommended.
- ROUGE measures lexical overlap and does not fully capture factuality, coherence, or semantic quality.
- A higher quick-run ROUGE score is encouraging but does not establish broad model superiority.
- The adapter contains learned PEFT weights, not a standalone copy of the full base model.

---

**Project:** Efficient LLM Fine-Tuning Using QLoRA  
**Internship:** Elevvo Pathways — NLP Internship  
**Model:** Qwen2.5-0.5B-Instruct  
**Dataset:** SAMSum
