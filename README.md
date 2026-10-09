# concept-surgery

**Surgical concept erasure for transformer language models: a measurement pipeline and a GPT-2 case study.**

I wanted to know whether a concept can be removed from a language model by editing its activations instead of retraining it. This repo is the tool I built to test that, plus the result of running it on GPT-2 with one concept, Mickey Mouse.

The idea is simple. Find a direction `v` in the residual stream that carries the concept, then project it out at inference time:

```
h_erased = h - (h · v) v
```

No weights are edited and nothing is retrained. Removing the hook restores the original model exactly.

![Figure 1. The pipeline: map, fit, select, ablate, prove.](assets/figures/pipeline.png)

*Figure 1. The pipeline: map activations, fit directions, select the best layer and estimator, install the ablation hook, then measure.*

## What I found

On GPT-2 with a Mickey Mouse prompt set, the automatically selected LEACE-derived eraser at layer 5 drops the layer-5 probe accuracy from **0.975 to 0.058**. Neutral-text NLL rises by 0.240 nats (4.293 to 4.533) and locality KL is 0.061. That is a large local effect at a modest cost.

It does not last. Probes at deeper layers recover most of the signal (0.669 at layer 6, 0.875 at layer 8, 0.942 at layer 10 against a 0.951 baseline). Ablating several layers at once does not prevent this and costs more fluency (neutral NLL 5.158). My reading is that when a concept is named in the prompt, later layers can rebuild it from the token identities, but I have not isolated that mechanism. What the experiment supports is **local suppression of a linear signal, not forgetting**.

![Figure 2. Probe accuracy by layer after a single-layer and a multi-layer edit.](assets/figures/probe_by_layer.png)

*Figure 2. Probe accuracy by layer. The edit at layer 5 collapses the probe there, and the signal returns in deeper layers.*

![Figure 3. Headline probe collapse and neutral-text cost.](assets/figures/headline_results.png)

*Figure 3. Left: probe accuracy at the edited layer. Right: the neutral-text cost of each edit.*

The saved generations are too weak to settle anything. GPT-2 already hallucinates and repeats itself before the edit, so I do not claim the model forgot the character. The probe and NLL numbers carry the evidence.

### How far to trust this

- One model, one concept, one seed (0), and no confidence intervals.
- The edit was chosen from 18 layer and estimator candidates, so the headline number is optimistically selected. The selection pass measured probe accuracy 0.225 on a subset; the final scorecard on the full set measured 0.058.
- The hook applies an orthogonal projection onto the subspace obtained from the LEACE fit. It does not apply the full oblique LEACE operator, and in the saved run the LEACE and difference-in-means candidates tie to three decimals at every layer. This run does not separate the two estimators.
- Neutral NLL and locality KL are narrow measures of collateral damage. I did not run capability benchmarks.
- A probe accuracy far below 0.5 means this particular probe collapsed. It is not a calibrated probability of forgetting.

## Why this matters

Fixing a model that has absorbed copyrighted text, toxic content, or a bias usually means retraining, which costs weeks of compute and a lot of money. Editing representations is a cheaper third option between doing nothing and retraining. Whether it works is an empirical question, and this repo is built to test it, including where it fails.

## What's inside

| Path | Purpose |
|---|---|
| `concept_surgery/core_math.py` | Pure NumPy math: difference-in-means, PCA, INLP (Ravfogel et al. 2020), a LEACE-style eraser (Belrose et al. 2023), projection and ablation operators |
| `concept_surgery/capture.py` | Activation capture from a Hugging Face causal LM through forward hooks |
| `concept_surgery/ablate.py` | Inference-time ablation hooks (the "surgery") |
| `concept_surgery/data.py` | Prompt sets (`mickey_mouse`, `harry_potter`, `neutral`) and a JSONL loader |
| `concept_surgery/evaluate.py` | Probes, neutral NLL, and KL locality metrics |
| `concept_surgery/pipeline.py` | End to end: map, fit, select, ablate, evaluate |
| `scripts/run_erasure.py` | Command line entry point |
| `concept_surgery/models.py` | Preset model IDs for GPT-2, four Qwen2.5-Coder sizes, StarCoder2-3B, and Phi-2 |
| `scripts/run_model_suite.py` | Sequential, resumable multi-model comparison with a JSON summary |
| `scripts/make_figures.py` | Rebuilds every figure in `assets/figures/` from the saved results |
| `paper/paper.md` | Full write-up with the math, protocol, results, and limitations |
| `paper/submission_draft.md` | Competition-length paper draft with the archived pilot and pending-run caveats |
| `paper/five-page-paper.md` | Short report on success rates and failure modes (PDF alongside) |
| `scripts/build_paper_pdf.py` | Rebuilds a typeset PDF from `paper/submission_draft.md` |
| `paper/results_gpt2_mickey.json` | Raw results of the run described above |
| `notebooks/colab_demo.ipynb` | Colab notebook that runs the pipeline on GPT-2 |

![Figure 4. How the modules connect.](assets/figures/module_dataflow.png)

*Figure 4. Module data flow, from prompt sets to the saved results file.*

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -q      # 7 NumPy math tests, plus 3 integration tests that need torch and transformers

# smallest real run (GPT-2, works on CPU or a free Colab GPU):
python scripts/run_erasure.py --model gpt2 --concept mickey_mouse

# a larger model (Llama weights are gated on Hugging Face; accept the license first):
python scripts/run_erasure.py --model meta-llama/Llama-3.2-1B --concept harry_potter --dtype float16
```

Only the GPT-2 run is reported here. The hook code is written to work with Llama, Mistral, Qwen, and Phi-3 style models, but I have not tested those yet.

The pipeline prints a before and after scorecard:

- `probe_acc`: cross-validated accuracy of a linear probe that decodes the concept from hidden states. Read it layer by layer. A low score at one layer is not evidence of durable forgetting.
- `neutral_nll`: negative log-likelihood on neutral text. It should stay close to the baseline.
- `locality_kl`: KL divergence between baseline and edited next-token distributions on neutral prompts. Smaller means more surgical.
- Side-by-side generations for eyeballing.

To rebuild the figures after a new run, point `paper/results_gpt2_mickey.json` at your results and run `python scripts/make_figures.py`. Numbers can shift a little across library versions, so rerun the GPT-2 experiment on your machine to confirm them.

## Multi-model runs

The suite keeps GPT-2 as a historical control and adds four instruction-tuned
Qwen2.5-Coder sizes, StarCoder2-3B, and Phi-2. Each model runs sequentially, uses the same
prompt sets, seed, candidate-selection rule, and metrics, and writes its own
`results.json`; `suite_results.json` records comparable headline metrics and
any model-loading failures. A tokenizer's chat template is applied
automatically when one is available; GPT-2 continues to use plain prompts.

```bash
python scripts/run_model_suite.py --list-models
python scripts/run_model_suite.py --models gpt2 qwen-coder-0.5b qwen-coder-1.5b qwen-coder-3b starcoder2-3b phi-2 --dtype auto --out runs/model_suite
```

Use `--resume` to reuse completed per-model results after an interrupted run.
Model weights may require several gigabytes of disk and RAM/VRAM; start with
`gpt2` and `qwen-coder-0.5b`, then add larger models if the runtime can hold
them. Qwen2.5-Coder-7B is marked as a high-memory option. StarCoder2-3B is
code-pretrained but not instruction-tuned, so its results are a different
prompting condition, not a direct instruction-model comparison. No cross-model
scores are included in the paper until the models have actually been run.

Build the paper PDF with:

```bash
python scripts/build_paper_pdf.py --output paper/submission_draft.pdf
```

## Design decisions

1. **Probe-verified erasure.** "The model says it doesn't know" is not erasure. I require a linear probe to collapse, and I measure it at every layer, because a single-layer probe overstates the effect.
2. **Automatic layer selection.** Every candidate layer and estimator pair is scored by probe distance from chance plus neutral-NLL cost, and the best is chosen. No hand-picked layers.
3. **Four estimators, one interface.** Difference-in-means, PCA of differences, INLP, and a LEACE-style eraser. PCA is implemented but was not part of the saved run.
4. **Locality measurement.** KL on neutral prompts measures collateral damage as a distribution shift, not just anecdotes.
5. **Tested math.** The core layer is pure NumPy with unit tests for planted-direction recovery, exact covariance annihilation, and preservation of orthogonal content.

## Limitations

This removes a concept's *linear* representation at one point in the forward pass. Superposition, nonlinear encodings, recomputation from the prompt, and fine-tuning attacks can bring the information back. The weights still contain it, and only the forward pass is edited. Treat this as a lens on representation geometry and a baseline for unlearning research, not a copyright or safety compliance guarantee. Details are in `paper/paper.md` section 6.

## Author

Built by [Zoe Faith Gumise](https://github.com/Zoe4370).

## Citation

If you build on this, cite the work the methods come from: Ravfogel et al. (2020), *Null It Out*; Belrose et al. (2023), *LEACE: Perfect linear concept erasure in closed form*; Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*.
