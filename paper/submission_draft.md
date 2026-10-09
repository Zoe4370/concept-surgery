# When Does Concept Erasure Survive the Forward Pass?

## Extending a GPT-2 study into a reproducible multi-model audit protocol for coding language models

author: Zoe Faith Gumise
date: October 2026
status: Follow-up to my earlier Zenodo paper on GPT-2 concept erasure [1]. The GPT-2 numbers below come from that archived run. The comparative coding-model runs have not been executed, and their results are not inferred.
keywords: concept erasure, activation editing, linear probes, LEACE, coding language models, evaluation protocol

---

### Abstract

Activation-space editing is attractive for developer assistants because it changes behavior without retraining or touching weights. Removing a decodable feature at one layer does not show that a model has forgotten it, and it does not show that an agent's behavior has changed. My earlier paper [1] measured this gap in GPT-2 on a lexically cued concept (Mickey Mouse). This follow-up carries that result forward and extends the code into a multi-model audit runner covering GPT-2, four Qwen2.5-Coder sizes, StarCoder2-3B, and Phi-2. In the archived GPT-2 run, a LEACE-derived orthogonal projection at layer 5 cut cross-validated linear-probe accuracy at that layer from 0.975 to 0.058. Neutral-text negative log-likelihood (NLL) rose from 4.293 to 4.533 nats per token, and the KL divergence between baseline and edited neutral next-token distributions was 0.061. The signal came back downstream: layer-10 probe accuracy was 0.942 after the edit against 0.951 before it. Editing every candidate layer cost more fluency (neutral NLL 5.158) and left the layer-10 probe at 0.958. The run supports local suppression of a linear readout and gives no evidence of forgetting. The comparative coding-model runs are pending, and no claim is made about code generation, repository-level agents, or cross-model generalization.

### 1. Introduction

Software-development assistants mediate code search, editing, and execution, so a cheap intervention that suppresses an unwanted behavior without retraining is worth studying. Activation editing offers one. Find a direction in the residual stream that carries a concept, then project it out at inference time. The edit is reversible, because removing the hook restores the original model exactly.

The measurement problem is that a probe failing at the edited layer is weaker evidence than it looks. Information can persist in other directions, become nonlinear, or be rebuilt by later layers from the prompt itself. Any safety or reliability control placed inside a coding agent inherits this problem, since a suppression that holds at one internal checkpoint is not a system-level guarantee.

This paper continues my earlier study, "Surgical Concept Erasure in GPT-2: A Single Concept Ablation Study with Measured Limits" [1], published on Zenodo. That study reported one GPT-2 run and concluded that a layer-5 edit suppresses a linear probe locally while the signal returns by layer 10. It did not test any other model. The present paper restates those numbers so it can be read alone, and adds three things. I describe a model registry and a sequential runner that applies one protocol to seven causal language models, including coding-specialised ones. I report two details of the archived run that [1] does not discuss, namely the tie between difference-in-means and the LEACE-derived eraser and the layer-by-layer candidate scores. I also specify the held-out design the next experiment needs, because the GPT-2 run fits directions and scores probes on the same prompts. The prompts concern Mickey Mouse rather than code tasks, so this work is methodological and offers no evidence about agentic software engineering.

### 2. Related work

Linear concept erasure removes information that linear predictors can access. INLP projects representations into the nullspace of a sequence of trained classifiers [2]. LEACE gives a closed-form linear eraser for a binary concept label and guarantees that the erased features have zero covariance with the label [3]. Activation interventions also connect to causal editing of factual associations, as in ROME [4], which changes weights, and to the finding that refusal in chat models is mediated by a single residual-stream direction [5]. These methods differ in objective and operator, and an orthogonal projection onto a fitted span is not interchangeable with the full oblique LEACE transformation.

For software engineering, SWE-bench scores issue resolution in real repositories with tests, instead of treating short snippets as the whole task [6]. I do not run SWE-bench or any tool-using agent. My question sits underneath such evaluations: does a representational edit survive the forward pass while ordinary language behavior is measured? A claim about agents needs patch success, tests, and tool-use results on top of internal probes.

### 3. Method

#### 3.1 Pipeline

The pipeline (Figure 1) captures last-token residual-stream activations at candidate layers for a concept prompt set and a neutral prompt set matched in length and register. It fits erasure directions, scores every layer and estimator pair, installs the winner as a forward hook, and evaluates the result at every candidate layer. Candidate layers default to 40% to 90% of model depth. For GPT-2 the saved run scored layers 4, 5, 6, 8, 9, and 10. Three estimators were run: difference-in-means, a LEACE-derived subspace, and INLP. A fourth, PCA of paired differences, is implemented but was not part of the saved run.

![Figure 1. The five pipeline stages. Activations are mapped, directions fitted, the best layer and estimator selected, a reversible hook installed, and the result evaluated at every layer.](assets/figures/pipeline.png){width=100}

#### 3.2 Ablation hook, estimators, and selection

The hook replaces the output of a chosen layer with its projection off the concept subspace (Figure 2), where the rows of *V* are *k* orthonormal concept directions:

$$ H' = H - \left(H V^{\top}\right) V $$ (1)

Weights are untouched. For a binary label *z* with pooled covariance Σ and *b* = Σ<super>+</super> Cov(*x*, *z*), the LEACE eraser is an oblique projection with exactly zero covariance to the label:

$$ P = I - \Sigma\, b\, b^{\top} \,/\, \left(b^{\top}\Sigma\, b\right), \qquad \mathrm{Cov}(Px,\, z) = 0 $$ (2)

In the pipeline the fitted LEACE matrix is converted to a singular-vector subspace, and the shared hook then applies the orthogonal projection of Equation 1 onto that subspace. The saved run is therefore LEACE-derived and does not test the full operator in Equation 2.

![Figure 2. The projection hook sits between two unchanged stretches of the transformer. Removing it restores the original forward pass.](assets/figures/surgery_hook.png){width=78}

Each layer and estimator pair is scored by the distance of its post-edit probe accuracy from chance plus any rise in neutral NLL, and the lowest score is kept:

$$ s(\ell, m) = \left| a_{\ell,m} - 0.5 \right| + \max\left(0,\ \Delta\mathrm{NLL}_{\ell,m}\right) $$ (3)

Here a is five-fold cross-validated probe accuracy after a temporary single-layer edit with estimator *m* at layer *l*.

#### 3.3 Archived GPT-2 run

The run reported in [1] uses GPT-2 with 124M parameters [7] in float32 on CPU, seed 0, with 60 target prompts and 61 neutral prompts. The shipped neutral set has one more prompt than the 60 per class stated in [1]. The probe is a logistic regression scored by five-fold stratified cross-validation on captured activations. The selection pass uses 40 prompts per class for the probe and 20 neutral prompts for NLL. The final scorecard uses all captured activations and 30 neutral prompts for NLL. Locality is the mean KL divergence from baseline to edited next-token distributions on neutral prompts.

Two choices make the reported numbers optimistic. Directions were fitted on the full prompt sets before probe cross-validation, so the folds are not an independent holdout for direction fitting. The chosen edit was also the best of 18 candidates. The measurements are a diagnostic and not an unbiased estimate of generalization. Reference [1] makes the same caveats.

#### 3.4 Multi-model extension

The runner defines seven presets (Table 1). Models run one at a time to bound peak memory. Each writes its own results file, and a suite manifest records headline metrics and load failures. A resume flag reuses completed models after an interruption. Decoder blocks are located for GPT-2, Llama, Qwen, Gemma, Phi, GPT-NeoX, OPT, and MPT layouts. When a tokenizer supplies a chat template, the same user-message wrapper is applied during activation capture, NLL, generation, and locality evaluation. Otherwise prompts stay raw. The repository carries 18 tests: 7 for the NumPy math, 3 integration tests that need torch and transformers, and 8 for the model registry.

Table 1. Model presets in the audit runner. StarCoder2-3B is a base model, so its results are a different prompting condition and not a like-for-like instruction-model comparison.

| Alias | Hugging Face ID | Parameters | Role |
|---|---|---|---|
| gpt2 | gpt2 | 124M | Legacy baseline [7] |
| qwen-coder-0.5b | Qwen/Qwen2.5-Coder-0.5B-Instruct | 0.5B | Instruction-tuned coding model [8] |
| qwen-coder-1.5b | Qwen/Qwen2.5-Coder-1.5B-Instruct | 1.5B | Instruction-tuned coding model [8] |
| qwen-coder-3b | Qwen/Qwen2.5-Coder-3B-Instruct | 3B | Instruction-tuned coding model [8] |
| qwen-coder-7b | Qwen/Qwen2.5-Coder-7B-Instruct | 7B | Instruction-tuned, high memory [8] |
| starcoder2-3b | bigcode/starcoder2-3b | 3B | Code-pretrained base model [9] |
| phi-2 | microsoft/phi-2 | 2.7B | General model with code data [10] |

A valid comparison needs every model loaded and evaluated under the same prompt sets, seed, and selection rule. The report should include parameter count, runtime, and precision, probe accuracy at every layer, neutral NLL and KL, repeated seeds, and failures. The suite summary stores the probe at the last candidate layer, so per-layer curves must be read from each model's own results file. Runs on the current character prompts test architectural variation and say nothing about coding quality. The next experiment should reserve prompts by template family, fit directions on training templates, select on a validation split, and report once on untouched test templates.

### 4. Results

All numbers in this section are read from the archived file paper/results_gpt2_mickey.json and match the values reported in [1]. No pretrained-model run beyond GPT-2 has been completed.

#### 4.1 Candidate selection

Eighteen candidates (six layers, three estimators) were scored (Figure 3, Appendix A). The LEACE-derived eraser at layer 5 won with a composite score of 0.514. INLP at layer 5 scored 0.976, with probe accuracy 0.275 and neutral NLL 5.097. Difference-in-means and the LEACE-derived eraser tie to three decimals at every layer, so this run cannot separate the two estimators, and the choice of LEACE reflects that tie.

The selection pass and the final scorecard disagree on size. Post-edit probe accuracy at layer 5 was 0.225 on the selection subset and 0.058 on the full set. Neutral NLL was 4.586 in selection and 4.533 in the final scorecard. The gap shows how far subset size and the selection step move a headline number.

![Figure 3. Composite selection score for all 18 candidates. Lower is better. Difference-in-means and the LEACE-derived eraser share a bar because they tie at every layer.](assets/figures/selection_candidates.png){width=92}

#### 4.2 Local collapse and downstream recovery

At layer 5, probe accuracy fell from 0.9753 to 0.0577 after the single-layer edit (Table 2, Figure 4). The layer-4 probe stayed at 1.000, as expected for a layer upstream of the hook. Downstream the signal returned: 0.6693 at layer 6, 0.8753 at layer 8, 0.9253 at layer 9, and 0.9423 at layer 10, against a layer-10 baseline of 0.9507.

Table 2. Cross-validated probe accuracy by layer, before and after each edit.

| Condition | L4 | L5 | L6 | L8 | L9 | L10 |
|---|---|---|---|---|---|---|
| Baseline (no edit) | 1.000 | 0.975 | 0.909 | 0.967 | 0.983 | 0.951 |
| Single-layer edit at layer 5 | 1.000 | 0.058 | 0.669 | 0.875 | 0.925 | 0.942 |
| Multi-layer edit | 0.033 | 0.603 | 0.744 | 0.884 | 0.950 | 0.958 |

![Figure 4. Probe accuracy by layer, as in [1]. The single-layer edit collapses the probe at layer 5 and the signal returns in deeper layers. The multi-layer edit does not keep it down.](assets/figures/probe_by_layer.png){width=94}

The multi-layer arm edits every candidate layer with the same method. It still leaves the layer-10 probe at 0.958, above the unedited baseline. An accuracy far below 0.5, such as 0.058, means this probe collapsed on this prompt set. It is not a calibrated probability of forgetting. The pattern fits information remaining available downstream, and it does not identify whether later layers rebuild the signal from token identity or expose another correlated representation.

#### 4.3 Collateral cost

Neutral NLL rose from 4.2932 to 4.5330 nats per token, an increase of 0.2397, and locality KL was 0.0609 (Table 3, Figure 5). The multi-layer arm reached neutral NLL 5.1583, an increase of 0.865, without removing the final-layer signal. Both measures are narrow. Neither is a capability benchmark.

Table 3. Scorecard for the archived GPT-2 run. Probe accuracy is at the last candidate layer (10). The edited-layer value is 0.975 before and 0.058 after.

| Metric | Baseline | Single-layer edit | Multi-layer edit |
|---|---|---|---|
| Probe accuracy, layer 10 | 0.951 | 0.942 | 0.958 |
| Neutral NLL (nats per token) | 4.293 | 4.533 | 5.158 |
| Concept-prompt NLL (nats per token) | 4.407 | 4.798 | 5.359 |
| Locality KL (neutral prompts) | n/a | 0.061 | n/a |

![Figure 5. Left: probe accuracy at the edited layer. Right: neutral-text cost of each edit.](assets/figures/headline_results.png){width=100}

#### 4.4 Generations

Ten prompts were generated before and after the single-layer edit, six about Mickey Mouse and four neutral. Among the six Mickey prompts, the name Mickey appears in three baseline outputs and three edited outputs. Minnie Mouse appears in none of the baseline outputs and in two of the edited ones. The baseline outputs already repeat themselves and invent facts, for example a Mickey Mouse born in 1885 in New York City. I treat no generation as evidence of behavioral forgetting.

#### 4.5 Status of the multi-model comparison

Table 4 lists the comparison the runner will fill. Only the GPT-2 row has data, taken from the archived run. The metrics follow the suite summary, which stores the last-layer probe. Every other row is empty until the model has been run.

Table 4. Cross-model comparison. Pending rows have not been run and contain no estimates.

| Model | Chosen layer, method | Probe L-last (before, after) | Neutral NLL (before, after) | Locality KL |
|---|---|---|---|---|
| gpt2 | 5, LEACE-derived | 0.951, 0.942 | 4.293, 4.533 | 0.061 |
| qwen-coder-0.5b | pending | pending | pending | pending |
| qwen-coder-1.5b | pending | pending | pending | pending |
| qwen-coder-3b | pending | pending | pending | pending |
| qwen-coder-7b | pending | pending | pending | pending |
| starcoder2-3b | pending | pending | pending | pending |
| phi-2 | pending | pending | pending | pending |

### 5. Discussion

A local intervention can make one linear decoder fail while a later decoder keeps working. For a code assistant, an activation edit therefore has to be evaluated at the output and task level and not only where the hook sits. The pending runs vary architecture, scale, and instruction tuning on the same character prompts. They test whether the GPT-2 pattern in [1] depends on model family. They do not test secure-coding decisions, patch quality, or tool use.

Those questions need held-out coding prompts with output-level security checks, task tests, confidence intervals across seeds, and a public notebook with resource and version metadata. An agent should also be tested in its real repository-editing loop, since tool feedback can regenerate a suppressed representation. The code exposes model IDs, dataset names, seeds, layer candidates, selection outputs, and result files so each of these can be audited.

### 6. Limitations

The completed evidence covers one 124M model, one concept, one seed, and a small prompt set. The target is named explicitly in the prompts, so the probe can exploit lexical and stylistic differences. The intervention is linear and local. Direction fitting and probe scoring share data, and the edit was chosen from 18 candidates, which both add optimism. Difference-in-means and LEACE tie in this run, so it says nothing about their relative merit. Probe accuracy, NLL, and next-token KL do not measure semantic unlearning, code correctness, secure behavior, or an agent's ability to recover through tools. No coding-model checkpoint has been run, so Table 4 is a protocol and not evidence.

### 7. Conclusion

In the GPT-2 run from [1], a LEACE-derived layer-5 projection collapsed a linear probe at the edited layer and the probe recovered toward baseline by layer 10. Editing all candidate layers raised fluency cost and left the final-layer signal intact. This is local suppression. Completing the seven-model runs and then the held-out coding-task experiments is the work needed before any statement about developer agents.

### References

[1] Gumise, Z. F. (2026). Surgical Concept Erasure in GPT-2: A Single Concept Ablation Study with Measured Limits. Zenodo.

[2] Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., and Goldberg, Y. (2020). Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. Proceedings of ACL.

[3] Belrose, N., Schneider-Joseph, D., Ravfogel, S., Cotterell, R., Raff, E., and Biderman, S. (2023). LEACE: Perfect linear concept erasure in closed form. NeurIPS.

[4] Meng, K., Bau, D., Andonian, A., and Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. NeurIPS.

[5] Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N. (2024). Refusal in Language Models Is Mediated by a Single Direction. NeurIPS.

[6] Jimenez, C. E. et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR.

[7] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI technical report.

[8] Hui, B. et al. (2024). Qwen2.5-Coder Technical Report. arXiv:2409.12186.

[9] Lozhkov, A. et al. (2024). StarCoder 2 and The Stack v2: The Next Generation. arXiv:2402.19173.

[10] Javaheripi, M. and Bubeck, S. (2023). Phi-2: The surprising power of small language models. Microsoft Research blog.

### Appendix A. All selection candidates

Table A1. Post-edit probe accuracy, neutral NLL, and composite score from the selection pass (subset of prompts). The selected candidate is layer 5 with the LEACE-derived eraser.

| Layer | Estimator | Probe accuracy | Neutral NLL | Score |
|---|---|---|---|---|
| 4 | Difference-in-means | 0.150 | 4.863 | 0.866 |
| 4 | LEACE-derived | 0.150 | 4.863 | 0.866 |
| 4 | INLP | 0.300 | 5.012 | 0.865 |
| 5 | Difference-in-means | 0.225 | 4.586 | 0.514 |
| 5 | LEACE-derived (selected) | 0.225 | 4.586 | 0.514 |
| 5 | INLP | 0.275 | 5.097 | 0.976 |
| 6 | Difference-in-means | 0.175 | 5.170 | 1.148 |
| 6 | LEACE-derived | 0.175 | 5.170 | 1.148 |
| 6 | INLP | 0.275 | 5.037 | 0.915 |
| 8 | Difference-in-means | 0.188 | 5.004 | 0.969 |
| 8 | LEACE-derived | 0.188 | 5.004 | 0.969 |
| 8 | INLP | 0.275 | 5.132 | 1.010 |
| 9 | Difference-in-means | 0.150 | 4.950 | 0.954 |
| 9 | LEACE-derived | 0.150 | 4.950 | 0.954 |
| 9 | INLP | 0.312 | 5.291 | 1.132 |
| 10 | Difference-in-means | 0.200 | 4.890 | 0.843 |
| 10 | LEACE-derived | 0.200 | 4.890 | 0.843 |
| 10 | INLP | 0.338 | 5.131 | 0.947 |

### Appendix B. Code structure

![Figure B1. How the modules connect, from prompt sets to the saved results file.](assets/figures/module_dataflow.png){width=100}
