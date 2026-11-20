# Quantized Retail Recommender

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/27fuk5hk7v/badge.svg)](https://ask.thili.ai/b/27fuk5hk7v)

This project develops a smart retail recommender using a language model trained on shopping basket data to predict next items. It then focuses on shrinking the model's footprint using quantization techniques, making it suitable for edge device deployment or faster loading. The goal is a smaller, functional recommendation model with minimal accuracy loss.

## What you are building

**Goal.** You will build a smart retail recommender that identifies frequently bought items and then apply quantization to shrink its model size for efficient deployment.

**Done when.** The project is done when you have a functional recommendation model, trained on shopping basket data for next-item prediction, and you have successfully applied post-training quantization to reduce its size while evaluating the impact on accuracy.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare synthetic shopping basket data for training**  
      Hint: Use the data preparation techniques from 'Teach a Language Model What Sells Together' to format your shopping basket sequences.
- [x] **2. Continue pretraining a small LM on the basket data**  
      Hint: Apply the fine-tuning methods from 'Teach a Language Model What Sells Together' to adapt your model.
- [x] **3. Implement next-item prediction and itemset probing**  
      Hint: Use the inference and analysis patterns from 'Teach a Language Model What Sells Together' to generate recommendations.
- [x] **4. Measure the baseline size and accuracy of the model**  
      Hint: Before quantization, record the model's original size and evaluate its performance metrics.
- [x] **5. Apply dynamic and static post-training quantization to the model**  
      Hint: Implement the quantization steps from 'Shrink a Neural Network with Quantisation' to reduce model precision.
- [x] **6. Demonstrate size reduction and evaluate accuracy impact**  
      Hint: Compare the size of the quantized model to the original and re-evaluate its accuracy using techniques from 'Shrink a Neural Network with Quantisation'.

## Run it

You need Python 3.11 or newer (verified locally on 3.11; CI targets 3.12). No git install is required.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional: only app/llm.py reads it; the recommender does not need it
streamlit run app/main.py
pytest
```

The six milestone implementations live in `app/core.py` and their tests in `tests/test_core.py`.
Dataset generation and training are seeded: the same seed and parameters produce the same
`data/dataset.json` and `data/model.pt`, including in separate processes.

### Using the demo

In the sidebar, **Generate dataset**, then **Train model + measure baseline**. Then pick a basket and press
**Recommend next items**, press **Quantize models**, and press **Compare all models** to see the size and accuracy
table and charts. Every number comes from the matching milestone function on the current data and model; page
reruns reuse the trained model instead of retraining.

`data/dataset.json`, `data/model.pt`, `data/model_dynamic.pt`, `data/model_static.pt`, `data/baseline.json`,
`data/quantization.json` and `data/comparison.json` are generated artifacts and are ignored by git. Replacing the
dataset disables the old model in the UI until you train again.

## Note on Milestone 2

The milestone hint points at the continued-pretraining lesson from 'Teach a Language Model What Sells
Together', whose reference implementation fine-tunes SmolLM2-135M (and is MLX/Apple-only there).
This project pins no external checkpoint: the brief asks for a tiny model that trains quickly on a
CPU without downloads. `milestone_2` therefore trains the small causal Transformer **from scratch**
on the basket corpus with the same next-token objective. It is a genuine autoregressive language
model, but it is not continued pretraining of a pretrained checkpoint.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different quantization bit-widths or methods (e.g., per-tensor vs. per-channel) to observe their trade-offs.
- Benchmark the inference speed of the original and quantized models to quantify performance improvements.
- Explore deploying the quantized model to a simulated edge environment to test its real-world feasibility.
