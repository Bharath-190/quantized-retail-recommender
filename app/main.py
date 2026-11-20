"""Streamlit entry point: `streamlit run app/main.py`

A thin UI over the six milestone functions in ``app/core.py``. Nothing is recomputed
unless the matching button is pressed, and no number on the page is hardcoded: every
metric comes from a milestone function call on the current dataset and model.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app import core

st.set_page_config(page_title="Quantized Retail Recommender", page_icon="🛒", layout="wide")


def _dataset_from_disk():
    path = Path(core.DEFAULT_DATASET_PATH)
    return core.load_dataset(str(path)) if path.exists() else None


def _model_from_disk():
    path = Path(core.DEFAULT_MODEL_PATH)
    if not path.exists():
        return None
    model, checkpoint = core.load_model(str(path))
    return {
        "model_path": str(path),
        "config": checkpoint["config"],
        "vocab": checkpoint["vocab"],
        "item2id": checkpoint["item2id"],
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "loss_history": None,          # not stored in the checkpoint, so it is not shown
    }


def _init_state():
    """Pick up existing files once, so a page rerun never retrains anything."""
    for key in ("baseline", "quantized", "comparison", "recommendations", "load_error"):
        st.session_state.setdefault(key, None)
    if "dataset" not in st.session_state:
        try:
            st.session_state.dataset = _dataset_from_disk()
        except Exception as error:
            st.session_state.dataset = None
            st.session_state.load_error = f"Could not read the saved dataset: {error}"
    if "trained" not in st.session_state:
        try:
            st.session_state.trained = (None if st.session_state.get("model_stale")
                                        else _model_from_disk())
        except Exception as error:
            st.session_state.trained = None
            st.session_state.load_error = f"Could not read the saved model: {error}"


def _inference_dataset(trained: dict) -> dict:
    """The vocab/config view that ``recommend`` needs, taken from the checkpoint."""
    return {"vocab": trained["vocab"], "item2id": trained["item2id"],
            "max_len": trained["config"]["max_len"]}


def _format(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _comparison_frame(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Model": row["label"],
        "Status": row["status"],
        "Size (KB)": None if row["model_size_bytes"] is None
        else round(row["model_size_bytes"] / 1024, 1),
        "Saved vs original (%)": None if row["size_reduction_percent"] is None
        else round(row["size_reduction_percent"], 1),
        "Top-1 accuracy": row["top1_accuracy"],
        "Top-3 accuracy": row["top3_accuracy"],
        "MRR": row["mrr"],
        "Δ Top-1": row["top1_delta"],
        "ms / prediction": None if row["avg_inference_ms"] is None
        else round(row["avg_inference_ms"], 3),
        "Speedup vs original": None if row["inference_speedup"] is None
        else round(row["inference_speedup"], 2),
    } for row in results])


_init_state()
dataset = st.session_state.dataset
trained = st.session_state.trained

st.title("🛒 Quantized Retail Recommender")
st.write(
    "A tiny Transformer language model learns which products sell together from synthetic "
    "shopping baskets. It recommends the next item for your basket, and post-training "
    "quantization shrinks it — with the size and accuracy impact measured, not assumed."
)
if st.session_state.get("load_error"):
    st.warning(f"{st.session_state.load_error} Regenerate the dataset or retrain to replace it.")

# ----------------------------------------------------------------------------------
# Sidebar: data and training
# ----------------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Dataset")
    n_baskets = st.slider("Baskets", 500, 5000, 2500, step=500)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=0, step=1)
    if st.button("Generate dataset", width="stretch"):
        try:
            with st.spinner("Generating baskets, vocabulary and train/eval split…"):
                fresh = core.milestone_1(n_baskets=int(n_baskets), seed=int(seed))
            st.session_state.dataset = fresh
            for key in ("trained", "baseline", "quantized", "comparison", "recommendations"):
                st.session_state[key] = None
            st.session_state.model_stale = True
            st.success(f"Generated {fresh['n_baskets']} baskets "
                       f"({fresh['stats']['n_train']} train / {fresh['stats']['n_eval']} eval).")
            st.caption("The dataset changed, so the old model is no longer used. Retrain below.")
        except Exception as error:
            st.error(f"Dataset generation failed: {error}")

    st.divider()
    st.header("2 · Training")
    epochs = st.slider("Epochs", 1, 40, 20)
    if st.button("Train model + measure baseline", width="stretch"):
        if dataset is None:
            st.error("Generate the dataset first.")
        else:
            try:
                with st.spinner(f"Training for {epochs} epochs on CPU…"):
                    info = core.milestone_2(dataset=dataset, epochs=int(epochs), seed=int(seed))
                with st.spinner("Measuring baseline size, accuracy and latency…"):
                    baseline = core.milestone_4(dataset=dataset, model_path=info["model_path"])
                st.session_state.trained = info
                st.session_state.baseline = baseline
                st.session_state.quantized = None
                st.session_state.comparison = None
                st.session_state.model_stale = False
                st.success(f"Trained {info['n_params']:,} parameters. "
                           f"Final training loss {info['final_loss']:.3f}.")
            except Exception as error:
                st.error(f"Training failed: {error}")

    st.divider()
    st.caption("Data files live in `data/` and are reused across restarts. "
               "Nothing is retrained unless you press the training button.")

# ----------------------------------------------------------------------------------
# Dataset overview
# ----------------------------------------------------------------------------------
st.subheader("Dataset")
if dataset is None:
    st.info("No dataset yet — use **Generate dataset** in the sidebar.")
else:
    stats = dataset["stats"]
    columns = st.columns(4)
    columns[0].metric("Baskets", dataset["n_baskets"])
    columns[1].metric("Training baskets", stats["n_train"])
    columns[2].metric("Evaluation baskets", stats["n_eval"])
    columns[3].metric("Products", stats["n_products"])
    with st.expander("Catalogue, co-purchase groups and sample baskets"):
        st.write(f"Vocabulary of {stats['vocab_size']} tokens "
                 f"(3 special tokens + {stats['n_products']} products), "
                 f"mean basket size {stats['mean_basket_items']:.1f} items.")
        catalogue = pd.DataFrame(
            [(department, product) for department, products in dataset["catalog"].items()
             for product in products],
            columns=["Department", "Product"])
        st.dataframe(catalogue, hide_index=True, width="stretch", height=220)
        st.write("Planted co-purchase groups (the pattern the model should learn):")
        st.write(" · ".join(" / ".join(theme) for theme in dataset["themes"]))
        st.write("Sample training sequences:")
        for tokens in dataset["train"][:3]:
            items = core.decode_sequence(dataset, tokens)
            st.code(f"<bos> {'; '.join(items)} <eos>", language=None)

# ----------------------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------------------
st.subheader("Next-item recommendations")
if trained is None:
    st.info("No trained model yet — train one in the sidebar.")
else:
    st.caption(f"Model ready: {trained['n_params']:,} parameters, "
               f"vocab {trained['config']['vocab_size']} tokens.")
    products = sorted(trained["item2id"], key=trained["item2id"].get)
    default = [name for name in ("Milk 1L", "Breakfast Cereal") if name in trained["item2id"]]
    basket = st.multiselect("Your basket", products, default=default,
                            help="Pick the products already in the basket. Empty is fine too.")
    k = st.slider("How many suggestions", 1, 10, 5)
    if st.button("Recommend next items"):
        try:
            with st.spinner("Scoring every product with the model…"):
                model, _ = core.load_model(trained["model_path"])
                recommendations = core.recommend(model, _inference_dataset(trained), basket, k=k)
            st.session_state.recommendations = recommendations
        except Exception as error:
            st.error(f"Recommendation failed: {error}")

    recommendations = st.session_state.recommendations
    if recommendations is not None:
        if not recommendations:
            st.warning("The basket already contains every product, so there is nothing to suggest.")
        else:
            frame = pd.DataFrame([{"Product": row["product"],
                                   "Probability": round(row["probability"], 4)}
                                  for row in recommendations])
            st.dataframe(frame, hide_index=True, width="stretch")
            st.bar_chart(frame.set_index("Product")["Probability"])

# ----------------------------------------------------------------------------------
# Baseline metrics
# ----------------------------------------------------------------------------------
st.subheader("Baseline (float32) metrics")
baseline = st.session_state.baseline
if baseline is None:
    st.info("Baseline metrics appear after training.")
else:
    columns = st.columns(4)
    columns[0].metric("Top-1 accuracy", _format(baseline["top1_accuracy"]))
    columns[1].metric("Top-3 accuracy", _format(baseline["top3_accuracy"]))
    columns[2].metric("MRR", _format(baseline["mrr"]))
    columns[3].metric("Size", f"{baseline['model_size_bytes'] / 1024:.1f} KB")
    columns = st.columns(4)
    columns[0].metric("Perplexity", _format(baseline["perplexity"], 2))
    columns[1].metric("ms / prediction", _format(baseline["avg_inference_ms"]))
    columns[2].metric("Parameters", f"{baseline['n_parameters']:,}")
    columns[3].metric("Evaluation baskets", baseline["n_eval_used"])
    st.caption(f"Measured on the {baseline['n_eval_used']} held-out baskets "
               f"(the last product of each basket is hidden and ranked). "
               f"Itemset probe: hit@3 {_format(baseline['probing']['hit@k'])}, "
               f"mean rank {_format(baseline['probing']['mean_rank'], 2)} over "
               f"{baseline['probing']['n_itemsets']} planted pairs.")

# ----------------------------------------------------------------------------------
# Quantization + comparison
# ----------------------------------------------------------------------------------
st.subheader("Quantization and comparison")
if trained is None or dataset is None:
    st.info("Train a model first; quantization needs the dataset and the trained model.")
else:
    st.caption("Dynamic int8 quantizes Linear weights; static int8 also quantizes "
               "activations using observers calibrated on training baskets only.")
    if st.button("Quantize models"):
        try:
            with st.spinner("Quantizing (dynamic and static) and re-evaluating…"):
                quantized = core.milestone_5(dataset=dataset,
                                             model_path=trained["model_path"])
            st.session_state.quantized = quantized
        except Exception as error:
            st.error(f"Quantization failed: {error}")

    quantized = st.session_state.quantized
    if quantized is not None:
        for method, label in (("dynamic", "Dynamic int8"), ("static", "Static int8")):
            result = quantized[method]
            if result["status"] == "ok":
                st.success(f"{label}: {result['quantized_layers']} quantized Linear layers, "
                           f"{result['model_size_bytes'] / 1024:.1f} KB, top-1 "
                           f"{_format(result['metrics']['top1_accuracy'])}.")
                if result.get("limitation"):
                    st.caption(result["limitation"])
            else:
                st.warning(f"{label} is unavailable: {result['error']}")
                if result.get("limitation"):
                    st.caption(result["limitation"])

    if st.button("Compare all models"):
        try:
            with st.spinner("Quantizing, benchmarking and comparing on the held-out baskets…"):
                comparison = core.milestone_6(dataset=dataset,
                                              model_path=trained["model_path"])
            st.session_state.comparison = comparison
        except Exception as error:
            st.error(f"Comparison failed: {error}")

comparison = st.session_state.comparison
if comparison is not None:
    st.write(f"Comparison on {comparison['n_eval_used']} held-out baskets "
             f"({comparison['benchmark_repetitions']} timed predictions per model).")
    st.dataframe(_comparison_frame(comparison["results"]), hide_index=True,
                 width="stretch")

    available = [row for row in comparison["results"] if row["status"] == "ok"]
    left, right = st.columns(2)
    with left:
        st.caption("Model size in KB")
        sizes = pd.DataFrame({"Model": [row["label"] for row in available],
                              "Size (KB)": [row["model_size_bytes"] / 1024
                                            for row in available]}).set_index("Model")
        st.bar_chart(sizes)
    with right:
        st.caption("Accuracy on held-out baskets")
        accuracy = pd.DataFrame({label: [row[key] for row in available]
                                 for key, label in (("top1_accuracy", "Top-1 accuracy"),
                                                    ("top3_accuracy", "Top-3 accuracy"))},
                                index=[row["label"] for row in available])
        st.bar_chart(accuracy)

    reductions = [row for row in available if row["method"] != "original"]
    if reductions:
        st.caption("Size saved compared with the original model (%)")
        saved = pd.DataFrame({"Model": [row["label"] for row in reductions],
                              "Saved (%)": [row["size_reduction_percent"]
                                            for row in reductions]}).set_index("Model")
        st.bar_chart(saved)

    for row in comparison["results"]:
        if row["status"] != "ok":
            st.warning(f"{row['label']}: quantization did not succeed — {row['error']}")
    st.caption(f"Smallest model: {comparison['summary']['smallest_model']}. "
               f"Best top-1 accuracy: {comparison['summary']['best_top1_accuracy']}. "
               f"Quantization succeeded for: "
               f"{', '.join(comparison['summary']['quantization_succeeded']) or 'none'}.")
