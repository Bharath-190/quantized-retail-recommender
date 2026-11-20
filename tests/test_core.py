"""Tests for the six milestones in README.md, one group per milestone."""
import math
from collections import Counter

import pytest

from app import core

# Milestone 1: Prepare synthetic shopping basket data for training
@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "dataset.json"
    return core.milestone_1(n_baskets=600, seed=7, out_path=str(path))


def test_milestone_1_generates_valid_sequences(dataset):
    assert dataset["n_baskets"] == 600
    sequences = dataset["train"] + dataset["eval"]
    assert len(sequences) == 600
    assert dataset["stats"]["n_train"] == len(dataset["train"])
    assert dataset["stats"]["n_eval"] == len(dataset["eval"])
    vocab_size = len(dataset["vocab"])
    for tokens in sequences:
        assert tokens[0] == core.BOS and tokens[-1] == core.EOS
        assert len(tokens) <= dataset["max_len"]
        assert all(0 <= token < vocab_size for token in tokens)
        products = core.decode_sequence(dataset, tokens)
        assert 2 <= len(products) <= core.MAX_BASKET_ITEMS
        assert len(set(products)) == len(products)          # a basket holds each product once
        assert all(name in dataset["item2id"] for name in products)


def test_milestone_1_vocabulary_maps_products_in_simplest_way(dataset):
    vocab, item2id = dataset["vocab"], dataset["item2id"]
    assert vocab[:3] == ["<pad>", "<bos>", "<eos>"]
    assert len(set(vocab)) == len(vocab)
    assert len(vocab) == len(item2id) + 3
    for name, token in item2id.items():
        assert vocab[token] == name                          # id -> name -> id is the identity
    catalog_products = {name for names in dataset["catalog"].values() for name in names}
    assert set(item2id) == catalog_products
    assert set(dataset["catalog"]) == {"groceries", "snacks", "beverages",
                                       "household", "personal care"}
    for theme in dataset["themes"]:
        assert len(theme) >= 2
        assert all(name in item2id for name in theme)


def test_milestone_1_splits_train_and_eval(dataset):
    n_train, n_eval = dataset["stats"]["n_train"], dataset["stats"]["n_eval"]
    assert n_eval > 0 and n_train > n_eval
    assert n_train + n_eval == dataset["n_baskets"]
    assert 0.85 <= n_train / dataset["n_baskets"] <= 0.95


def test_milestone_1_is_reproducible_and_reusable(tmp_path):
    path = tmp_path / "once.json"
    first = core.milestone_1(n_baskets=120, seed=3, out_path=str(path))
    second = core.milestone_1(n_baskets=120, seed=3, out_path=str(tmp_path / "twice.json"))
    other = core.milestone_1(n_baskets=120, seed=4, out_path=str(tmp_path / "other.json"))
    assert first == second
    assert first["train"] != other["train"]
    assert core.load_dataset(str(path)) == first            # later milestones can reuse the file


def test_milestone_1_is_reproducible_across_processes(tmp_path):
    """The same seed must give byte-identical data even when PYTHONHASHSEED differs.

    Python set iteration order changes with hash randomisation, so a generator that keeps
    its items in a set leaks that order into the RNG and into the saved file.
    """
    import os
    import subprocess
    import sys

    script = ("import sys; from app import core; "
              "core.milestone_1(n_baskets=300, seed=5, out_path=sys.argv[1])")
    outputs = []
    for hash_seed in ("0", "987654"):
        path = tmp_path / f"dataset_{hash_seed}.json"
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        subprocess.run([sys.executable, "-c", script, str(path)], check=True,
                       cwd=core.PROJECT_ROOT, env=environment, capture_output=True)
        outputs.append(path.read_text())
    assert outputs[0] == outputs[1]


def test_milestone_1_has_co_purchase_patterns(dataset):
    baskets = [set(core.decode_sequence(dataset, tokens)) for tokens in dataset["train"]]
    counts = Counter(name for basket in baskets for name in basket)
    n = len(baskets)

    def lift(a, b):
        both = sum(1 for basket in baskets if a in basket and b in basket) / n
        return both / ((counts[a] / n) * (counts[b] / n))

    anchors = [theme[:2] for theme in dataset["themes"]]
    lifts = [lift(a, b) for a, b in anchors]
    assert min(lifts) > 1.3, lifts                          # every errand's pair beats chance
    assert sum(1 for value in lifts if value > 2.0) >= len(anchors) // 2


# Milestone 2: Continue pretraining a small LM on the basket data
@pytest.fixture(scope="module")
def trained(tmp_path_factory, dataset):
    path = tmp_path_factory.mktemp("model") / "model.pt"
    return core.milestone_2(dataset=dataset, epochs=8, batch_size=64, lr=5e-3,
                            seed=0, out_path=str(path))


def test_milestone_2_creates_a_small_causal_model(dataset):
    import torch

    config = core.model_config(len(dataset["vocab"]), dataset["max_len"])
    model = core.create_model(config)
    assert isinstance(model, torch.nn.Module)
    assert 10_000 < sum(p.numel() for p in model.parameters()) < 1_000_000

    tokens = torch.tensor([[core.BOS, 5, 9, core.EOS], [core.BOS, 12, core.EOS, core.PAD]])
    logits = model(tokens)
    assert logits.shape == (2, 4, len(dataset["vocab"]))
    assert torch.isfinite(logits).all()                 # padded positions must not produce NaNs


def test_milestone_2_trains_and_saves(trained):
    import torch

    assert trained["n_train"] == 540
    assert len(trained["loss_history"]) == trained["epochs"] == 8
    assert all(torch.isfinite(torch.tensor(trained["loss_history"])))
    assert trained["final_loss"] < trained["loss_history"][0]   # it actually learned something

    model, checkpoint = core.load_model(trained["model_path"])
    assert model.training is False
    assert checkpoint["config"] == trained["config"]
    assert checkpoint["vocab"] == trained["vocab"]
    assert checkpoint["item2id"] == trained["item2id"]
    assert sum(p.numel() for p in model.parameters()) == trained["n_params"]

    logits = model(torch.tensor([[core.BOS, 5, 9]]))
    assert logits.shape == (1, 3, len(trained["vocab"]))
    assert torch.isfinite(logits).all()
    predicted = int(logits[0, -1].argmax())
    assert 0 <= predicted < len(trained["vocab"])        # a valid next-token prediction


def test_milestone_2_is_causal(trained):
    import torch

    model, _ = core.load_model(trained["model_path"])
    prefix = [core.BOS, 4, 7]
    first = model(torch.tensor([prefix + [11]]))[0]
    second = model(torch.tensor([prefix + [23]]))[0]
    assert torch.allclose(first[:3], second[:3])         # the future cannot change the past
    assert not torch.allclose(first[3], second[3])


def test_milestone_2_is_reproducible(dataset):
    import torch

    first = core.milestone_2(dataset=dataset, epochs=2, batch_size=64, lr=5e-3,
                             seed=11, out_path=None)
    second = core.milestone_2(dataset=dataset, epochs=2, batch_size=64, lr=5e-3,
                              seed=11, out_path=None)
    other = core.milestone_2(dataset=dataset, epochs=2, batch_size=64, lr=5e-3,
                             seed=12, out_path=None)
    assert first["loss_history"] == pytest.approx(second["loss_history"], abs=1e-6)
    assert first["loss_history"] != pytest.approx(other["loss_history"], abs=1e-6)

    model_a, _ = core.train_model(dataset, epochs=2, batch_size=64, lr=5e-3, seed=11)
    model_b, _ = core.train_model(dataset, epochs=2, batch_size=64, lr=5e-3, seed=11)
    for left, right in zip(model_a.state_dict().values(), model_b.state_dict().values()):
        assert torch.allclose(left, right, atol=1e-8)


# Milestone 3: Implement next-item prediction and itemset probing
@pytest.fixture(scope="module")
def recommender(trained):
    model, checkpoint = core.load_model(trained["model_path"])
    return model, checkpoint


def test_milestone_3_recommends_ranked_valid_products(recommender):
    model, checkpoint = recommender
    dataset = {"vocab": checkpoint["vocab"], "item2id": checkpoint["item2id"],
               "max_len": checkpoint["config"]["max_len"]}
    basket = ["Milk 1L", "Breakfast Cereal"]
    recommendations = core.recommend(model, dataset, basket, k=5)

    assert len(recommendations) == 5
    probabilities = [item["probability"] for item in recommendations]
    assert probabilities == sorted(probabilities, reverse=True)   # ranked, best first
    assert all(0.0 < probability <= 1.0 for probability in probabilities)
    assert all(item["product"] in dataset["item2id"] for item in recommendations)
    assert not {item["product"] for item in recommendations} & set(basket)


def test_milestone_3_excludes_products_already_in_the_basket(recommender):
    model, checkpoint = recommender
    dataset = {"vocab": checkpoint["vocab"], "item2id": checkpoint["item2id"],
               "max_len": checkpoint["config"]["max_len"]}
    basket = list(dataset["item2id"])[:10] + ["Milk 1L"]        # includes a duplicate
    recommendations = core.recommend(model, dataset, basket, k=5)
    assert len(recommendations) == 5
    assert not {item["product"] for item in recommendations} & set(basket)

    everything = core.recommend(model, dataset, list(dataset["item2id"]), k=10)
    assert everything == []                                      # nothing left to suggest


def test_milestone_3_handles_empty_unknown_and_long_baskets(recommender):
    model, checkpoint = recommender
    dataset = {"vocab": checkpoint["vocab"], "item2id": checkpoint["item2id"],
               "max_len": checkpoint["config"]["max_len"]}

    empty = core.recommend(model, dataset, [], k=3)              # <bos> only
    assert len(empty) == 3
    assert all(item["product"] in dataset["item2id"] for item in empty)

    with pytest.raises(ValueError):
        core.recommend(model, dataset, ["Unicorn Steak"], k=3)
    with pytest.raises(ValueError):
        core.recommend(model, dataset, ["Milk 1L", "Nope"], k=3)

    long_basket = list(dataset["item2id"])[:20]                  # longer than the context
    assert len(core.recommend(model, dataset, long_basket, k=2)) == 2
    with pytest.raises(ValueError):
        core.recommend(model, dataset, ["Milk 1L"], k=0)


def test_milestone_3_probes_itemsets(recommender, dataset):
    model, _ = recommender
    pair = ["Laundry Detergent", "Fabric Softener"]
    result = core.probe_itemset(model, dataset, pair, k=3)
    assert result["items"] == pair
    assert len(result["ranks"]) == 2
    assert all(1 <= rank <= len(dataset["item2id"]) for rank in result["ranks"])
    assert 0.0 <= result["hit@k"] <= 1.0

    summary = core.probe_itemsets(model, dataset, k=3)
    assert summary["n_itemsets"] == len(dataset["themes"]) == 10
    assert len(summary["per_itemset"]) == 10
    assert 0.0 <= summary["hit@k"] <= 1.0
    assert summary["mean_rank"] < len(dataset["item2id"]) / 2   # better than chance

    with pytest.raises(ValueError):
        core.probe_itemset(model, dataset, ["Milk 1L"], k=3)
    with pytest.raises(ValueError):
        core.probe_itemset(model, dataset, ["Milk 1L", "Nope"], k=3)


def test_milestone_3_is_deterministic(trained, dataset):
    first = core.milestone_3(basket=["Pasta"], k=4, model_path=trained["model_path"],
                             dataset=dataset)
    second = core.milestone_3(basket=["Pasta"], k=4, model_path=trained["model_path"],
                              dataset=dataset)
    assert first == second                                           # eval mode, no sampling
    assert first["basket"] == ["Pasta"]
    assert len(first["recommendations"]) == 4
    assert first["probing"]["n_itemsets"] == 10


# Milestone 4: Measure the baseline size and accuracy of the model
@pytest.fixture(scope="module")
def baseline(tmp_path_factory, trained, dataset):
    path = tmp_path_factory.mktemp("baseline") / "baseline.json"
    return core.milestone_4(dataset=dataset, model_path=trained["model_path"],
                            max_eval=40, benchmark_repetitions=10, out_path=str(path))


def test_milestone_4_reports_baseline_metrics(baseline):
    assert baseline["label"] == "float32 (baseline)"
    assert baseline["n_parameters"] == 78_208
    assert baseline["n_train"] == 540 and baseline["n_eval"] == 60
    assert baseline["n_eval_used"] == 40                     # max_eval capped it

    assert baseline["model_size_bytes"] > 0
    assert baseline["model_size_mb"] == pytest.approx(baseline["model_size_bytes"] / 1e6)

    for name in ("top1_accuracy", "top3_accuracy", "mrr", "mean_probability", "avg_inference_ms"):
        assert isinstance(baseline[name], float)
        assert math.isfinite(baseline[name])
    assert 0.0 <= baseline["top1_accuracy"] <= 1.0
    assert 0.0 <= baseline["top3_accuracy"] <= 1.0
    assert 0.0 < baseline["mrr"] <= 1.0
    assert baseline["top1_accuracy"] <= baseline["top3_accuracy"] <= 1.0
    assert 0.0 < baseline["mean_probability"] < 1.0
    assert baseline["avg_inference_ms"] > 0.0
    assert baseline["mean_nll"] > 0.0 and baseline["perplexity"] > 1.0

    probing = baseline["probing"]
    assert probing["n_itemsets"] == 10
    assert 0.0 <= probing["hit@k"] <= 1.0
    assert 1.0 <= probing["mean_rank"] <= 76              # one rank per catalogue product


def test_milestone_4_evaluates_held_out_data_deterministically(trained, dataset, tmp_path):
    path = tmp_path / "baseline.json"
    first = core.milestone_4(dataset=dataset, model_path=trained["model_path"],
                             max_eval=20, benchmark_repetitions=5, out_path=str(path))
    second = core.milestone_4(dataset=dataset, model_path=trained["model_path"],
                              max_eval=20, benchmark_repetitions=5, out_path=None)
    for key in ("top1_accuracy", "top3_accuracy", "mrr", "mean_nll", "perplexity",
                "model_size_bytes", "n_eval_used"):
        assert first[key] == second[key]                    # no sampling, no dropout

    import json
    assert json.loads(path.read_text()) == first             # reusable by milestone 6

    # Different data must give different numbers: the evaluation really reads the split.
    smaller = {**dataset, "eval": dataset["train"][:20]}
    other = core.milestone_4(dataset=smaller, model_path=trained["model_path"],
                             max_eval=20, benchmark_repetitions=5, out_path=None)
    assert other["n_train"] == len(dataset["train"])
    assert other["top1_accuracy"] != first["top1_accuracy"] or other["mean_nll"] != first["mean_nll"]


# Milestone 5: Apply dynamic and static post-training quantization to the model
@pytest.fixture(scope="module")
def quantized(tmp_path_factory, trained, dataset):
    out_dir = tmp_path_factory.mktemp("quantized")
    return core.milestone_5(dataset=dataset, model_path=trained["model_path"],
                            n_calibration=32, max_eval=40, out_dir=str(out_dir))


def test_milestone_5_dynamic_quantization_works(quantized, dataset):
    import torch

    dynamic = quantized["dynamic"]
    assert dynamic["method"] == "dynamic"
    assert dynamic["status"] == "ok" and dynamic["error"] is None
    assert dynamic["quantized_layers"] == 9                  # 8 block Linears + the head
    assert dynamic["model_size_bytes"] > 0
    assert dynamic["model_size_bytes"] < quantized["baseline"]["model_size_bytes"]
    assert dynamic["metrics"]["n_eval_used"] == 40
    assert 0.0 <= dynamic["metrics"]["top1_accuracy"] <= 1.0

    logits = dynamic["model"](torch.tensor([[core.BOS, 5, 9]]))
    assert logits.shape == (1, 3, len(dataset["vocab"]))
    assert torch.isfinite(logits).all()
    assert any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear)
               for m in dynamic["model"].modules())


def test_milestone_5_static_quantization_is_real_and_honest(quantized, dataset, trained,
                                                            monkeypatch):
    import torch

    static = quantized["static"]
    assert static["method"] == "static"
    assert static["status"] == "ok" and static["error"] is None
    assert static["quantized_layers"] == 9
    assert static["int8_weights"] == 9
    quantized_linears = [module for module in static["model"].modules()
                         if isinstance(module, torch.ao.nn.quantized.Linear)]
    assert len(quantized_linears) == 9
    assert all(module.weight().dtype == torch.qint8 for module in quantized_linears)
    assert static["model_size_bytes"] < quantized["baseline"]["model_size_bytes"]
    assert static["calibrated_on"] == 32                    # training baskets only
    assert "FX graph mode" in static["limitation"]
    assert 0.0 <= static["metrics"]["top1_accuracy"] <= 1.0
    logits = static["model"](torch.tensor([[core.BOS, 5, 9]]))
    assert logits.shape == (1, 3, len(dataset["vocab"]))
    assert torch.isfinite(logits).all()

    # Calibration must not need (or touch) the held-out split.
    model, _ = core.load_model(trained["model_path"])
    train_only = core.quantize_static_model(model, {**dataset, "eval": []}, n_calibration=8)
    assert train_only["status"] == "ok"
    assert train_only["calibrated_on"] == 8

    # A conversion failure must be reported as a failure, never as success.
    def boom(*args, **kwargs):
        raise RuntimeError("conversion unsupported")
    monkeypatch.setattr(core.tq, "convert", boom)
    failed = core.quantize_static_model(model, dataset, n_calibration=8)
    assert failed["status"] == "failed"
    assert failed["model"] is None
    assert "conversion unsupported" in failed["error"]


def test_milestone_5_original_model_is_untouched(trained, dataset):
    import torch

    model, _ = core.load_model(trained["model_path"])
    before = core.evaluate_model(model, dataset, max_examples=20)
    core.quantize_dynamic_model(model)
    core.quantize_static_model(model, dataset, n_calibration=8)
    after = core.evaluate_model(model, dataset, max_examples=20)
    assert before == after
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())


def test_milestone_5_sizes_and_saved_artifacts(quantized):
    import os

    baseline_size = quantized["baseline"]["model_size_bytes"]
    for name in ("dynamic", "static"):
        result = quantized[name]
        assert isinstance(result["model_size_bytes"], int)
        assert 0 < result["model_size_bytes"] < baseline_size
        assert os.path.getsize(result["saved_path"]) > 0
    assert quantized["dynamic"]["model_size_bytes"] != quantized["static"]["model_size_bytes"]


# Milestone 6: Demonstrate size reduction and evaluate accuracy impact
@pytest.fixture(scope="module")
def comparison(tmp_path_factory, trained, dataset):
    path = tmp_path_factory.mktemp("comparison") / "comparison.json"
    return core.milestone_6(dataset=dataset, model_path=trained["model_path"],
                            n_calibration=16, max_eval=40, benchmark_repetitions=5,
                            out_dir=str(tmp_path_factory.mktemp("artifacts")),
                            out_path=str(path))


def test_milestone_6_comparison_structure(comparison):
    assert comparison["baseline_label"] == "Original (float32)"
    assert [row["method"] for row in comparison["results"]] == ["original", "dynamic", "static"]
    for row in comparison["results"]:
        for key in ("method", "label", "status", "model_size_bytes", "model_size_mb",
                    "size_reduction_bytes", "size_reduction_percent", "compression_ratio",
                    "top1_accuracy", "top3_accuracy", "mrr", "mean_nll", "perplexity",
                    "top1_delta", "top3_delta", "mrr_delta", "avg_inference_ms",
                    "inference_speedup", "quantized_layers", "limitation", "error"):
            assert key in row
    assert comparison["results"][0]["status"] == "ok"
    assert comparison["n_eval_used"] == 40
    assert set(comparison["summary"]) == {"quantization_succeeded", "quantization_failed",
                                          "smallest_model", "best_top1_accuracy"}


def test_milestone_6_size_reduction_math(comparison):
    original, dynamic, static = comparison["results"]
    assert original["model_size_bytes"] > 0
    assert original["size_reduction_bytes"] == 0
    assert original["size_reduction_percent"] == 0.0
    assert original["compression_ratio"] == 1.0

    for row in (dynamic, static):
        assert row["status"] == "ok"
        assert row["model_size_bytes"] < original["model_size_bytes"]
        assert row["size_reduction_bytes"] == original["model_size_bytes"] - row["model_size_bytes"]
        assert row["size_reduction_percent"] == pytest.approx(
            100.0 * row["size_reduction_bytes"] / original["model_size_bytes"])
        assert row["compression_ratio"] == pytest.approx(
            original["model_size_bytes"] / row["model_size_bytes"])
        assert 0.0 < row["size_reduction_percent"] < 100.0
        assert row["model_size_mb"] == pytest.approx(row["model_size_bytes"] / 1e6)


def test_milestone_6_accuracy_and_latency_math(comparison):
    original, dynamic, static = comparison["results"]
    for name in ("top1_accuracy", "top3_accuracy", "mrr"):
        assert original[name] is not None
    assert original["top1_delta"] == 0.0 and original["mrr_delta"] == 0.0
    assert original["inference_speedup"] == 1.0

    for row in (dynamic, static):
        assert 0.0 <= row["top1_accuracy"] <= 1.0
        assert 0.0 <= row["top3_accuracy"] <= 1.0
        assert 0.0 < row["mrr"] <= 1.0
        assert row["top1_delta"] == pytest.approx(row["top1_accuracy"] - original["top1_accuracy"])
        assert row["top3_delta"] == pytest.approx(row["top3_accuracy"] - original["top3_accuracy"])
        assert row["mrr_delta"] == pytest.approx(row["mrr"] - original["mrr"])
        assert row["avg_inference_ms"] > 0
        assert row["inference_speedup"] == pytest.approx(
            original["avg_inference_ms"] / row["avg_inference_ms"])
    assert comparison["n_eval_used"] == 40


def test_milestone_6_handles_unavailable_quantization(trained, dataset, tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        return {"method": "static", "status": "failed", "model": None,
                "model_size_bytes": None, "quantized_layers": 0, "int8_weights": 0,
                "calibrated_on": 0, "limitation": "not supported in this build",
                "error": "RuntimeError: quantization unsupported"}

    monkeypatch.setattr(core, "quantize_static_model", unavailable)
    result = core.milestone_6(dataset=dataset, model_path=trained["model_path"],
                              n_calibration=8, max_eval=20, benchmark_repetitions=3,
                              out_dir=str(tmp_path / "artifacts"),
                              out_path=str(tmp_path / "comparison.json"))
    static = result["results"][2]
    assert static["method"] == "static" and static["status"] == "failed"
    assert static["error"] == "RuntimeError: quantization unsupported"
    for key in ("model_size_bytes", "size_reduction_bytes", "size_reduction_percent",
                "compression_ratio", "top1_accuracy", "top1_delta", "avg_inference_ms",
                "inference_speedup"):
        assert static[key] is None                       # no invented numbers, no zero-division
    assert result["summary"]["quantization_failed"] == ["static"]
    assert result["summary"]["quantization_succeeded"] == ["dynamic"]
    assert result["results"][1]["status"] == "ok"


def test_milestone_6_writes_reusable_json(trained, dataset, tmp_path):
    import json

    out_path = tmp_path / "comparison.json"
    result = core.milestone_6(dataset=dataset, model_path=trained["model_path"],
                              n_calibration=8, max_eval=10, benchmark_repetitions=2,
                              out_dir=str(tmp_path / "artifacts"), out_path=str(out_path))
    assert json.loads(out_path.read_text()) == result       # the UI can read it back as is
