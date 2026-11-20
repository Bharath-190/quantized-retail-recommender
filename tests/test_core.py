"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
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
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Apply dynamic and static post-training quantization to the model
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")


# Milestone 6: Demonstrate size reduction and evaluate accuracy impact
@pytest.mark.skip(reason="remove this line when you start milestone 6")
def test_milestone_6():
    raise NotImplementedError("write the test for milestone 6")
