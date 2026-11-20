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
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Implement next-item prediction and itemset probing
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


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
