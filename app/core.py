"""Your code. One stub per milestone; build them in order and keep each one small."""
import copy
import io
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.ao.quantization as tq
from torch import nn
from torch.nn import functional as F

# --- Milestone 1: Prepare synthetic shopping basket data for training
# Hint: Use the data preparation techniques from 'Teach a Language Model What Sells Together' to format your shopping basket sequences.

# Tokens 0, 1 and 2 are reserved so every product id is >= 3 and the sequences read
# like text: <bos> item item item <eos>.
PAD, BOS, EOS = 0, 1, 2
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]

DEFAULT_DATASET_PATH = "data/dataset.json"

# A small shop: 76 products over five departments. Names are unique, so a product name
# is already a perfectly good token and the mapping stays a plain dict.
CATALOG = {
    "groceries": [
        "Milk 1L", "Whole Wheat Bread", "Eggs 12 Pack", "Butter", "Cheddar Cheese",
        "Plain Yogurt", "Basmati Rice", "Pasta", "Tomato Sauce", "Olive Oil",
        "Soy Sauce", "Sugar", "Salt", "Black Pepper", "Garlic",
        "Onions", "Tomatoes", "Bananas", "Apples", "Spinach",
        "Chicken Breast", "Minced Beef", "Firm Tofu", "Baked Beans", "Canned Tuna",
        "Peanut Butter", "Strawberry Jam", "Honey", "All-Purpose Flour", "Rolled Oats",
        "Breakfast Cereal",
    ],
    "snacks": [
        "Potato Chips", "Tortilla Chips", "Salsa", "Milk Chocolate Bar", "Butter Cookies",
        "Saltine Crackers", "Microwave Popcorn", "Mixed Nuts", "Pretzels", "Vanilla Ice Cream",
        "Granola Bars", "Instant Noodles",
    ],
    "beverages": [
        "Ground Coffee", "Tea Bags", "Orange Juice", "Cola", "Sparkling Water",
        "Lager Beer", "Red Wine", "Green Tea", "Energy Drink", "Bottled Water",
        "Lemonade",
    ],
    "household": [
        "Laundry Detergent", "Fabric Softener", "Dish Soap", "Sponges", "Paper Towels",
        "Toilet Paper", "Trash Bags", "Bleach", "Glass Cleaner", "Aluminum Foil",
        "Cling Film",
    ],
    "personal care": [
        "Toothpaste", "Toothbrush", "Shampoo", "Conditioner", "Body Wash",
        "Soap Bar", "Deodorant", "Razor Blades", "Hand Sanitizer", "Moisturizer",
        "Cotton Swabs",
    ],
}

# Things that get bought together. The first two names of a theme are its anchors: if a
# shopper is on that errand they almost always take both, which is the signal the model
# has to learn later. The remaining names are loose extras.
THEMES = [
    ("Milk 1L", "Breakfast Cereal", "Whole Wheat Bread", "Butter", "Ground Coffee", "Strawberry Jam"),
    ("Pasta", "Tomato Sauce", "Cheddar Cheese", "Garlic", "Olive Oil", "Red Wine"),
    ("Tortilla Chips", "Salsa", "Minced Beef", "Cheddar Cheese", "Onions", "Tomatoes", "Cola"),
    ("Potato Chips", "Microwave Popcorn", "Cola", "Milk Chocolate Bar", "Vanilla Ice Cream"),
    ("Laundry Detergent", "Fabric Softener", "Bleach", "Glass Cleaner", "Sponges", "Paper Towels"),
    ("Toothpaste", "Toothbrush", "Shampoo", "Conditioner", "Soap Bar", "Deodorant"),
    ("Spinach", "Tomatoes", "Onions", "Olive Oil", "Chicken Breast", "Apples"),
    ("Tea Bags", "Butter Cookies", "Milk 1L", "Sugar"),
    ("Lager Beer", "Pretzels", "Mixed Nuts", "Potato Chips"),
    ("Firm Tofu", "Basmati Rice", "Soy Sauce", "Spinach", "Garlic"),
]

MAX_BASKET_ITEMS = 14   # longest basket we keep, so one sequence is at most 16 tokens
THEME_PROB = 0.65       # share of baskets that start from a shopping errand
MAX_LEN = MAX_BASKET_ITEMS + 2  # <bos> + items + <eos>


def _products() -> list[str]:
    """Every product name, in catalogue order (this order fixes the token ids)."""
    products = [name for names in CATALOG.values() for name in names]
    if len(set(products)) != len(products):
        raise ValueError("product names must be unique: a name is its token")
    return products


def _vocab() -> tuple[list[str], dict[str, int]]:
    """The token id of every product: specials first, then the catalogue in order."""
    products = _products()
    vocab = SPECIAL_TOKENS + products
    return vocab, {name: i for i, name in enumerate(vocab) if i >= len(SPECIAL_TOKENS)}


def _popularity(products: list[str]) -> np.ndarray:
    """Sales weights: each department's best sellers are common, the tail is rare."""
    weight = {}
    for names in CATALOG.values():
        for rank, name in enumerate(names):
            weight[name] = 1.0 if rank < 4 else (0.55 if rank < 8 else 0.3)
    shares = np.array([weight[name] for name in products], dtype=float)
    return shares / shares.sum()


def _make_basket(rng, products, shares, item2id, position) -> list[int]:
    """One basket of product ids, in a random order (a basket is a set, not a list)."""
    size = int(np.clip(rng.poisson(5.0) + 2, 2, MAX_BASKET_ITEMS))
    chosen: set[str] = set()
    if rng.random() < THEME_PROB:
        theme = THEMES[int(rng.integers(len(THEMES)))]  # pick a shopping errand
        chosen.update(theme[:2])                        # ...and take its anchors
        extras = list(theme[2:])
        take = min(len(extras), max(0, size - len(chosen)))
        if take:
            chosen.update(str(name) for name in rng.choice(extras, size=take, replace=False))
    available = np.ones(len(products), dtype=bool)
    for name in chosen:
        available[position[name]] = False
    while len(chosen) < size:                           # fill up on ordinary popular items
        weights = shares * available
        weights = weights / weights.sum()
        index = int(rng.choice(len(products), p=weights))
        chosen.add(products[index])
        available[index] = False
    order = list(chosen)
    rng.shuffle(order)
    return [item2id[name] for name in order]


def milestone_1(n_baskets: int = 2500, seed: int = 0,
                out_path: str | None = DEFAULT_DATASET_PATH) -> dict:
    """Generate the synthetic shopping baskets and the vocabulary a language model needs.

    Returns one plain-JSON dict (also written to ``out_path`` so later milestones reuse
    the exact same data instead of generating something new)::

        {
          "seed": 0, "n_baskets": 2500, "max_len": 16,
          "catalog": {department: [product, ...]},
          "themes": [[product, ...], ...],        # the planted co-purchase groups
          "vocab": ["<pad>", "<bos>", "<eos>", product, ...],   # index == token id
          "item2id": {product: token id},
          "train": [[token, ...], ...],           # <bos> items... <eos>
          "eval":  [[token, ...], ...],
          "stats": {"n_train": ..., "n_eval": ..., "n_products": ..., ...},
        }

    Every sequence is a causal-LM training example for next-item prediction: predict
    each item from the items before it, and ``<eos>`` from the last item.
    """
    if n_baskets < 20:
        raise ValueError("need at least 20 baskets to split into train and eval")
    rng = np.random.default_rng(seed)
    products = _products()
    vocab, item2id = _vocab()
    for theme in THEMES:
        if len(theme) < 2:
            raise ValueError(f"a theme needs at least two anchors: {theme}")
        for name in theme:
            if name not in item2id:
                raise ValueError(f"theme item not in the catalogue: {name}")
    shares = _popularity(products)
    position = {name: i for i, name in enumerate(products)}

    sequences = []
    for _ in range(n_baskets):
        items = _make_basket(rng, products, shares, item2id, position)
        sequences.append([BOS] + items + [EOS])

    order = rng.permutation(len(sequences))          # shuffle so themes are not clustered
    n_train = int(round(0.9 * len(sequences)))
    train = [sequences[int(i)] for i in order[:n_train]]
    evaluation = [sequences[int(i)] for i in order[n_train:]]

    lengths = np.array([len(s) - 2 for s in sequences])
    dataset = {
        "seed": seed,
        "n_baskets": n_baskets,
        "max_len": MAX_LEN,
        "catalog": CATALOG,
        "themes": [list(theme) for theme in THEMES],
        "vocab": vocab,
        "item2id": item2id,
        "train": train,
        "eval": evaluation,
        "stats": {
            "n_train": len(train),
            "n_eval": len(evaluation),
            "n_products": len(products),
            "vocab_size": len(vocab),
            "mean_basket_items": float(lengths.mean()),
            "max_basket_items": int(lengths.max()),
        },
    }
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset))
    return dataset


def load_dataset(path: str = DEFAULT_DATASET_PATH) -> dict:
    """Read back what ``milestone_1`` wrote, in the same shape it returned."""
    return json.loads(Path(path).read_text())


def decode_sequence(dataset: dict, tokens: list[int]) -> list[str]:
    """The product names in a token sequence, with the ``<bos>``/``<eos>``/``<pad>`` dropped."""
    return [dataset["vocab"][int(token)] for token in tokens
            if int(token) >= len(SPECIAL_TOKENS)]


# --- Milestone 2: Continue pretraining a small LM on the basket data
# Hint: Apply the fine-tuning methods from 'Teach a Language Model What Sells Together' to adapt your model.

DEFAULT_MODEL_PATH = "data/model.pt"

# Deliberately tiny: ~79k parameters, a 64-wide residual stream and two blocks. That is
# enough to learn 79 product tokens and still trains in seconds on a laptop CPU.
DEFAULT_MODEL_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 2,
    "d_ff": 128,
    "dropout": 0.0,
}


def model_config(vocab_size: int, max_len: int = MAX_LEN, **overrides) -> dict:
    """The full model configuration: vocabulary, context length and the tiny architecture."""
    config = {
        "vocab_size": vocab_size,
        "max_len": max_len,
        "pad_token_id": PAD,
        "bos_token_id": BOS,
        "eos_token_id": EOS,
        **DEFAULT_MODEL_CONFIG,
    }
    config.update(overrides)
    return config


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention written with plain Linear layers and plain matmuls.

    Keeping the matrix multiplications explicit (instead of using a fused attention op)
    keeps the model easy to read and easy to quantize in milestone 5.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.n_heads = config["n_heads"]
        self.head_dim = config["d_model"] // self.n_heads
        self.qkv = nn.Linear(config["d_model"], 3 * config["d_model"])
        self.proj = nn.Linear(config["d_model"], config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, steps, width = x.shape
        shape = (batch, steps, self.n_heads, self.head_dim)
        query, key, value = self.qkv(x).split(width, dim=2)
        query = query.view(shape).transpose(1, 2)   # (batch, heads, steps, head_dim)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ value).transpose(1, 2).reshape(batch, steps, width)
        return self.proj(out)


class Block(nn.Module):
    """One pre-norm Transformer block: attention, then a small feed-forward network."""

    def __init__(self, config: dict):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["d_model"])
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["d_model"])
        self.mlp = nn.Sequential(
            nn.Linear(config["d_model"], config["d_ff"]),
            nn.GELU(),
            nn.Linear(config["d_ff"], config["d_model"]),
        )
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.ln_1(x), mask))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x


class TinyCausalLM(nn.Module):
    """A tiny decoder-only Transformer language model over product tokens.

    ``forward(tokens)`` takes a ``(batch, steps)`` LongTensor of token ids and returns
    ``(batch, steps, vocab_size)`` logits; position ``t`` predicts token ``t + 1``.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = dict(config)
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"],
                                            padding_idx=config["pad_token_id"])
        self.position_embedding = nn.Embedding(config["max_len"], config["d_model"])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["n_layers"])])
        self.ln_f = nn.LayerNorm(config["d_model"])
        self.head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def attention_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """True where attention is forbidden: the future, and padded positions."""
        steps = tokens.shape[1]
        future = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=tokens.device),
                            diagonal=1)
        padding = tokens == self.config["pad_token_id"]
        return padding[:, None, None, :] | future[None, None, :, :]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        steps = tokens.shape[1]
        positions = torch.arange(steps, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        mask = self.attention_mask(tokens)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))


def create_model(config: dict) -> TinyCausalLM:
    """Build the tiny model from its configuration dict (no weights loaded)."""
    return TinyCausalLM(config)


def _pad_batch(sequences: list[list[int]], pad_token_id: int):
    """Stack ragged baskets into one batch: inputs are ``tokens[:-1]``, targets ``tokens[1:]``."""
    width = max(len(sequence) for sequence in sequences)
    tokens = torch.full((len(sequences), width), pad_token_id, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        tokens[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return tokens[:, :-1], tokens[:, 1:]


def train_model(dataset: dict, config: dict | None = None, epochs: int = 20,
                batch_size: int = 64, lr: float = 3e-3, seed: int = 0):
    """Train the tiny LM with a plain next-token cross-entropy loop.

    Returns ``(model, loss_history)`` with one mean training loss per epoch.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = config or model_config(len(dataset["vocab"]), dataset["max_len"])
    model = create_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sequences = dataset["train"]
    history = []
    for _ in range(epochs):
        model.train()
        total, seen = 0.0, 0
        order = rng.permutation(len(sequences))
        for start in range(0, len(sequences), batch_size):
            batch = [sequences[int(i)] for i in order[start:start + batch_size]]
            inputs, targets = _pad_batch(batch, config["pad_token_id"])
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, config["vocab_size"]),
                                   targets.reshape(-1),
                                   ignore_index=config["pad_token_id"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        history.append(total / max(1, seen))
    model.eval()
    return model, history


def save_model(path: str, model: TinyCausalLM, dataset: dict, seed: int) -> str:
    """Write weights plus the minimal metadata later milestones need (vocab + config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": dict(model.config),
        "vocab": dataset["vocab"],
        "item2id": dataset["item2id"],
        "seed": seed,
        "state_dict": model.state_dict(),
    }, path)
    return str(path)


def load_model(path: str = DEFAULT_MODEL_PATH) -> tuple[TinyCausalLM, dict]:
    """Read back a saved checkpoint; the model is returned in eval mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def milestone_2(dataset: dict | None = None, epochs: int = 20, batch_size: int = 64,
                lr: float = 3e-3, seed: int = 0,
                out_path: str | None = DEFAULT_MODEL_PATH) -> dict:
    """Train the tiny causal LM on the milestone 1 basket sequences.

    The objective is ordinary next-token prediction, so the model is a real language
    model over the product vocabulary. The checkpoint (weights, vocabulary and model
    config) is written to ``out_path`` and the returned dict is JSON-safe metadata::

        {
          "model_path": "data/model.pt",
          "config": {"vocab_size": 79, "d_model": 64, ...},
          "vocab": ["<pad>", "<bos>", "<eos>", product, ...],
          "item2id": {product: token id},
          "seed": 0, "epochs": 20, "n_train": 2250, "n_params": 78592,
          "loss_history": [...one mean training loss per epoch...],
          "final_loss": 1.234,
        }
    """
    if dataset is None:
        data_path = Path(DEFAULT_DATASET_PATH)
        dataset = load_dataset() if data_path.exists() else milestone_1()
    model, history = train_model(dataset, epochs=epochs, batch_size=batch_size,
                                 lr=lr, seed=seed)
    if out_path:
        save_model(out_path, model, dataset, seed)
    return {
        "model_path": str(out_path) if out_path else None,
        "config": dict(model.config),
        "vocab": dataset["vocab"],
        "item2id": dataset["item2id"],
        "seed": seed,
        "epochs": epochs,
        "n_train": len(dataset["train"]),
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "loss_history": [float(value) for value in history],
        "final_loss": float(history[-1]),
    }


# --- Milestone 3: Implement next-item prediction and itemset probing
# Hint: Use the inference and analysis patterns from 'Teach a Language Model What Sells Together' to generate recommendations.

PRODUCT_OFFSET = len(SPECIAL_TOKENS)   # token id of the first real product


def _basket_tokens(dataset: dict, basket: list[str]) -> list[int]:
    """Validate a basket of product names and turn it into model input tokens.

    Duplicates are ignored and a basket longer than the model's context keeps only its
    most recent items. An empty basket is allowed: it is just the ``<bos>`` token.
    """
    item2id = dataset["item2id"]
    unknown = [name for name in basket if name not in item2id]
    if unknown:
        raise ValueError(f"unknown product(s): {unknown}. "
                         f"Pick names from the catalogue, e.g. {list(item2id)[:3]}")
    ids = list(dict.fromkeys(item2id[name] for name in basket))
    return [BOS] + ids[-(dataset["max_len"] - 2):]


def next_item_probabilities(model: TinyCausalLM, dataset: dict,
                            basket: list[str]) -> torch.Tensor:
    """P(next product | basket) for every token, straight from the language model."""
    tokens = _basket_tokens(dataset, basket)
    with torch.no_grad():
        logits = model(torch.tensor([tokens]))[0, -1]
    return torch.softmax(logits, dim=-1)


def recommend(model: TinyCausalLM, dataset: dict, basket: list[str],
              k: int = 5) -> list[dict]:
    """Rank the products the model thinks come next, skipping what is already in the basket.

    Returns ``[{"product": name, "probability": p}, ...]`` sorted by probability.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    probabilities = next_item_probabilities(model, dataset, basket)
    vocab, item2id = dataset["vocab"], dataset["item2id"]
    seen = {item2id[name] for name in basket if name in item2id}
    ranked = [(vocab[token], float(probabilities[token]))
              for token in range(PRODUCT_OFFSET, len(vocab)) if token not in seen]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return [{"product": name, "probability": probability} for name, probability in ranked[:k]]


def probe_itemset(model: TinyCausalLM, dataset: dict, items: list[str], k: int = 3) -> dict:
    """Itemset probing: hide one item of a known combination and rank it from the rest.

    This asks the itemset question - do these products go together? - of the model rather
    than of a counter. For every item in turn the others are shown as the basket and the
    hidden item's rank among all products (1 is best) is recorded. Nothing is sampled,
    so the probe is deterministic.
    """
    if len(items) < 2:
        raise ValueError("probe a combination of at least two products")
    _basket_tokens(dataset, items)                      # validates the product names
    vocab, item2id = dataset["vocab"], dataset["item2id"]
    ranks, probabilities = [], []
    for hidden in dict.fromkeys(items):
        visible = [name for name in items if name != hidden]
        scores = next_item_probabilities(model, dataset, visible)
        target = item2id[hidden]
        seen = {item2id[name] for name in visible}
        better = sum(1 for token in range(PRODUCT_OFFSET, len(vocab))
                     if token not in seen and token != target
                     and float(scores[token]) >= float(scores[target]))
        ranks.append(1 + better)
        probabilities.append(float(scores[target]))
    return {
        "items": list(items),
        "ranks": ranks,                                  # one rank per item, same order
        "hit@k": sum(rank <= k for rank in ranks) / len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "mean_probability": sum(probabilities) / len(probabilities),
    }


def probe_itemsets(model: TinyCausalLM, dataset: dict, k: int = 3,
                   themes: list[list[str]] | None = None) -> dict:
    """Probe every planted co-purchase pair (and optionally other combinations)."""
    themes = dataset["themes"] if themes is None else themes
    per_itemset = [probe_itemset(model, dataset, list(theme[:2]), k=k) for theme in themes]
    return {
        "k": k,
        "n_itemsets": len(per_itemset),
        "hit@k": sum(result["hit@k"] for result in per_itemset) / len(per_itemset),
        "mean_rank": sum(result["mean_rank"] for result in per_itemset) / len(per_itemset),
        "per_itemset": [{"items": result["items"], "ranks": result["ranks"],
                         "hit@k": result["hit@k"]} for result in per_itemset],
    }


def milestone_3(basket: list[str] | None = None, k: int = 5,
                model_path: str = DEFAULT_MODEL_PATH, dataset: dict | None = None,
                probe_k: int = 3) -> dict:
    """Recommend the next products for a basket and probe the planted combinations.

    The recommendations are the model's own next-token distribution, so they change when
    the basket changes. Returns JSON-safe results::

        {
          "basket": [product, ...],
          "recommendations": [{"product": name, "probability": 0.12}, ...],
          "probing": {"k": 3, "n_itemsets": 10, "hit@k": 0.6,
                      "mean_rank": 4.2, "per_itemset": [...]},
        }
    """
    if dataset is None:
        dataset = load_dataset() if Path(DEFAULT_DATASET_PATH).exists() else milestone_1()
    if not Path(model_path).exists():
        milestone_2(dataset=dataset, out_path=model_path)
    model, _ = load_model(model_path)
    basket = list(basket) if basket is not None else ["Milk 1L", "Breakfast Cereal"]
    return {
        "basket": basket,
        "recommendations": recommend(model, dataset, basket, k=k),
        "probing": probe_itemsets(model, dataset, k=probe_k),
    }


# --- Milestone 4: Measure the baseline size and accuracy of the model
# Hint: Before quantization, record the model's original size and evaluate its performance metrics.

DEFAULT_BASELINE_PATH = "data/baseline.json"


def model_size_bytes(model: nn.Module) -> int:
    """The bytes the model's weights really take when written to disk.

    Saving the state dict to memory instead of the file system keeps the measurement
    repeatable and gives every variant (float32, dynamic int8, static int8) the same
    fair measurement in milestones 4 and 6.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue())


def evaluate_model(model: TinyCausalLM, dataset: dict,
                   max_examples: int | None = None) -> dict:
    """Next-item accuracy on the held-out baskets; the same procedure every variant uses.

    Each evaluation basket hides its last product: the model sees the basket without it
    and we rank the hidden product among the products not already in the basket. Top-1,
    top-3, MRR, mean negative log-likelihood and perplexity are computed from those ranks
    and probabilities. Nothing is sampled, so the numbers are deterministic.
    """
    sequences = dataset["eval"] if max_examples is None else dataset["eval"][:max_examples]
    vocab_size = len(dataset["vocab"])
    ranks, probabilities, losses = [], [], []
    for tokens in sequences:
        if len(tokens) < 4:                        # <bos> + one product + <eos> is nothing to predict
            continue
        prefix, target = tokens[:-2], tokens[-2]    # drop the <eos>, then hide the last product
        with torch.no_grad():
            logits = model(torch.tensor([prefix]))[0, -1]
        log_probabilities = torch.log_softmax(logits, dim=-1)
        losses.append(-float(log_probabilities[target]))
        probabilities.append(float(log_probabilities[target].exp()))
        visible = set(prefix)
        better = sum(1 for token in range(PRODUCT_OFFSET, vocab_size)
                     if token not in visible and token != target
                     and float(logits[token]) >= float(logits[target]))
        ranks.append(1 + better)
    n = len(ranks)
    if n == 0:
        raise ValueError("no evaluation baskets to score")
    mean_nll = sum(losses) / n
    return {
        "n_eval_used": n,
        "top1_accuracy": sum(rank == 1 for rank in ranks) / n,
        "top3_accuracy": sum(rank <= 3 for rank in ranks) / n,
        "mrr": sum(1.0 / rank for rank in ranks) / n,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "mean_probability": sum(probabilities) / n,
    }


def benchmark_inference(model: TinyCausalLM, dataset: dict,
                        repetitions: int = 100) -> float:
    """Mean seconds for one basket's next-item prediction, batch size one, on CPU."""
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    prefixes = [tokens[:-2] for tokens in dataset["eval"] if len(tokens) >= 4]
    if not prefixes:
        raise ValueError("no evaluation baskets to benchmark")
    model.eval()
    with torch.no_grad():
        for _ in range(3):                          # warm-up, not measured
            model(torch.tensor([prefixes[0]]))
        start = time.perf_counter()
        for step in range(repetitions):
            model(torch.tensor([prefixes[step % len(prefixes)]]))
        elapsed = time.perf_counter() - start
    return elapsed / repetitions


def milestone_4(dataset: dict | None = None, model_path: str = DEFAULT_MODEL_PATH,
                max_eval: int | None = None, benchmark_repetitions: int = 100,
                probe_k: int = 3, out_path: str | None = DEFAULT_BASELINE_PATH) -> dict:
    """Baseline size, accuracy and latency of the unquantized model.

    Everything is measured on the held-out evaluation baskets with the same functions the
    quantized models will use in milestones 5 and 6. The JSON-safe result is returned and
    (by default) written to ``data/baseline.json`` so milestone 6 can reuse it::

        {
          "label": "float32 (baseline)", "model_size_bytes": 324870,
          "model_size_mb": 0.325, "n_parameters": 78208,
          "n_train": 2250, "n_eval": 250, "n_eval_used": 250,
          "top1_accuracy": 0.31, "top3_accuracy": 0.55, "mrr": 0.44,
          "mean_nll": 2.85, "perplexity": 17.3, "mean_probability": 0.11,
          "avg_inference_ms": 0.4,
          "probing": {"k": 3, "n_itemsets": 10, "hit@k": 0.9, "mean_rank": 2.5},
        }
    """
    if dataset is None:
        dataset = load_dataset() if Path(DEFAULT_DATASET_PATH).exists() else milestone_1()
    if not Path(model_path).exists():
        milestone_2(dataset=dataset, out_path=model_path)
    model, _ = load_model(model_path)

    size = model_size_bytes(model)
    probing = probe_itemsets(model, dataset, k=probe_k)
    results = {
        "label": "float32 (baseline)",
        "model_path": str(model_path),
        "model_size_bytes": size,
        "model_size_mb": size / 1e6,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "n_train": len(dataset["train"]),
        "n_eval": len(dataset["eval"]),
        **evaluate_model(model, dataset, max_examples=max_eval),
        "avg_inference_ms": 1000.0 * benchmark_inference(model, dataset,
                                                         repetitions=benchmark_repetitions),
        "probing": {"k": probe_k, "n_itemsets": probing["n_itemsets"],
                    "hit@k": probing["hit@k"], "mean_rank": probing["mean_rank"]},
    }
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
    return results


# --- Milestone 5: Apply dynamic and static post-training quantization to the model
# Hint: Implement the quantization steps from 'Shrink a Neural Network with Quantisation' to reduce model precision.

DEFAULT_DYNAMIC_PATH = "data/model_dynamic.pt"
DEFAULT_STATIC_PATH = "data/model_static.pt"
DEFAULT_QUANTIZATION_PATH = "data/quantization.json"

STATIC_LIMITATION = (
    "PyTorch 2.15 no longer ships FX graph mode (prepare_fx/convert_fx were removed), and eager "
    "mode cannot put softmax, matmul, LayerNorm or GELU on int8 because they have no quantized "
    "kernels. So every nn.Linear is quantized on its own between QuantStub and DeQuantStub "
    "(int8 weights and int8 activations, calibrated on training baskets); the attention math "
    "around it stays float32. Quantizing the head also requires dequantizing its output."
)


def _count_quantized_linears(module: nn.Module, quantized_type) -> int:
    return sum(isinstance(child, quantized_type) for child in module.modules())


def quantize_dynamic_model(model: TinyCausalLM) -> dict:
    """Dynamic post-training quantization: int8 weights for every ``nn.Linear``.

    Linears are the only supported layer type in this tiny Transformer (embeddings,
    LayerNorm and attention stay float32). The original model is copied, never changed.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantized = tq.quantize_dynamic(copy.deepcopy(model), {nn.Linear},
                                           dtype=torch.qint8)
        quantized.eval()
        layers = _count_quantized_linears(quantized, torch.ao.nn.quantized.dynamic.Linear)
        if layers == 0:
            return {"method": "dynamic", "status": "failed", "model": None,
                    "model_size_bytes": None, "quantized_layers": 0, "limitation": None,
                    "error": "no nn.Linear layer was quantized"}
        return {
            "method": "dynamic",
            "status": "ok",
            "model": quantized,
            "model_size_bytes": model_size_bytes(quantized),
            "quantized_layers": layers,
            "limitation": ("Only Linear weights are int8 (activations are quantized on the fly); "
                           "embeddings, LayerNorm and the attention math remain float32."),
            "error": None,
        }
    except Exception as error:                      # report honestly instead of pretending
        return {"method": "dynamic", "status": "failed", "model": None,
                "model_size_bytes": None, "quantized_layers": 0, "limitation": None,
                "error": f"{type(error).__name__}: {error}"}


class _StaticLinear(nn.Module):
    """One ``nn.Linear`` quantized in eager mode between a QuantStub and a DeQuantStub.

    Eager static quantization needs the activation range observed during calibration, so
    the input is quantized, the int8 matmul runs, and the result is dequantized to float
    for whatever float operation comes next.
    """

    def __init__(self, linear: nn.Linear, qconfig):
        super().__init__()
        self.quant = tq.QuantStub()
        self.linear = linear
        self.dequant = tq.DeQuantStub()
        self.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.linear(self.quant(x)))


def _wrap_linears(module: nn.Module, qconfig) -> None:
    """Replace every Linear child (recursively) with a quantizable wrapper."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _StaticLinear(child, qconfig))
        else:
            _wrap_linears(child, qconfig)


def quantize_static_model(model: TinyCausalLM, dataset: dict, n_calibration: int = 128,
                          batch_size: int = 16) -> dict:
    """Static post-training quantization with observers calibrated on training baskets.

    Calibration only ever reads ``dataset["train"]``; the held-out eval baskets are used
    later, for scoring, never for choosing the activation ranges.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qconfig = tq.get_default_qconfig(torch.backends.quantized.engine)
            prepared_model = copy.deepcopy(model).eval()
            _wrap_linears(prepared_model, qconfig)
            prepared = tq.prepare(prepared_model, inplace=False)
            baskets = dataset["train"][:max(1, n_calibration)]
            with torch.no_grad():                   # observers record the activation ranges
                for start in range(0, len(baskets), batch_size):
                    inputs, _ = _pad_batch(baskets[start:start + batch_size], PAD)
                    prepared(inputs)
            converted = tq.convert(prepared, inplace=False).eval()
        quantized_linears = [child for child in converted.modules()
                             if isinstance(child, torch.ao.nn.quantized.Linear)]
        int8_weights = sum(1 for child in quantized_linears
                           if child.weight().dtype == torch.qint8)
        if not quantized_linears or int8_weights != len(quantized_linears):
            return {"method": "static", "status": "failed", "model": None,
                    "model_size_bytes": None, "quantized_layers": len(quantized_linears),
                    "int8_weights": int8_weights, "calibrated_on": len(baskets),
                    "limitation": STATIC_LIMITATION,
                    "error": "conversion produced no int8 weights"}
        return {
            "method": "static",
            "status": "ok",
            "model": converted,
            "model_size_bytes": model_size_bytes(converted),
            "quantized_layers": len(quantized_linears),
            "int8_weights": int8_weights,
            "calibrated_on": len(baskets),
            "limitation": STATIC_LIMITATION,
            "error": None,
        }
    except Exception as error:
        return {"method": "static", "status": "failed", "model": None,
                "model_size_bytes": None, "quantized_layers": 0, "int8_weights": 0,
                "calibrated_on": 0, "limitation": STATIC_LIMITATION,
                "error": f"{type(error).__name__}: {error}"}


def _json_safe(result: dict) -> dict:
    """Drop the live module so a quantization result can be written as JSON."""
    return {key: value for key, value in result.items() if key != "model"}


def milestone_5(dataset: dict | None = None, model_path: str = DEFAULT_MODEL_PATH,
                n_calibration: int = 128, max_eval: int | None = None,
                out_dir: str = "data") -> dict:
    """Apply dynamic and static post-training quantization and measure both.

    Returns the baseline, the dynamic result and the static result. Each result carries
    ``method``, ``status`` (``"ok"``/``"failed"``), the quantized ``model`` itself, its
    measured ``model_size_bytes``, its metrics on the held-out baskets, and any
    ``limitation``/``error``. JSON-safe copies are written to ``data/quantization.json``;
    the quantized weights are saved next to the baseline checkpoint.
    """
    if dataset is None:
        dataset = load_dataset() if Path(DEFAULT_DATASET_PATH).exists() else milestone_1()
    if not Path(model_path).exists():
        milestone_2(dataset=dataset, out_path=model_path)
    model, _ = load_model(model_path)

    baseline = {
        "label": "float32 (baseline)",
        "model_size_bytes": model_size_bytes(model),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": evaluate_model(model, dataset, max_examples=max_eval),
    }
    dynamic = quantize_dynamic_model(model)
    static = quantize_static_model(model, dataset, n_calibration=n_calibration)

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for result, filename in ((dynamic, DEFAULT_DYNAMIC_PATH), (static, DEFAULT_STATIC_PATH)):
        if result["model"] is None:
            result["metrics"], result["saved_path"] = None, None
            continue
        result["metrics"] = evaluate_model(result["model"], dataset, max_examples=max_eval)
        result["saved_path"] = str(directory / Path(filename).name)
        try:
            torch.save(result["model"].state_dict(), result["saved_path"])
        except Exception as error:
            result["saved_path"] = None
            result["save_error"] = f"{type(error).__name__}: {error}"

    results = {"baseline": baseline, "dynamic": dynamic, "static": static}
    summary = {"baseline": baseline, "dynamic": _json_safe(dynamic),
               "static": _json_safe(static)}
    path = directory / Path(DEFAULT_QUANTIZATION_PATH).name
    path.write_text(json.dumps(summary, indent=2))
    return results


# --- Milestone 6: Demonstrate size reduction and evaluate accuracy impact
# Hint: Compare the size of the quantized model to the original and re-evaluate its accuracy using techniques from 'Shrink a Neural Network with Quantisation'.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
