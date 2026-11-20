"""Your code. One stub per milestone; build them in order and keep each one small."""
import json
import math
from pathlib import Path

import numpy as np
import torch
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
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Measure the baseline size and accuracy of the model
# Hint: Before quantization, record the model's original size and evaluate its performance metrics.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Apply dynamic and static post-training quantization to the model
# Hint: Implement the quantization steps from 'Shrink a Neural Network with Quantisation' to reduce model precision.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Demonstrate size reduction and evaluate accuracy impact
# Hint: Compare the size of the quantized model to the original and re-evaluate its accuracy using techniques from 'Shrink a Neural Network with Quantisation'.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
