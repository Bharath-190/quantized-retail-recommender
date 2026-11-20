"""Your code. One stub per milestone; build them in order and keep each one small."""
import json
from pathlib import Path

import numpy as np

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
