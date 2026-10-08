import math, random, json

# ─────────────────────────────────────────────
# 1. DATA PREPARATION
# ─────────────────────────────────────────────

AMHARIC_CORPUS = [
    "ልጁ ወደ ትምህርት ቤት ሄደ",
    "ልጅቷ ወደ ገበያ ሄደች",
    "አባቴ ወደ ገበያ ሄደ",
    "እናቴ ወደ ትምህርት ቤት ሄደች",
    "ውሻው ወደ ቤት ሄደ",
    "ድመቷ ወደ ቤት ሄደች",
    "ልጁ ወደ ገበያ ሄደ",
    "ልጅቷ ወደ ትምህርት ቤት ሄደች",
    "አባቴ ወደ ቤት ሄደ",
    "እናቴ ወደ ገበያ ሄደች",
    "ውሻው ምግብ በላ",
    "ድመቷ ምግብ በላች",
    "ልጁ ምግብ በላ",
    "አባቴ ዳቦ በላ",
    "ልጅቷ ዳቦ በላች",
    "እናቴ ምግብ በላች",
    "ውሻው ዳቦ በላ",
    "ድመቷ ዳቦ በላች",
    "ልጁ ፍሬ በላ",
    "ልጅቷ ፍሬ በላች",
    "አባቴ ቡና ጠጣ",
    "ድመቷ ወተት ጠጣች",
    "ልጅቷ ቡና ጠጣች",
    "እናቴ ቡና ጠጣች",
    "ልጁ ወተት ጠጣ",
    "ውሻው ቡና ጠጣ",
    "አባቴ ወተት ጠጣ",
    "ልጅቷ ወተት ጠጣች",
    "እናቴ ወተት ጠጣች",
    "ልጁ ቡና ጠጣ",
    "እናቴ ዳቦ ጋገረች",
    "ልጅቷ ዳቦ ጋገረች",
    "አባቴ ምግብ ሠራ",
    "እናቴ ምግብ ሠራች",
    "ልጅቷ ምግብ ሠራች",
    "አባቴ ዳቦ ጋገረ",
    "ልጁ ምግብ ሠራ",
    "እናቴ ፍሬ ሠራች",
    "ውሻው ምግብ ሠራ",
    "ድመቷ ዳቦ ጋገረች",
    "ልጁ መጽሐፍ አነበበ",
    "ልጅቷ መጽሐፍ አነበበች",
    "አባቴ ጋዜጣ አነበበ",
    "እናቴ መጽሐፍ አነበበች",
    "ልጁ ወደ ትምህርት ቤት ሄዶ አነበበ",
    "ልጅቷ ጋዜጣ አነበበች",
    "አባቴ ወደ ቤት ሄዶ አነበበ",
    "እናቴ ጋዜጣ አነበበች",
    "ልጁ ትምህርት ቤት ውስጥ አነበበ",
    "ልጅቷ ትምህርት ቤት ውስጥ አነበበች",
    "ልጁ ተጫወተ",
    "ልጅቷ ተጫወተች",
    "ውሻው ተጫወተ",
    "ልጁ ወደ ቤት ሄዶ ተጫወተ",
    "ልጅቷ ወደ ገበያ ሄዳ ተጫወተች",
    "አባቴ ቤት ውስጥ ተጫወተ",
    "እናቴ ቤት ውስጥ ተጫወተች",
    "ልጁ ምግብ በልቶ ተጫወተ",
    "ልጅቷ ቡና ጠጥታ ተጫወተች",
    "ውሻው ወደ ቤት ሄዶ ተጫወተ",
]

def preprocess(corpus):
    """Tokenise each sentence; return list of token lists."""
    sentences = []
    for line in corpus:
        tokens = line.strip().split()
        if tokens:
            sentences.append(tokens)
    return sentences

def build_vocab(sentences):
    """Build word↔id mappings from tokenised sentences."""
    counts = {}
    for sent in sentences:
        for w in sent:
            counts[w] = counts.get(w, 0) + 1
    vocab = sorted(counts)          
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w

def make_pairs(sentences, w2i, window):
    """Create (center_id, context_id) pairs; no cross-sentence pairs."""
    pairs = []
    for sent in sentences:
        ids = [w2i[w] for w in sent if w in w2i]
        for ci, center in enumerate(ids):
            lo = max(0, ci - window)
            hi = min(len(ids) - 1, ci + window)
            for ctx_pos in range(lo, hi + 1):
                if ctx_pos != ci:
                    pairs.append((center, ids[ctx_pos]))
    return pairs

# ─────────────────────────────────────────────
# 2. SKIP-GRAM MODEL (row-vector notation)
# ─────────────────────────────────────────────

def init_matrix(rows, cols, seed):
    """Small random values in [-0.01, 0.01]."""
    rng = random.Random(seed)
    return [[rng.uniform(-0.01, 0.01) for _ in range(cols)] for _ in range(rows)]

def forward(E, U, i):
    """h = E[i];  s = h U  (shape: V)."""
    h = E[i]
    d, V = len(h), len(U[0])
    s = [sum(h[k] * U[k][j] for k in range(d)) for j in range(V)]
    return h, s

def softmax(s):
    """Numerically stable softmax."""
    m = max(s)
    a = [math.exp(v - m) for v in s]
    total = sum(a)
    return [v / total for v in a], a, m, total

def loss_one(s, t):
    """L = (m - s[t]) + ln(sum(exp(s-m)))."""
    p, a, m, total = softmax(s)
    L = (m - s[t]) + math.log(total)
    return L, p

def gradients(h, U, p, t):
    """
    e[j] = p[j] - (1 if j==t else 0)
    grad_U[k][j] = h[k] * e[j]
    grad_h[k]    = sum_j U[k][j] * e[j]
    """
    V, d = len(p), len(h)
    e = [p[j] - (1 if j == t else 0) for j in range(V)]
    grad_U = [[h[k] * e[j] for j in range(V)] for k in range(d)]
    grad_h = [sum(U[k][j] * e[j] for j in range(V)) for k in range(d)]
    return grad_U, grad_h

def update(E, U, i, grad_U, grad_h, lr):
    """SGD update; only row E[i] is touched."""
    d, V = len(grad_U), len(grad_U[0])
    for k in range(d):
        for j in range(V):
            U[k][j] -= lr * grad_U[k][j]
    for k in range(d):
        E[i][k] -= lr * grad_h[k]

def mean_loss(E, U, pairs):
    """Evaluate J = (1/N) sum L_n with frozen weights."""
    if not pairs:
        return 0.0
    total = 0.0
    for (i, t) in pairs:
        _, s = forward(E, U, i)
        L, _ = loss_one(s, t)
        total += L
    return total / len(pairs)

def train(E, U, pairs, lr, epochs):
    """SGD training; shuffle pairs each epoch."""
    rng = random.Random(42)
    print(f"\n  Loss before training : {mean_loss(E, U, pairs):.6f}")
    for ep in range(1, epochs + 1):
        rng.shuffle(pairs)
        for (i, t) in pairs:
            h, s = forward(E, U, i)
            _, p = loss_one(s, t)
            gU, gh = gradients(h, U, p, t)
            update(E, U, i, gU, gh, lr)
        print(f"  Epoch {ep:3d} loss       : {mean_loss(E, U, pairs):.6f}")

# ─────────────────────────────────────────────
# 3. REQUIRED IMPLEMENTATION CHECKS
# ─────────────────────────────────────────────

TOL = 1e-5

def approx(a, b, tol=TOL):
    return abs(a - b) < tol

def check1():
    print("Check 1 – Context window")
    sentences = [["a","b","c","d"]]
    w2i = {"a":0,"b":1,"c":2,"d":3}
    pairs = make_pairs(sentences, w2i, window=1)
    expected = {(0,1),(1,0),(1,2),(2,1),(2,3),(3,2)}
    assert set(pairs) == expected, f"Got {pairs}"
    # cross-sentence: two sentences, no pair should span them
    s2 = [["a","b"],["c","d"]]
    p2 = make_pairs(s2, w2i, window=2)
    for (c,ctx) in p2:
        assert not ((c in (0,1)) and (ctx in (2,3))), "Cross-sentence pair found"
        assert not ((c in (2,3)) and (ctx in (0,1))), "Cross-sentence pair found"
    print("  PASSED")

def check2():
    print("Check 2 – Forward pass")
    E = [[1,0],[0,1],[1,1]]
    U = [[0,1,-1],[1,0,1]]
    h, s = forward(E, U, 0)   # cats=0
    expected_s = [0.0, 1.0, -1.0]
    for j in range(3):
        assert approx(s[j], expected_s[j]), f"s[{j}]={s[j]}"
    print(f"  scores = {[round(v,6) for v in s]}  PASSED")

def check3():
    print("Check 3 – Probabilities and loss")
    E = [[1,0],[0,1],[1,1]]
    U = [[0,1,-1],[1,0,1]]
    h, s = forward(E, U, 0)
    L, p = loss_one(s, 1)      # target = eat = 1
    exp_p = [0.244728, 0.665241, 0.090031]
    exp_L = 0.407606
    for j in range(3):
        assert approx(p[j], exp_p[j], 1e-5), f"p[{j}]={p[j]}"
    assert approx(L, exp_L, 1e-5), f"L={L}"
    assert all(v >= 0 for v in p)
    assert approx(sum(p), 1.0)
    print(f"  p = {[round(v,6) for v in p]}")
    print(f"  L = {round(L,6)}  PASSED")

def check4():
    print("Check 4 – Gradients")
    E = [[1,0],[0,1],[1,1]]
    U = [[0,1,-1],[1,0,1]]
    h, s = forward(E, U, 0)
    L, p = loss_one(s, 1)
    gU, gh = gradients(h, U, p, 1)
    exp_gU = [[0.244728,-0.334759,0.090031],[0,0,0]]
    exp_gh = [-0.424790, 0.334759]
    for k in range(2):
        for j in range(3):
            assert approx(gU[k][j], exp_gU[k][j], 1e-5), f"gU[{k}][{j}]={gU[k][j]}"
    for k in range(2):
        assert approx(gh[k], exp_gh[k], 1e-5), f"gh[{k}]={gh[k]}"
    print(f"  grad_U = {[[round(v,6) for v in row] for row in gU]}")
    print(f"  grad_h = {[round(v,6) for v in gh]}  PASSED")

def check5():
    print("Check 5 – One update")
    import copy
    E = [[1,0],[0,1],[1,1]]
    U = [[0,1,-1],[1,0,1]]
    E2 = copy.deepcopy(E)
    U2 = copy.deepcopy(U)
    h, s = forward(E2, U2, 0)
    L, p = loss_one(s, 1)
    gU, gh = gradients(h, U2, p, 1)
    update(E2, U2, 0, gU, gh, lr=0.1)
    exp_cats = [1.042479, -0.033476]
    for k in range(2):
        assert approx(E2[0][k], exp_cats[k], 1e-5), f"E[cats][{k}]={E2[0][k]}"
    # other rows unchanged
    for row in range(1, 3):
        for k in range(2):
            assert approx(E2[row][k], E[row][k]), f"Row {row} changed"
    # recomputed loss
    _, s2 = forward(E2, U2, 0)
    L2, _ = loss_one(s2, 1)
    assert approx(L2, 0.361859, 1e-5), f"L2={L2}"
    print(f"  E[cats] = {[round(v,6) for v in E2[0]]}")
    print(f"  Recomputed loss = {round(L2,6)}  PASSED")

def check6():
    print("Check 6 – Numerical stability")
    s = [1000.0, 0.0]
    L, p = loss_one(s, 1)
    assert approx(L, 1000.0, 1e-5), f"L={L}"
    # shift invariance
    s_shifted = [v + 100 for v in [0.0, 1.0, -1.0]]
    p1, _, _, _ = softmax([0.0, 1.0, -1.0])
    p2, _, _, _ = softmax(s_shifted)
    for j in range(3):
        assert approx(p1[j], p2[j]), f"Shift changed p[{j}]"
    L1, _ = loss_one([0.0, 1.0, -1.0], 1)
    L2, _ = loss_one(s_shifted, 1)
    assert approx(L1, L2), f"Shift changed loss"
    print(f"  Stable loss for [1000,0] target=1 : {round(L,6)}")
    print(f"  Shift invariance verified  PASSED")

def run_checks():
    print("=" * 50)
    print("IMPLEMENTATION CHECKS")
    print("=" * 50)
    check1(); check2(); check3(); check4(); check5(); check6()
    print("All checks passed.\n")

# ─────────────────────────────────────────────
# 5. EMBEDDING QUERIES
# ─────────────────────────────────────────────

def cosine(a, b):
    dot  = sum(a[k]*b[k] for k in range(len(a)))
    na   = math.sqrt(sum(v*v for v in a))
    nb   = math.sqrt(sum(v*v for v in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def nearest_neighbors(E, w2i, i2w, query, n=3):
    if query not in w2i:
        print(f"  '{query}' not in vocabulary.")
        return
    qi = w2i[query]
    qv = E[qi]
    sims = []
    for word, idx in w2i.items():
        if idx == qi:
            continue
        sims.append((cosine(qv, E[idx]), word))
    sims.sort(reverse=True)
    print(f"  Query: {query}")
    for sim, word in sims[:n]:
        print(f"    {word:20s}  cosine={sim:.6f}")

# ─────────────────────────────────────────────
# 6. OPTIONAL SAVE / LOAD + GRADIENT CHECK
# ─────────────────────────────────────────────

def save_model(path, w2i, i2w, E, U):
    data = {"w2i": w2i, "i2w": {str(k): v for k,v in i2w.items()}, "E": E, "U": U}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Model saved to {path}")

def load_model(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    w2i = data["w2i"]
    i2w = {int(k): v for k,v in data["i2w"].items()}
    return w2i, i2w, data["E"], data["U"]

def numerical_grad_check(E, U, i, t, eps=1e-5):
    """Check one element of grad_U and one of grad_h numerically."""
    import copy

    def L_val(E_, U_):
        _, s = forward(E_, U_, i)
        L, _ = loss_one(s, t)
        return L

    # check grad_U[0][0]
    U_p = copy.deepcopy(U); U_p[0][0] += eps
    U_m = copy.deepcopy(U); U_m[0][0] -= eps
    num_gU00 = (L_val(E, U_p) - L_val(E, U_m)) / (2*eps)

    h, s = forward(E, U, i)
    _, p = loss_one(s, t)
    gU, gh = gradients(h, U, p, t)
    ana_gU00 = gU[0][0]

    # check grad_h[0] via E[i][0]
    E_p = copy.deepcopy(E); E_p[i][0] += eps
    E_m = copy.deepcopy(E); E_m[i][0] -= eps
    num_gh0 = (L_val(E_p, U) - L_val(E_m, U)) / (2*eps)
    ana_gh0 = gh[0]

    print(f"  grad_U[0][0]: analytic={ana_gU00:.8f}  numeric={num_gU00:.8f}  diff={abs(ana_gU00-num_gU00):.2e}")
    print(f"  grad_h[0]  : analytic={ana_gh0:.8f}  numeric={num_gh0:.8f}  diff={abs(ana_gh0-num_gh0):.2e}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    run_checks()

  
    SEED    = 42
    DIM     = 10
    WINDOW  = 2
    LR      = 0.05
    EPOCHS  = 100


    sentences = preprocess(AMHARIC_CORPUS)
    w2i, i2w  = build_vocab(sentences)
    pairs     = make_pairs(sentences, w2i, WINDOW)
    V         = len(w2i)

    print("=" * 50)
    print("TRAINING – Amharic Skip-gram")
    print("=" * 50)
    print(f"  Vocabulary size   : {V}")
    print(f"  Training pairs    : {len(pairs)}")
    print(f"  Embedding dim     : {DIM}")
    print(f"  Context window    : {WINDOW}")
    print(f"  Learning rate     : {LR}")
    print(f"  Epochs            : {EPOCHS}")
    print(f"  Random seed       : {SEED}")

    E = init_matrix(V, DIM, SEED)
    U = init_matrix(DIM, V, SEED + 1)

    train(E, U, pairs, LR, EPOCHS)

    print("\n" + "=" * 50)
    print("NEAREST-NEIGHBOR QUERIES (from E)")
    print("=" * 50)
    query_words = ["ልጁ", "ምግብ", "ቡና", "ወደ", "ሄደ"]
    for qw in query_words:
        nearest_neighbors(E, w2i, i2w, qw)

    print("""
Note: With a small corpus (~60 sentences) the learned embeddings
reflect co-occurrence statistics rather than deep semantics.
Lower training loss confirms the model is fitting the data, but
does not guarantee meaningful semantic neighbours.
""")


    print("=" * 50)
    print("SAVE / LOAD CHECK")
    print("=" * 50)
    MODEL_PATH = "amharic_skipgram.json"
    save_model(MODEL_PATH, w2i, i2w, E, U)
    w2i2, i2w2, E2, U2 = load_model(MODEL_PATH)

    # verify nearest neighbours are identical after reload
    qi = w2i["ልጁ"]
    sims_before = sorted([(cosine(E[qi], E[w2i[w]]), w) for w in w2i if w != "ልጁ"], reverse=True)[:3]
    sims_after  = sorted([(cosine(E2[w2i2["ልጁ"]], E2[w2i2[w]]), w) for w in w2i2 if w != "ልጁ"], reverse=True)[:3]
    assert sims_before == sims_after, "Neighbour rankings changed after reload"
    print("  Nearest-neighbour rankings consistent after reload  PASSED")

    # ── Numerical gradient check ──────────────
    print("\n" + "=" * 50)
    print("NUMERICAL GRADIENT CHECK")
    print("=" * 50)
    E_small = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    U_small = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    numerical_grad_check(E_small, U_small, i=0, t=1)

if __name__ == "__main__":
    main()
