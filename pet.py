import torch
import ltn
import random
import matplotlib.pyplot as plt

object_grounding_dim = (5, 4)
_pred_model_sizes = [4, 16, 1]
stage_epoch = 2000
stages = 3 # number of sample by feature size

And  = ltn.Connective(ltn.fuzzy_ops.AndProd())
Impl = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Not  = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")


def _predicate_model():
    layers = []
    for in_f, out_f in zip(_pred_model_sizes, _pred_model_sizes[1:]):
        layers.append(torch.nn.Linear(in_f, out_f))
        layers.append(torch.nn.ReLU())
    layers = layers[:-1]  # remove final ReLU
    layers.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*layers)

def set_global_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Groundings
# ============================================================

def make_groundings():
    Normal_Birds = torch.randn(object_grounding_dim)
    Penguins     = torch.randn(object_grounding_dim)
    Cows         = torch.randn(object_grounding_dim)
    Animals = torch.cat([Normal_Birds, Penguins, Cows], dim=0)

    return {
        "Normal_Birds": Normal_Birds,
        "Penguins": Penguins,
        "Cows": Cows,
        "Animals": Animals
    }

# ============================================================
# Build LTN Predicates
# ============================================================

def build_predicates():
    is_bird     = ltn.Predicate(_predicate_model())
    is_penguin  = ltn.Predicate(_predicate_model())
    can_fly     = ltn.Predicate(_predicate_model())
    return is_bird, is_penguin, can_fly


# ============================================================
# Build the PET rules (Appendix A)
# ============================================================

def build_rules(preds, g):
    """
    KB consists of the following 8 rules
    each rule is a zero-arity lambda function that returns the grounding of statement directly,
    evaluated with appropriate objects
    """
    is_bird, is_penguin, can_fly = preds
    xNB = ltn.Variable("x", g["Normal_Birds"])
    xC = ltn.Variable("x", g["Cows"])
    xA = ltn.Variable("x", g["Animals"])
    xP = ltn.Variable("x", g["Penguins"])
    xNon_B = ltn.Variable("x", torch.cat([g["Normal_Birds"], g["Cows"]], dim=0))

    return [
        # 1 normal birds are birds
        lambda: Forall(xNB, is_bird(xNB)),
        # 2 cows are not birds
        lambda: Forall(xC, Not(is_bird(xC))),
        # 3 birds can fly
        lambda: Forall(xA, Impl(is_bird(xA), can_fly(xA))),
        # 4 non-birds cannot fly
        lambda: Forall(xA, Impl(Not(is_bird(xA)), Not(can_fly(xA)))),
        # 5 penguins are penguins
        lambda: Forall(xP, is_penguin(xP)),
        # 6 non-penguins are not penguins
        lambda: Forall(xNon_B, Not(is_penguin(xNon_B))),
        # 7 penguins are birds
        lambda: Forall(xA, Impl(is_penguin(xA), is_bird(xA))),
        # 8 penguins do not fly
        lambda: Forall(xA, Impl(is_penguin(xA), Not(can_fly(xA))))
    ]

# ============================================================
# Training utilities
# ============================================================

def satisfaction(rule_fns):
    """
    simply compute the mean satisfaction of all rules
    """
    sats = [r().value for r in rule_fns]
    return torch.mean(torch.stack(sats))

def train_stage(preds, rules, grounding, curriculum=None):
    params = []
    for p in preds:
        params += list(p.parameters())

    opt = torch.optim.Adam(params)

    g1s, g2s, g3s, g4s = [], [], [], []
    for _ in range(stage_epoch * stages if curriculum == 'baseline' else stage_epoch):
        g1, g2, g3, g4 = evaluate(preds, grounding)
        g1s.append(g1)
        g2s.append(g2)
        g3s.append(g3)
        g4s.append(g4)

        opt.zero_grad()
        loss = 1 - satisfaction(rules)
        loss.backward()
        opt.step()

    return g1s, g2s, g3s, g4s
# ============================================================
# Evaluation (matches Table 1 metrics)
# ============================================================

def evaluate(preds, g):
    """
    predicates of interest according to Table 1
    """

    is_bird, _, can_fly = preds

    NB = ltn.Variable("xNB", g["Normal_Birds"])
    PG = ltn.Variable("xPG", g["Penguins"])

    g1 = Forall(NB, is_bird(NB)).value
    g2 = Forall(PG, is_bird(PG)).value
    g3 = Forall(NB, can_fly(NB)).value
    g4 = Forall(PG, Not(can_fly(PG))).value

    return [float(g1), float(g2), float(g3), float(g4)]

    #     {
    #     "is_bird(Normal_Birds)": float(q1),
    #     "is_bird(Penguins)": float(q2),
    #     "can_fly(Birds)": float(q3),
    #     "not(can_fly(Penguins))": float(q4)
    # }

def plot_groundings(g1s, g2s, g3s, g4s, title):
    plt.plot(g1s, label='is_bird(Normal_Birds)')
    plt.plot(g2s, label='is_bird(Penguins)')
    plt.plot(g3s, label='can_fly(Birds)')
    plt.plot(g4s, label='not(can_fly(Penguins))')
    plt.xlabel('Epochs')
    plt.ylabel('Satisfaction Level')
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.show()

# ============================================================
# Baseline curriculum
# ============================================================

def run_baseline():
    print("\n=== BASELINE ===")
    g = make_groundings()
    preds = build_predicates()
    rules = build_rules(preds, g)
    g1s, g2s, g3s, g4s = train_stage(preds, rules, g,'baseline')
    plot_groundings(g1s, g2s, g3s, g4s, title="Baseline")
    print("Baseline Results:")
    print(f"is_bird(Normal_Birds) {g1s[-1]:.4f}")
    print(f"is_bird(Penguins) {g2s[-1]:.4f}")
    print(f"can_fly(Birds) {g3s[-1]:.4f}")
    print(f"not(can_fly(Penguins)) {g4s[-1]:.4f}")

# ============================================================
# Random 3-stage curriculum (with recall)
# ============================================================

def run_random_curriculum():
    print("\n=== RANDOM CURRICULUM ===")
    g = make_groundings()
    preds = build_predicates()
    all_rules = build_rules(preds, g)

    random.shuffle(all_rules)
    stages = [all_rules[:3], all_rules[3:5], all_rules[5:]]

    recall = []

    for stage in stages:
        stage_rules = stage + random.sample(recall, k=min(2, len(recall)))
        train_stage(preds, stage_rules) # TODO: return value lists
        recall += stage

    return evaluate(preds, g)

# ============================================================
# Run both experiments
# ============================================================

# TODO: average over multiple seeds
set_global_seed(42)
run_baseline()
# run_random_curriculum()


