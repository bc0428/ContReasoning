import torch
import ltn
import random
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

_pred_model_sizes = [4, 16, 1]
def _predicate_model():
    layers = []
    for in_f, out_f in zip(_pred_model_sizes, _pred_model_sizes[1:]):
        layers.append(torch.nn.Linear(in_f, out_f))
        layers.append(torch.nn.ReLU())
    layers = layers[:-1]  # remove final ReLU
    layers.append(torch.nn.Sigmoid())
    return torch.nn.Sequential(*layers)

def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# Groundings
# ============================================================

GROUNDING_SEED = 0  # any fixed integer
object_grounding_dim = (5, 4)
def make_groundings():
    # Local RNG, independent of torch.manual_seed(...)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(GROUNDING_SEED)

    Normal_Birds = torch.randn(object_grounding_dim, generator=gen)
    Penguins     = torch.randn(object_grounding_dim, generator=gen)
    Cows         = torch.randn(object_grounding_dim, generator=gen)
    Animals      = torch.cat([Normal_Birds, Penguins, Cows], dim=0)

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
And  = ltn.Connective(ltn.fuzzy_ops.AndProd())
Impl = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Not  = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")

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
    xNon_P = ltn.Variable("x", torch.cat([g["Normal_Birds"], g["Cows"]], dim=0))

    return [
        # 0 normal birds are birds
        lambda: Forall(xNB, is_bird(xNB)),
        # 1 cows are not birds
        lambda: Forall(xC, Not(is_bird(xC))),
        # 2 birds can fly
        lambda: Forall(xA, Impl(is_bird(xA), can_fly(xA))),
        # 3 non-birds cannot fly
        lambda: Forall(xA, Impl(Not(is_bird(xA)), Not(can_fly(xA)))),
        # 4 penguins are penguins
        lambda: Forall(xP, is_penguin(xP)),
        # 5 non-penguins are not penguins
        lambda: Forall(xNon_P, Not(is_penguin(xNon_P))),
        # 6 penguins are birds
        lambda: Forall(xA, Impl(is_penguin(xA), is_bird(xA))),
        # 7 penguins do not fly
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

stage_epoch = 2000
stages = 3 # number of sample by feature size
def train_stage(preds, rules, grounding, baseline=False, optimizer=None):
    params = []
    for p in preds:
        params += list(p.parameters())

    opt = torch.optim.Adam(params) if optimizer is None else optimizer

    g1s, g2s, g3s, g4s = [], [], [], []
    for _ in range(stage_epoch * stages if baseline else stage_epoch):
        g1, g2, g3, g4 = evaluate(preds, grounding)
        g1s.append(g1)
        g2s.append(g2)
        g3s.append(g3)
        g4s.append(g4)

        opt.zero_grad()
        loss = 1 - satisfaction(rules)
        loss.backward()
        opt.step()

    return g1s, g2s, g3s, g4s, opt
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

def plot_groundings(g1s, g2s, g3s, g4s, title, stage_periods=None):
    plt.plot(g1s, label='is_bird(Normal_Birds)')
    plt.plot(g2s, label='is_bird(Penguins)')
    plt.plot(g3s, label='can_fly(Birds)')
    plt.plot(g4s, label='not(can_fly(Penguins))')
    plt.xlabel('Epochs')
    plt.ylabel('Satisfaction Level')
    plt.title(title)
    plt.legend()
    if stage_periods:
        # compute cumulative boundaries and mark them
        cum = []
        s = 0
        for p in stage_periods:
            s += p
            cum.append(s)
        ymax = plt.ylim()[1]
        for idx, b in enumerate(cum[:-1]):
            plt.axvline(x=b, color='k', linestyle='--', alpha=0.6)
            plt.text(b, ymax * 0.95, f'Stage {idx+1}', rotation=90, va='top', ha='right')
    plt.savefig(f"{title}.png")
    plt.show()

# ============================================================
# Baseline curriculum
# ============================================================

def run_baseline(seeds, out_dir='baseline_results'):
    os.makedirs(out_dir, exist_ok=True)

    for seed in seeds:
        # set seed, build models, run training -> returns per-step arrays
        set_global_seed(seed)
        preds = build_predicates()
        g = make_groundings()
        rules = build_rules(preds, g)

        g1s, g2s, g3s, g4s, _ = train_stage(preds, rules, g, baseline=True)

        path = os.path.join(out_dir, f"baseline_seed_{seed}.npz")
        # save each run independently
        np.savez_compressed(path,
                            seed=seed,
                            g1s=np.array(g1s),
                            g2s=np.array(g2s),
                            g3s=np.array(g3s),
                            g4s=np.array(g4s))
        print(f"Saved run {seed} -> {path}")

# ============================================================
# Random 3-stage curriculum (with recall)
# ============================================================

def run_random(seeds, out_dir=None,  original=False):
    os.makedirs(out_dir, exist_ok=True)

    for seed in seeds:
        set_global_seed(seed)
        g = make_groundings()
        preds = build_predicates()
        all_rules = build_rules(preds, g)

        if original:  # following the specific split from the paper: stage 1:(3,4,6,2) 2:(5,1) 3:(7,8)
            print("\n=== RANDOM (ORIGINAL) CURRICULUM ===")
            stages = [[all_rules[i] for i in idxs] for idxs in [[1, 2, 3, 5], [0, 4], [6, 7]]]
            curriculum = "random_original"
        else:
            print("\n=== RANDOM CURRICULUM ===")
            sizes = [4, 2, 2]
            indices = list(range(len(all_rules)))
            random.shuffle(indices)
            it = iter(indices)
            stages = [[all_rules[i] for i in (next(it) for _ in range(s))] for s in sizes]
            curriculum = "random"

        recall = []
        full_g1s, full_g2s, full_g3s, full_g4s = [], [], [], []

        for stage in stages:
            optimizer = None
            stage_rules = stage + recall
            g1s, g2s, g3s, g4s, optimizer = train_stage(preds, stage_rules, g, optimizer=optimizer)
            full_g1s.extend(g1s)
            full_g2s.extend(g2s)
            full_g3s.extend(g3s)
            full_g4s.extend(g4s)

            # recall, one rule per stage added to memory
            recalled_rule = random.choice(stage)
            recall.append(recalled_rule)

            # save per-seed identical format to baseline
        path = os.path.join(out_dir, curriculum + f"_seed_{seed}.npz")
        np.savez_compressed(path,
                            seed=seed,
                            g1s=np.array(full_g1s),
                            g2s=np.array(full_g2s),
                            g3s=np.array(full_g3s),
                            g4s=np.array(full_g4s))
        print(f"Saved run {seed} -> {path}")


        # print final results per stage
        # print("Per-stage final results:")
        # for idx, (r1, r2, r3, r4) in enumerate(stage_final_results):
        #     print(f" Stage {idx+1}: is_bird(Normal_Birds) {r1:.4f}, is_bird(Penguins) {r2:.4f}, can_fly(Birds) {r3:.4f}, not(can_fly(Penguins)) {r4:.4f}")
        #
        # # merge all data points for a complete plot and demarcate stage periods
        # full_g1s, full_g2s, full_g3s, full_g4s = [], [], [], []
        # stage_periods = []
        # for (g1s, g2s, g3s, g4s) in stage_all_results:
        #     full_g1s.extend(g1s)
        #     full_g2s.extend(g2s)
        #     full_g3s.extend(g3s)
        #     full_g4s.extend(g4s)
        #     stage_periods.append(len(g1s))
        #
        # plot_groundings(full_g1s, full_g2s, full_g3s, full_g4s, title=title, stage_periods=stage_periods)


# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('curriculum', type=str, default='baseline', help='Curriculum to run')
    parser.add_argument('seed', type=int, default=None, help='Single seed to run')
    parser.add_argument('out_dir', type=str, default='baseline_results', help='Output directory')
    args = parser.parse_args()

    if args.seed is None:
        raise Exception("Seed missing")
    else:
        if args.curriculum == 'baseline':
            run_baseline(seeds=[args.seed], out_dir=args.out_dir)
        elif args.curriculum == 'random_original':
            run_random(seeds = [args.seed], out_dir=args.out_dir, original=True)
        elif args.curriculum == 'random':
            run_random(seeds = [args.seed], out_dir=args.out_dir)