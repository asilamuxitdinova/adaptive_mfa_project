"""
Adaptive MFA selection demo, based on Nag, Roy & Dasgupta (IEEE SSCI 2015).

At each authentication trigger the system:
  1. senses the context: device, network medium, light (lux), noise (dB), motion
  2. filters out factors that cannot work right now (no hardware, too dark, too noisy, moving)
  3. scores every candidate set of 1-3 factors using the paper's Table II trust values
  4. picks a high-trust, small set that is different from the previous selection

It compares three strategies, as the paper's simulation does:
  - Adaptive: senses the environment and avoids repeating the previous set
  - Optimal:  highest trust only; ignores the environment and history (static)
  - Random:   any available factors, chosen at random

Run:  python3 adaptive_mfa_demo.py [--seed N]
Only the Python standard library is needed.
"""
import argparse
import random
from itertools import combinations

DEVICES = ["FD", "PD", "HD"]          # fixed (desktop), portable (laptop), handheld (phone)
MEDIA = ["WI", "WL", "CL"]            # wired, wireless, cellular
DEVICE_NAMES = {"FD": "Desktop", "PD": "Laptop", "HD": "Phone"}
MEDIUM_NAMES = {"WI": "Wired", "WL": "Wi-Fi", "CL": "Cellular"}

# Table II: calibrated trust values per device (FD, PD, HD) and per medium (WI, WL, CL)
TRUST = {
    "Face (lip)":       (2.02, 1.67, 1.00, 1.95, 1.92, 0.87),
    "Face (eye)":       (2.15, 1.79, 1.07, 2.08, 2.04, 0.93),
    "Face (brow)":      (1.84, 1.52, 0.92, 1.78, 1.74, 0.80),
    "Iris":             (2.17, 1.79, 1.07, 2.09, 2.05, 0.94),
    "Fingerprint":      (2.12, 1.76, 1.05, 2.04, 2.01, 0.92),
    "Palm print":       (2.07, 1.71, 1.03, 1.99, 1.96, 0.89),
    "Hand geometry":    (2.05, 1.70, 1.02, 1.98, 1.94, 0.89),
    "Password":         (1.76, 1.26, 1.13, 1.76, 1.43, 1.26),
    "CAPTCHA":          (1.44, 0.87, 0.87, 1.00, 0.50, 0.50),
    "SMS code":         (0.63, 0.78, 0.63, 1.76, 1.64, 1.48),
    "Voice":            (1.95, 1.61, 0.97, 1.88, 1.85, 0.84),
    "Keystroke":        (1.84, 1.52, 0.91, 1.77, 1.74, 0.80),
    "Mouse dynamics":   (0.99, 0.96, 0.92, 0.92, 0.82, 0.86),
}

# Hardware each factor needs, and what each device has (an assumption for the demo)
NEEDS_HARDWARE = {
    "Face (lip)": "camera", "Face (eye)": "camera", "Face (brow)": "camera",
    "Iris": "iris scanner", "Fingerprint": "fingerprint reader",
    "Palm print": "hand scanner", "Hand geometry": "hand scanner",
    "Voice": "microphone", "Keystroke": "keyboard", "Mouse dynamics": "mouse",
}
HARDWARE = {
    "FD": {"camera", "microphone", "keyboard", "mouse", "fingerprint reader", "iris scanner", "hand scanner"},
    "PD": {"camera", "microphone", "keyboard", "mouse", "fingerprint reader"},
    "HD": {"camera", "microphone", "keyboard", "fingerprint reader"},
}

# Table IV: surrounding conditions each factor depends on
NEEDS_LIGHT = {"Face (lip)", "Face (eye)", "Face (brow)", "Iris"}       # light: 100-1000 lux
NEEDS_QUIET = {"Voice"}                                                 # background noise
NEEDS_STILL = {"Face (lip)", "Face (eye)", "Face (brow)", "Iris", "Voice"}  # motion

LUX_RANGE = (100, 1000)   # paper: common light level 100-1000 lux
MAX_NOISE_DB = 60         # demo threshold for usable voice recognition
U_MAX = 3.97              # paper: highest trust value, used as the upper bound
REQUIRED_TRUST = 2.5      # minimum trust a selection must reach (demo setting)
SIZE_PENALTY = 0.15       # objective 1: fewer factors is better
REPEAT_PENALTY = 1.0      # objective 2: overlap with the previous selection is worse
NEAR_BEST = 0.10          # sets this close to the best score are all acceptable


def context_trust(factor, device, medium):
    """Trust of one factor in this device + medium: the mean of its device and medium values."""
    t = TRUST[factor]
    return (t[DEVICES.index(device)] + t[3 + MEDIA.index(medium)]) / 2


def set_trust(factors, device, medium):
    return min(U_MAX, sum(context_trust(f, device, medium) for f in factors))


def has_hardware(factor, device):
    need = NEEDS_HARDWARE.get(factor)
    return need is None or need in HARDWARE[device]


def environment_ok(factor, ctx):
    if factor in NEEDS_LIGHT and not LUX_RANGE[0] <= ctx["lux"] <= LUX_RANGE[1]:
        return False
    if factor in NEEDS_QUIET and ctx["noise_db"] > MAX_NOISE_DB:
        return False
    if factor in NEEDS_STILL and ctx["moving"]:
        return False
    return True


def candidate_sets(factors, max_size=3):
    for k in range(1, max_size + 1):
        yield from combinations(sorted(factors), k)


def objective1(s, ctx):
    """High trust with few factors."""
    return set_trust(s, ctx["device"], ctx["medium"]) - SIZE_PENALTY * len(s)


def select_adaptive(ctx, previous, rng):
    usable = [f for f in TRUST if has_hardware(f, ctx["device"]) and environment_ok(f, ctx)]
    scored = []
    for s in candidate_sets(usable):
        if set_trust(s, ctx["device"], ctx["medium"]) < REQUIRED_TRUST:
            continue
        if previous and set(s) == set(previous):
            continue  # objective 2: never repeat the previous set exactly
        overlap = len(set(s) & set(previous)) / len(set(s) | set(previous)) if previous else 0
        # Markov property: only the previous selection (X_t-1) matters, not the full history
        scored.append((objective1(s, ctx) - REPEAT_PENALTY * overlap, s))
    best = max(score for score, _ in scored)
    # pick randomly among near-best sets, so an attacker cannot predict the exact choice
    return rng.choice([s for score, s in scored if score >= best - NEAR_BEST])


def select_optimal(ctx):
    """Static: highest trust only, ignores environment and history."""
    usable = [f for f in TRUST if has_hardware(f, ctx["device"])]
    ok = [s for s in candidate_sets(usable) if set_trust(s, ctx["device"], ctx["medium"]) >= REQUIRED_TRUST]
    return max(ok, key=lambda s: (objective1(s, ctx), s))


def select_random(ctx, rng):
    usable = [f for f in TRUST if has_hardware(f, ctx["device"])]
    return tuple(sorted(rng.sample(usable, 2)))


def effective_trust(s, ctx):
    """Factors that cannot work in this environment contribute nothing (they would fail)."""
    working = [f for f in s if environment_ok(f, ctx)]
    return set_trust(working, ctx["device"], ctx["medium"])


SCENARIOS = [
    dict(device="FD", medium="WI", lux=400, noise_db=35, moving=False, note="office desk"),
    dict(device="FD", medium="WI", lux=400, noise_db=35, moving=False, note="same office, re-check"),
    dict(device="PD", medium="WL", lux=50,  noise_db=40, moving=False, note="dark room"),
    dict(device="PD", medium="WL", lux=300, noise_db=75, moving=False, note="noisy cafe"),
    dict(device="HD", medium="CL", lux=800, noise_db=70, moving=True,  note="walking outside"),
    dict(device="HD", medium="CL", lux=800, noise_db=70, moving=True,  note="still walking, re-check"),
    dict(device="HD", medium="WL", lux=20,  noise_db=30, moving=False, note="bedroom at night"),
    dict(device="FD", medium="WI", lux=400, noise_db=35, moving=False, note="back at the office"),
]


def fmt(s, ctx):
    return " + ".join(f if environment_ok(f, ctx) else f"{f} (FAILS)" for f in s)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=7, help="random seed (default 7)")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    totals = {"Adaptive": 0.0, "Optimal": 0.0, "Random": 0.0}
    failures = {k: 0 for k in totals}
    history = {k: [] for k in totals}
    previous = ()

    for t, ctx in enumerate(SCENARIOS, 1):
        picks = {
            "Adaptive": select_adaptive(ctx, previous, rng),
            "Optimal": select_optimal(ctx),
            "Random": select_random(ctx, rng),
        }
        previous = picks["Adaptive"]

        print(f"\nTrigger t{t}: {DEVICE_NAMES[ctx['device']]} on {MEDIUM_NAMES[ctx['medium']]}, "
              f"{ctx['lux']} lux, {ctx['noise_db']} dB, {'moving' if ctx['moving'] else 'still'}  ({ctx['note']})")
        for name, s in picks.items():
            eff = effective_trust(s, ctx)
            ok = eff >= REQUIRED_TRUST
            totals[name] += eff
            failures[name] += not ok
            history[name].append(frozenset(s))
            print(f"  {name:<9} trust {eff:4.2f} {'ok  ' if ok else 'FAIL'}  {fmt(s, ctx)}")

    n = len(SCENARIOS)
    print("\n" + "=" * 72)
    print(f"{'Strategy':<10}{'Avg trust':>10}{'Failed logins':>15}{'Repeats (same set twice in a row)':>36}")
    for name in totals:
        repeats = sum(a == b for a, b in zip(history[name], history[name][1:]))
        print(f"{name:<10}{totals[name] / n:>10.2f}{failures[name]:>15}{repeats:>36}")
    print("\nAdaptive senses light, noise and motion, so it never picks a factor that would fail,")
    print("and it never asks for the same combination twice in a row.")


if __name__ == "__main__":
    main()
