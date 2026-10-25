"""Headless heavy-traffic run, for terminals, CI, and reproducible demos.

Example:
    python -m cldb.simulate --qps 120 --seconds 60 --json experiments/results/live_sim.json
"""
import argparse
import json
import os
import time
from typing import Optional

from cldb.traffic import HeavyTrafficSimulator, TrafficConfig


def _fmt_actions(rates: dict) -> str:
    return "  ".join(f"{name}={rate * 100:5.1f}%" for name, rate in rates.items())


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Headless CLDB heavy-traffic simulation")
    parser.add_argument("--qps", type=float, default=120.0)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--steps-per-phase", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--cwil", action="store_true")
    parser.add_argument("--no-ewc", action="store_true")
    parser.add_argument("--json", default=None, help="write the final snapshot here")
    args = parser.parse_args(argv)

    sim = HeavyTrafficSimulator(TrafficConfig(
        qps=args.qps, steps_per_phase=args.steps_per_phase,
        seed=args.seed, cwil=args.cwil, ewc=not args.no_ewc,
    ))
    print("=" * 82)
    print(f"CLDB heavy-traffic run · target {args.qps:.0f} q/s · "
          f"{args.steps_per_phase} steps/phase · {args.seconds:.0f}s")
    print("=" * 82)
    sim.start()
    started = time.time()
    try:
        while time.time() - started < args.seconds:
            time.sleep(min(args.interval, max(0.2, args.seconds)))
            if time.time() - started >= args.seconds:
                break
            s = sim.snapshot()
            a = s["agent"]
            t = s["traffic"]
            print(f"[{time.time() - started:5.1f}s] "
                  f"qps={t['qps_actual']:6.1f}  queries={t['total_queries']:7d}  "
                  f"phase=T{s['phase']['task']} {s['phase']['name'][:18]:18s}  "
                  f"eps={a['epsilon']:.3f}  reward_ma={a['reward_ma']:+.3f}  "
                  f"updates={a['updates']:5d}  drift={a['drift_events']:2d}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        sim.stop()

    final = sim.snapshot()
    print("\n" + "=" * 82)
    print("FINAL SUMMARY")
    print("=" * 82)
    t = final["traffic"]
    a = final["agent"]
    print(f"queries={t['total_queries']}  actual_qps={t['qps_actual']:.1f}  "
          f"updates={a['updates']}  replay={a['replay_size']}  "
          f"epsilon={a['epsilon']:.3f}")
    print(f"mean_reward={a['mean_reward']:+.3f}  reward_ma={a['reward_ma']:+.3f}  "
          f"drift_events={a['drift_events']}  consolidations={a['consolidations']}")
    print(f"actions: {_fmt_actions(t['action_rates'])}")
    print("\nper phase (higher reward is better):")
    for p in final["phases"]:
        if p["steps"] == 0:
            continue
        flag = "conflict" if p["is_conflict"] else "        "
        print(f"  T{p['task']} {p['name'][:24]:24s} {flag}  "
              f"steps={p['steps']:6d}  reward={p['mean_reward']:+.3f}  "
              f"KEEP={p['keep_rate'] * 100:5.1f}%  CREATE={p['create_rate'] * 100:5.1f}%  "
              f"DROP={p['drop_rate'] * 100:5.1f}%")
    if final["status"]["error"]:
        print(f"\nERROR: {final['status']['error']}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"summary": sim.summary(), "snapshot": final}, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
