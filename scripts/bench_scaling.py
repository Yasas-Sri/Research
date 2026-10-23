"""CPU scaling check (AGENTS M5): forward wall-clock and memory vs L for the mixer kernels.

    python scripts/bench_scaling.py   ->  results/scaling.csv, results/scaling_env.json

Kernels only, tiny-config shapes, B = 1, float32, torch.no_grad, fixed threads. Each (method, L) runs in a fresh
subprocess (`--worker <method> <L>`) so peak RSS can be measured per run; the whole sweep is repeated SWEEPS times.
Evidence for Chapter 3, not a benchmark claim: absolute times compare implementations of different kinds
(Python einsum vs fused C++ SDPA), so the scaling exponents are the comparable quantity.
"""
import csv
import datetime
import json
import math
import os
import pathlib
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
LS = [256, 512, 1024, 2048, 4096]
METHODS = ["dense_swa", "dense_sdpa", "swa", "ssd"]
QUADRATIC = {"dense_swa", "dense_sdpa"}
THREADS, WARMUP, REPEATS, SWEEPS = 4, 3, 10, 3
BUDGET_GIB = 2.0
H_A, D_H, W = 4, 64, 128  # attention: heads, head dim, window (tiny config)
H_S, P, N, T = 8, 64, 64, 64  # SSD: heads, head dim, state dim, chunk (tiny config)
FOOTER = [
    "# Kernels only: swa_efficient / F.scaled_dot_product_attention / ssd_efficient on pre-built q,k,v or x,dt,A,B,C,D.",
    "# Projections (W_q/W_k/W_v/W_o, in_proj/out_proj, conv) are excluded: they add the same O(L*d^2) term to every",
    "# method and would hide the dense O(L^2) exponent at small L.",
    "# dense_swa = swa_efficient(w=L): (L, 2L) scores, half masked (~2x the needed work). dense_sdpa = fused CPU kernel.",
    "# max_numel = largest single tensor produced by any aten op (not total live memory; blind to fused-kernel internals).",
    "# peak_rss_delta_mib = rise in ru_maxrss over one forward in a fresh process (includes allocator caching; ~MiB noise).",
]


def maxrss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KiB


def worker(method, L):
    """Runs in a fresh process: one (method, L). Prints one JSON line."""
    import torch
    import torch.nn.functional as F

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
    from conftest import MaxNumel  # the same recorder the no-(L, L) tests use
    from hybrid.attention import swa_efficient
    from hybrid.ssm import ssd_efficient

    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    if method == "ssd":
        # weak decay as in tests/test_ssm.py: chunk sums of dt·A stay ~ −10, so no float32 denormals slow the timing
        args = (torch.randn(1, L, H_S, P), F.softplus(torch.randn(1, L, H_S) - 2), -torch.empty(H_S).uniform_(0.5, 2),
                torch.randn(1, L, N), torch.randn(1, L, N), torch.ones(H_S))
        fn = lambda: ssd_efficient(*args, T)
    elif method == "dense_sdpa":
        q, k, v = (torch.randn(1, H_A, L, D_H) for _ in range(3))  # SDPA layout (B, H, L, D): no transpose timed
        fn = lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        q, k, v = (torch.randn(1, L, H_A, D_H) for _ in range(3))
        w = L if method == "dense_swa" else W
        fn = lambda: swa_efficient(q, k, v, w)

    with torch.no_grad():
        rss0 = maxrss_mib()
        fn()  # first warm-up sets the RSS high-water mark; every call is identical
        rss = maxrss_mib() - rss0
        for _ in range(WARMUP - 1):
            fn()
        times = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        with MaxNumel() as rec:  # separate, untimed call
            fn()
    print(json.dumps({"median_ms": statistics.median(times) * 1e3, "max_numel": rec.max, "peak_rss_delta_mib": rss}))


def estimate_gib(method, L, prev_rss_mib):
    """Peak-memory estimate before running: measured RSS rise at L/2 scaled by the expected growth
    (x4 quadratic, x2 linear); for dense methods also an analytic floor of ~3 live float32 score tensors."""
    est = (prev_rss_mib or 0) / 1024 * (4 if method in QUADRATIC else 2)
    if method in QUADRATIC:
        est = max(est, 3 * 4 * H_A * L * (2 * L if method == "dense_swa" else L) / 2**30)
    return est


def slope(points):
    """Least-squares slope of log(time) vs log(L)."""
    xs, ys = [math.log(L) for L, _ in points], [math.log(t) for _, t in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def main():
    import torch

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for sweep in range(SWEEPS):
        prev = {}
        for L in LS:
            for m in METHODS:
                est = estimate_gib(m, L, prev.get(m))
                row = {"sweep": sweep, "method": m, "L": L, "est_gib": round(est, 3)}
                if est > BUDGET_GIB:
                    row["status"] = f"skipped (est. {est:.1f} GiB)"
                    prev[m] = est * 1024  # larger L extrapolate from this estimate, not the last measured run
                else:
                    r = subprocess.run([sys.executable, __file__, "--worker", m, str(L)], capture_output=True, text=True, check=True)
                    res = json.loads(r.stdout.strip().splitlines()[-1])
                    prev[m] = res["peak_rss_delta_mib"]
                    row |= {"status": "ok", "median_ms": round(res["median_ms"], 4), "max_numel": res["max_numel"],
                            "max_numel_mib": round(res["max_numel"] * 4 / 2**20, 3),
                            "peak_rss_delta_mib": round(res["peak_rss_delta_mib"], 1)}
                rows.append(row)
                print(f"sweep {sweep} {m:>10} L={L:>5}  {row.get('median_ms', row['status'])}", flush=True)

    cols = ["sweep", "method", "L", "status", "median_ms", "max_numel", "max_numel_mib", "peak_rss_delta_mib", "est_gib"]
    with open(out / "scaling.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
        f.write("\n".join(FOOTER) + "\n")

    # exponent per sweep over the largest three L that ran; summary = median and min-max across sweeps
    exps = {}
    for m in METHODS:
        per_sweep = []
        for s in range(SWEEPS):
            pts = [(r["L"], r["median_ms"]) for r in rows if r["method"] == m and r["sweep"] == s and r["status"] == "ok"][-3:]
            per_sweep.append({"exponent": slope(pts) if len(pts) >= 2 else None, "L_used": [L for L, _ in pts]})
        e = [p["exponent"] for p in per_sweep if p["exponent"] is not None]
        exps[m] = {"median": statistics.median(e), "min": min(e), "max": max(e), "per_sweep": per_sweep} if e else None

    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    env = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "cpu_model": cpu, "logical_cpus": os.cpu_count(),
           "torch_threads": THREADS, "python": platform.python_version(), "torch": torch.__version__, "os": platform.platform(),
           "warmup": WARMUP, "repeats": REPEATS, "sweeps": SWEEPS, "memory_budget_gib": BUDGET_GIB, "dtype": "float32", "batch": 1,
           "shapes": {"attention": {"heads": H_A, "head_dim": D_H, "swa_window": W},
                      "ssd": {"heads": H_S, "head_dim": P, "d_state": N, "chunk": T}},
           "exponent_fit": "least-squares slope of log(median_ms) vs log(L), largest three L that ran, per sweep",
           "exponents": exps, "notes": [l.lstrip("# ") for l in FOOTER]}
    (out / "scaling_env.json").write_text(json.dumps(env, indent=2) + "\n")

    # printout: median over sweeps of each sweep's median time
    print(f"\n{cpu} | {THREADS} threads | torch {torch.__version__} | Python {platform.python_version()}")
    print("median ms (median of the per-sweep medians)\n" + f"{'L':>6}" + "".join(f"{m:>13}" for m in METHODS))
    for L in LS:
        cells = []
        for m in METHODS:
            ts = [r["median_ms"] for r in rows if r["method"] == m and r["L"] == L and r["status"] == "ok"]
            cells.append(f"{statistics.median(ts):13.3f}" if ts else f"{'skipped':>13}")
        print(f"{L:>6}" + "".join(cells))
    print("\nlargest tensor MiB (max_numel) / peak RSS rise MiB, sweep 0\n" + f"{'L':>6}" + "".join(f"{m:>17}" for m in METHODS))
    for L in LS:
        rs = {r["method"]: r for r in rows if r["sweep"] == 0 and r["L"] == L}
        print(f"{L:>6}" + "".join(f"{rs[m]['max_numel_mib']:>9.2f} /{rs[m]['peak_rss_delta_mib']:>6.1f}" if rs[m]["status"] == "ok"
                                  else f" {rs[m]['status']:>16}" for m in METHODS))
    print(f"\nscaling exponent (log time vs log L, largest 3 L), median [min-max] over {SWEEPS} sweeps")
    for m in METHODS:
        e = exps[m]
        print(f"{m:>10}: " + (f"{e['median']:.2f} [{e['min']:.2f}-{e['max']:.2f}]  L={e['per_sweep'][0]['L_used']}" if e
                              else "n/a (fewer than 2 L values ran)"))
    print("\n" + "\n".join(FOOTER))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        main()
