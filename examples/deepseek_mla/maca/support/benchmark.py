"""Check C500 MLA against an FP32 reference and compare complete calls."""

import argparse
import csv
import math
from statistics import mean, median, stdev
import functools
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import tempfile

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "figures/benchmark_results.json"
IMPLEMENTATIONS = ("triton", "dense", "default")
# Allow direct script execution and explicit pytest collection from any directory.
sys.path.insert(0, str(ROOT))


def reference(q, q_pe, kv, k_pe):
    """Dense MLA with FP32 scores, softmax, and value accumulation."""
    if q.shape[0] > 4:
        return torch.cat([reference(*parts) for parts in zip(*(x.split(4) for x in (q, q_pe, kv, k_pe)))])
    k, k_rope = kv[:, :, 0, :].float(), k_pe[:, :, 0, :].float()
    scale = (q.shape[-1] + q_pe.shape[-1]) ** -0.5
    logits = torch.bmm(q.float(), k.transpose(1, 2))
    logits += torch.bmm(q_pe.float(), k_rope.transpose(1, 2))
    return torch.bmm(torch.softmax(logits * scale, dim=-1), k)


def make_inputs(batch_size, kv_length, seed, profile, device, heads=16):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shapes = ((batch_size, heads, 512), (batch_size, heads, 64), (batch_size, kv_length, 1, 512), (batch_size, kv_length, 1, 64))
    q, q_pe, kv, k_pe = [torch.randn(shape, generator=generator, dtype=torch.float32) for shape in shapes]
    if profile in ("early-peak", "late-peak"):
        q.zero_()
        q_pe.zero_()
        kv[..., 0] = 0
        q[..., 0] = 32
        kv[:, 0 if profile == "early-peak" else -1, :, 0] = 64
    elif profile == "large-v":
        # Uniform attention makes the unnormalized partial exceed the FP16 range.
        q.zero_()
        q_pe.zero_()
        kv[..., ::2] = 2048
        kv[..., 1::2] = -2048
    elif profile != "random":
        raise ValueError(f"unknown input profile {profile!r}")
    return tuple(x.to(device=device, dtype=torch.float16).contiguous() for x in (q, q_pe, kv, k_pe))


def load_baseline(path):
    """Use the baseline source from an explicitly selected TileLang-MetaX checkout."""
    spec = importlib.util.spec_from_file_location("tilelang_metax_mla_baseline", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load a Python baseline from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.flashattn


def check_kernel(name, kernel, batch_size, kv_length, cases, device, trace, heads=16):
    max_error = 0.0
    for profile, seed in cases:
        inputs = make_inputs(batch_size, kv_length, seed, profile, device, heads)
        assert all(t.is_contiguous() for t in inputs)
        expected = reference(*inputs)
        if trace:
            torch.cuda.synchronize()
            print(f"{name}: calling {profile}, seed {seed}", file=sys.stderr, flush=True)
        output = kernel(*inputs)
        if trace:
            torch.cuda.synchronize()
            print(f"{name}: checking {profile}, seed {seed}", file=sys.stderr, flush=True)
        if output.shape != expected.shape or output.dtype != torch.float16:
            raise AssertionError(f"{name} returned an unexpected shape or dtype")
        try:
            torch.testing.assert_close(output.float(), expected, rtol=2e-3, atol=2e-3)
        except AssertionError as error:
            raise AssertionError(f"{name}, profile={profile}, seed={seed}: {error}") from error
        max_error = max(max_error, (output.float() - expected).abs().max().item())
    if trace:
        print(f"{name}: passed {len(cases)} checks, max absolute error {max_error:g}", file=sys.stderr, flush=True)
    return max_error


def time_call(call, warmup, iterations, clock):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    if clock == "event":
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def make_report_inputs(batch, heads, length, seed, profile):
    """Match the report's GPU FP16 generator, including its stress inputs."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shapes = ((batch, heads, 512), (batch, heads, 64), (batch, length, 1, 512), (batch, length, 1, 64))
    inputs = tuple(torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator) for shape in shapes)
    if profile != "random":
        q, q_pe, kv, _ = inputs
        q.zero_()
        q_pe.zero_()
        if profile == "large-v":
            kv[..., ::2] = 2048
            kv[..., 1::2] = -2048
        elif profile in ("early-peak", "late-peak"):
            q[..., 0] = 32
            kv[..., 0] = 0
            kv[:, 0 if profile == "early-peak" else -1, :, 0] = 64
        else:
            raise ValueError(f"unknown report profile {profile!r}")
    return inputs


def run_report(args, tilelang, triton, build_candidate, schedule, triton_mla):
    """Replay saved configurations with fresh inputs; do not rerun tuning."""
    recipe = json.loads(RESULTS.read_text())
    baseline_hash = hashlib.sha256(args.baseline_file.read_bytes()).hexdigest()
    if baseline_hash != recipe["sources"]["dense_baseline_sha256"]:
        raise ValueError("report mode requires the pinned dense baseline source linked in README")
    keys = list(recipe["results"]) if args.report_shapes is None else args.report_shapes
    environment = {
        "device": torch.cuda.get_device_name(),
        "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
        "triton": triton.__version__,
    }
    for label, command in (
        ("mx_smi", ["mx-smi"]),
        ("scheduler", ["mx-smi", "sgpu", "--show-sched-class"]),
        ("timeslice", ["mx-smi", "sgpu", "--show-timeslice"]),
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            environment[label] = {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except (OSError, subprocess.TimeoutExpired) as error:
            environment[label] = {"unavailable": str(error)}
    sources = {"dense_baseline_sha256": baseline_hash}
    for filename in ("example_mla_decode_maca.py", "support/benchmark.py", "support/triton_reference.py"):
        sources[filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    document = {
        "environment": environment,
        "sources": sources,
        "protocol": recipe["protocol"],
        "configuration_source_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
        "configuration_selection": "Replay published configurations; no tuning performed in this run",
        "requested_shapes": keys,
        "report_complete": False,
        "results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=args.output.name, delete=False) as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            temporary = Path(stream.name)
        temporary.replace(args.output)

    build_baseline = load_baseline(args.baseline_file)
    save()

    def run_shape(key):
        batch, heads, length = map(int, key.split("x"))
        if torch.cuda.mem_get_info()[0] < 4 * 1024**3:
            raise RuntimeError("report mode requires at least 4 GiB of free GPU memory before each shape")
        config = recipe["results"][key]
        if list(schedule(batch, heads, length)) != config["default_config"]:
            raise ValueError(f"{key}: the current default schedule differs from the recorded report")
        row = {name: config[name] for name in ("default_config", "dense_config", "triton_config", "measurement")}
        row.update(validation={}, complete=False)
        document["results"][key] = row
        compiled = build_candidate(batch, length, heads)

        def candidate(*inputs):
            output = torch.empty_like(inputs[0])
            compiled(*inputs, output)
            return output

        tile_h, tile_n, splits = row["dense_config"]
        dense = build_baseline(batch, heads, 1, length, 512, 64, tile_n, tile_h, splits, 1 / 24.0)
        tile_h, tile_n, warps, splits = row["triton_config"]
        functions = {
            "default": candidate,
            "dense": dense,
            "triton": functools.partial(triton_mla, head_tile=tile_h, key_tile=tile_n, warps=warps, splits=splits),
        }
        protocol = recipe["protocol"]
        cases = [("random", seed) for seed in protocol["validation_random_seeds"]]
        cases += [(profile, protocol["seed"]) for profile in protocol["validation_stress_profiles"]]
        for profile, seed in cases:
            inputs = make_report_inputs(batch, heads, length, seed, profile)
            expected = reference(*inputs)
            errors = {}
            for name, function in functions.items():
                output = function(*inputs)
                if output.dtype != torch.float16 or output.shape != expected.shape:
                    raise AssertionError(f"{key}/{name}: unexpected output shape or dtype")
                torch.testing.assert_close(output.float(), expected, atol=protocol["atol"], rtol=protocol["rtol"])
                errors[name] = (output.float() - expected).abs().max().item()
            row["validation"][f"{profile}/{seed}"] = errors
            save()
            del inputs, expected, output
        measure = row["measurement"]
        inputs = make_report_inputs(batch, heads, length, measure["seed"], "random")
        calls = {name: functools.partial(function, *inputs) for name, function in functions.items()}
        orders = list(itertools.permutations(calls))
        if measure["batches"] != len(orders):
            raise ValueError("the recorded three-way report requires all six implementation orders")
        for clock in ("wall", "event"):
            row[clock] = []
            for order in orders:
                row[clock].append({name: time_call(calls[name], measure["warmup"], measure["iterations"], clock) for name in order})
                save()
        row["complete"] = True
        save()
        print(
            f"PASS {key}: "
            + ", ".join(f"{name}={statistics.median(s[name] for s in row['wall']):.2f} us" for name in ("triton", "dense", "default")),
            flush=True,
        )

    for key in keys:
        print(f"Report shape {key}", flush=True)
        try:
            run_shape(key)
        except Exception as error:
            document["failure"] = {"shape": key, "error": str(error)}
            save()
            raise
        gc.collect()
        torch.cuda.empty_cache()
    records = summarize_results(document, keys)
    write_csv(records, args.output.with_suffix(".csv"))
    document["report_complete"] = set(keys) == set(recipe["results"])
    save()
    print(f"Saved {len(keys)} fresh shapes to {args.output} and {args.output.with_suffix('.csv')}", flush=True)


def summarize_results(document, expected_shapes):
    """Refuse missing shapes or incomplete samples instead of filling a chart."""
    rows = document["results"]
    if set(rows) != set(expected_shapes):
        raise ValueError("shape set differs from the requested matrix; partial reports cannot produce the full charts")
    output = []
    for key, row in rows.items():
        if not row.get("complete") or len(row.get("validation", {})) != 5:
            raise ValueError(f"{key}: five completed output checks per implementation are required")
        for errors in row["validation"].values():
            if set(errors) != set(IMPLEMENTATIONS) or any(not math.isfinite(v) or v < 0 for v in errors.values()):
                raise ValueError(f"{key}: missing or invalid validation error record")
        batch, heads, length = map(int, key.split("x"))
        record = {"batch": batch, "heads": heads, "kv_length": length}
        for clock in ("wall", "event"):
            samples = row.get(clock, [])
            if len(samples) != 6 or any(set(s) != set(IMPLEMENTATIONS) for s in samples):
                raise ValueError(f"{key}: six complete three-way {clock} samples are required")
            for name in IMPLEMENTATIONS:
                values = [s[name] for s in samples]
                if any(not math.isfinite(v) or v <= 0 for v in values):
                    raise ValueError(f"{key}/{name}: invalid latency")
                record[f"{name}_{clock}_us"] = median(values)
                record[f"{name}_{clock}_cv"] = stdev(values) / mean(values)
        for name in ("triton", "dense"):
            record[f"speedup_vs_{name}"] = record[f"{name}_wall_us"] / record["default_wall_us"]
        output.append(record)
    return output


def write_csv(records, path):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def render(records, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.labelcolor": "#253041",
            "axes.edgecolor": "#8a97a6",
        }
    )
    series = [
        ("triton", "Triton (saved config)", "#d18c37", ""),
        ("dense", "TileLang-MetaX dense (saved config)", "#627da9", "//"),
        ("default", "C500 TileLang (default)", "#00746e", ""),
    ]
    lookup = {(r["batch"], r["heads"], r["kv_length"]): r for r in records}
    for small in (False, True):
        batches = (1, 2, 4) if small else (64, 128)
        fig, axes = plt.subplots(
            1, len(batches), figsize=(12.4, 4.5), sharey=True, gridspec_kw={"width_ratios": [6, 3, 3] if small else [1, 1]}
        )
        fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.76, wspace=0.09)
        values_all = []
        for ax, b in zip(axes, batches):
            h = 16 if small else 128
            ns = sorted(n for bb, hh, n in lookup if bb == b and hh == h)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e0e6ec", linewidth=0.8)
            for i, (impl, label, color, hatch) in enumerate(series):
                values = [
                    lookup[b, h, n]["dense_wall_us"] / lookup[b, h, n][impl + "_wall_us"]
                    if small
                    else 2 * b * h * n * (576 + 512) / (lookup[b, h, n][impl + "_wall_us"] * 1e6)
                    for n in ns
                ]
                values_all.extend(values)
                bars = ax.bar(
                    [j + (i - 1) * 0.25 for j in range(len(ns))],
                    values,
                    width=0.24,
                    label=label,
                    color=color,
                    hatch=hatch,
                    edgecolor="white",
                    linewidth=0.4,
                )
                if not small or impl != "dense":
                    ax.bar_label(bars, labels=[f"{v:.2f}" if small else f"{v:.1f}" for v in values], padding=3, fontsize=8, rotation=0)
            if small:
                ax.axhline(1, color="#657181", linestyle="--", linewidth=0.8)
            ax.tick_params(length=0, pad=5)
            ax.set(title=f"Batch {b}", xticks=range(len(ns)), xticklabels=[f"{n // 1024}K" for n in ns])
        axes[0].set_ylabel("Speedup over TileLang dense (×)" if small else "Effective throughput (TFLOP/s)")
        axes[0].set_ylim(0, max(values_all) * (1.25 if small else 1.14))
        fig.supxlabel("KV length (tokens)")
        fig.suptitle("Dense MLA decode on C500 · " + ("16 heads" if small else "128 heads"), fontsize=15, y=0.98)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False, fontsize=10)
        stem = "flashmla-maca-small" if small else "flashmla-maca"
        for ext in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{ext}", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def plot_report(args):
    document = json.loads(args.input.read_text())
    expected = json.loads(RESULTS.read_text())["results"]
    try:
        records = summarize_results(document, expected)
    except ValueError as error:
        raise ValueError(str(error)) from error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, args.output_dir / "summary.csv")
    summary = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(), "shapes": len(records), "groups": {}}
    for name, heads in (("small", 16), ("large", 128)):
        group = [r for r in records if r["heads"] == heads]
        summary["groups"][name] = {
            baseline: {
                "range": [min(r[f"speedup_vs_{baseline}"] for r in group), max(r[f"speedup_vs_{baseline}"] for r in group)],
                "wins": sum(r[f"speedup_vs_{baseline}"] > 1 for r in group),
                "count": len(group),
            }
            for baseline in ("triton", "dense")
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(records, args.output_dir)
    print(json.dumps(summary, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "compare", "report", "plot"),
        default="smoke",
        help="smoke: correctness; compare: one recorded shape; report: full matrix; plot: regenerate charts",
    )
    parser.add_argument("--batch-size", type=int, help="Smoke only (default: 1)")
    parser.add_argument("--heads", type=int, help="Smoke only (default: 16)")
    parser.add_argument("--kv-length", type=int, help="Smoke only (default: 2049)")
    parser.add_argument("--seeds", type=int, help="Smoke random seeds (default: 8); also checks three stress inputs")
    parser.add_argument("--baseline-file", type=Path, help="Pinned TileLang-MetaX example_mla_decode.py")
    parser.add_argument("--output", type=Path, help="Fresh comparison/report JSON path; also writes CSV")
    parser.add_argument("--report-shapes", nargs="+", metavar="BxHxN", help="Report subset, e.g. 1x16x2048")
    parser.add_argument("--input", type=Path, help="Plot only: measurements JSON (default: published data)")
    parser.add_argument("--output-dir", type=Path, help="Plot only: figure output directory")
    args = parser.parse_args(argv)
    if args.preset == "plot":
        if any(
            value is not None
            for value in (args.batch_size, args.heads, args.kv_length, args.seeds, args.baseline_file, args.output, args.report_shapes)
        ):
            parser.error("plot accepts only --input and --output-dir")
        args.input = args.input or RESULTS
        args.output_dir = args.output_dir or Path("mla-report-plots")
        return parser, args
    if args.input is not None or args.output_dir is not None:
        parser.error("--input and --output-dir require --preset plot")
    if args.preset == "smoke":
        if args.baseline_file is not None or args.output is not None or args.report_shapes is not None:
            parser.error("baseline, output and report shapes require --preset compare or report")
        args.batch_size = 1 if args.batch_size is None else args.batch_size
        args.heads = 16 if args.heads is None else args.heads
        args.kv_length = 2049 if args.kv_length is None else args.kv_length
        args.seeds = 8 if args.seeds is None else args.seeds
        if args.batch_size < 1 or args.heads < 16 or args.heads % 16 or not 1 <= args.kv_length <= 65536 or args.seeds < 1:
            parser.error("require B >= 1, H a positive multiple of 16, N in [1, 65536], and seeds >= 1")
    else:
        if any(value is not None for value in (args.batch_size, args.heads, args.kv_length, args.seeds)):
            parser.error("comparison/report shapes and validation inputs are fixed by the published measurements")
        if args.preset == "compare":
            if args.report_shapes is not None:
                parser.error("use --preset report to select report shapes")
            args.report_shapes = ["1x16x2048"]
        keys = json.loads(RESULTS.read_text())["results"]
        if args.report_shapes is not None and (
            len(set(args.report_shapes)) != len(args.report_shapes) or any(key not in keys for key in args.report_shapes)
        ):
            parser.error("--report-shapes must be distinct keys from benchmark_results.json")
        if args.baseline_file is None:
            root = Path(os.environ.get("TILELANG_METAX_ROOT", ROOT.parents[2]))
            args.baseline_file = root / "examples/deepseek_mla/example_mla_decode.py"
        if not args.baseline_file.is_file():
            parser.error("set TILELANG_METAX_ROOT to the pinned baseline checkout, or pass --baseline-file")
        args.output = args.output or Path(f"mla-{args.preset}.json")
        if args.output.suffix.lower() != ".json":
            parser.error("--output must have a .json suffix")
        if args.output.exists() or args.output.with_suffix(".csv").exists():
            parser.error("output already exists; choose a new --output path")
    return parser, args


def main():
    parser, args = parse_args()
    if args.preset == "plot":
        plot_report(args)
        return
    if torch is None:
        parser.error("activate the MetaX PyTorch environment; --help does not require GPU packages")
    if not torch.cuda.is_available() or "c500" not in torch.cuda.get_device_name().lower():
        parser.error("this benchmark requires a visible MetaX C500 and PyTorch-MACA")
    try:
        import tilelang
        from example_mla_decode_maca import default_schedule, mla_decode_maca
    except ImportError as error:
        parser.error(f"activate the TileLang-MetaX MACA backend: {error}")
    torch.set_float32_matmul_precision("highest")
    if args.preset != "smoke":
        try:
            import triton
            from triton_reference import run as triton_mla
        except ImportError as error:
            parser.error(f"three-way comparisons require MetaX Triton: {error}")
        run_report(args, tilelang, triton, mla_decode_maca, default_schedule, triton_mla)
        return

    compiled = mla_decode_maca(args.batch_size, args.kv_length, args.heads)

    def candidate(*inputs):
        output = torch.empty_like(inputs[0])
        compiled(*inputs, output)
        return output

    cases = [("random", seed) for seed in range(args.seeds)] + [("early-peak", 0), ("late-peak", 0), ("large-v", 0)]
    error = check_kernel("C500", candidate, args.batch_size, args.kv_length, cases, torch.device("cuda"), True, args.heads)
    print(f"PASS: {len(cases)} full-output checks; max absolute error {error:.6g}")


def pytest_generate_tests(metafunc):
    if metafunc.function.__name__ == "test_mla":
        metafunc.parametrize(
            "heads,length,head_tile,splits",
            [(16, 31, 16, 1), (16, 65, 16, 2), (32, 65, 32, 2), (64, 65, 64, 2)],
            ids=[
                "shared-query-tail-no-merge",
                "shared-query-uneven-splits",
                "fragment-query-two-wave-pairs",
                "fragment-query-four-wave-pairs",
            ],
        )


def test_mla(heads, length, head_tile, splits):
    """Run with pytest on this file; non-C500 environments skip before kernel imports."""
    import pytest

    if torch is None or not torch.cuda.is_available() or "c500" not in torch.cuda.get_device_name().lower():
        pytest.skip("This example has only been validated on MetaX C500")
    from example_mla_decode_maca import mla_decode_maca

    compiled = mla_decode_maca(1, length, heads, num_split=splits, head_tile=head_tile)

    def run(q, q_pe, kv, k_pe):
        output = torch.empty_like(q)
        compiled(q, q_pe, kv, k_pe, output)
        return output

    cases = [("random", 7), ("random", 19), ("early-peak", 7), ("late-peak", 7), ("large-v", 7)]
    check_kernel("candidate", run, 1, length, cases, torch.device("cuda"), trace=False, heads=heads)


if __name__ == "__main__":
    main()
