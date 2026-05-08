import math
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import json
from typing import Dict, Tuple, Optional
import tilelang
import tilelang.language as T
from tilelang.autotuner import *
from custom_fusedmoe import RoutedMoEKernel
from ref_fusedmoe import ref_kernel


class Expert(nn.Module):
    def __init__(self, config: Dict, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor, d_expert: Optional[int] = None):
        super().__init__()
        self.config = config
        self.act_fn = nn.SiLU()
        self.d_hidden: int = config["d_hidden"]
        self.d_expert: int = config["d_expert"] if d_expert is None else d_expert
        self.device = torch.device("cuda")

        self.W_gate_weight = gate.t().contiguous().to(self.device)
        self.W_up_weight = up.t().contiguous().to(self.device)
        self.W_down_weight = down.t().contiguous().to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(x @ self.W_gate_weight)
        out = (gate * (x @ self.W_up_weight)) @ self.W_down_weight
        return out

class MoEGate(nn.Module):
    def __init__(self, config: Dict, weights: Dict):
        super().__init__()
        self.top_k: int = config["n_experts_per_token"]
        self.num_experts: int = config["n_routed_experts"]
        self.d_hidden: int = config["d_hidden"]

        self.W_g_weight = weights["router.weight"].t()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = x @ self.W_g_weight
        scores = logits.softmax(dim=-1)
        topk_scores, topk_indices = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        return topk_indices, topk_scores

class MoE(nn.Module):
    def __init__(
        self, config: Dict, routed_kernel, weights: Dict, padding_M: int = 128
    ):
        super().__init__()
        self.config = config
        self.routed_kernel = routed_kernel
        self.padding_M = padding_M
        self.experts = nn.ModuleList(
            [
                Expert(
                    config,
                    gate=weights[f"experts.{i}.0.weight"],
                    up=weights[f"experts.{i}.1.weight"],
                    down=weights[f"experts.{i}.2.weight"],
                )
                for i in range(config["n_routed_experts"])
            ]
        )
        self.device = torch.device("cuda")
        self.gating_network = MoEGate(config, weights).to(self.device)

        self.expert_cache = torch.zeros(
            (config["batch_size"] * config["seq_len"], config["d_hidden"]), dtype=torch.float16, device=self.device
        )
        self.stacked_expert_w_gate = torch.stack([expert.W_gate_weight for expert in self.experts], dim=0)
        self.stacked_expert_w_up = torch.stack([expert.W_up_weight for expert in self.experts], dim=0)
        self.stacked_expert_w_down = torch.stack([expert.W_down_weight for expert in self.experts], dim=0)
        self.stacked_expert_tokens = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"], self.config["d_hidden"]),
            dtype=torch.float16,
            device=self.device,
        )
        self.stacked_expert_weights = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"]), dtype=torch.float16, device=self.device
        )
        self.stacked_expert_tokens_idxs = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"]), dtype=torch.int64, device=self.device
        )

        self.up_logits_routed = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"], self.config["d_expert"]),
            dtype=torch.float16,
            device=self.device,
        )
        self.expert_output_routed = torch.empty(
            (config["batch_size"] * config["seq_len"] * config["n_experts_per_token"], self.config["d_hidden"]),
            dtype=torch.float16,
            device=self.device,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        batch_size, seq_len, hidden_dim = x.shape
        expert_indices, expert_scores = self.gating_network(x)
        flat_expert_indices = expert_indices.view(-1)
        flat_expert_weights = expert_scores.view(-1)
        x_flat = x.view(-1, hidden_dim)

        # Prepare for grouped GEMM
        idxs = flat_expert_indices.argsort()
        counts = flat_expert_indices.bincount().cpu().numpy()
        # counts = flat_expert_indices.bincount()
        tokens_per_expert = counts.cumsum()
        # tokens_per_expert = torch.cumsum(counts, dim=0)
        num_per_tok = self.config["n_experts_per_token"]
        token_idxs = idxs // num_per_tok

        # Get stacked expert tokens and expert weights

        for expert_id, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
            if start_idx == end_idx:
                continue

            exp_token_idxs = token_idxs[start_idx:end_idx]
            expert_tokens = x_flat[exp_token_idxs]

            self.stacked_expert_tokens[start_idx:end_idx] = expert_tokens
            self.stacked_expert_tokens_idxs[start_idx:end_idx] = exp_token_idxs
            self.stacked_expert_weights[start_idx:end_idx] = flat_expert_weights[idxs[start_idx:end_idx]]

        group_sizes = torch.tensor(counts, dtype=torch.int32, device=self.device)
        group_offset = torch.tensor(tokens_per_expert - counts, dtype=torch.int32, device=self.device)

        group_padded_offsets = [0 for _ in range(len(group_sizes))]
        for i in range(1, len(group_sizes)):
            group_padded_offsets[i] = group_padded_offsets[i - 1] + math.ceil((counts[i - 1] + 1) / self.padding_M) * self.padding_M

        block_token = 128
        M = (
            math.ceil(self.config["batch_size"] * self.config["seq_len"] * self.config["n_experts_per_token"] / block_token)
            + self.config["n_routed_experts"]
        )
        group_idx_for_bx = [0 for _ in range(M)]

        for bx in range(M):
            m_start_padded = bx * block_token
            for i in range(self.config["n_routed_experts"]):
                if m_start_padded >= group_padded_offsets[i]:
                    group_idx_for_bx[bx] = i

        group_padded_offsets = torch.tensor(group_padded_offsets, dtype=torch.int32, device=self.device)
        group_idx_for_bx = torch.tensor(group_idx_for_bx, dtype=torch.int32, device=self.device)

        routed_stream = torch.cuda.default_stream()
        torch.cuda.synchronize()

        with torch.cuda.stream(routed_stream):
            # Tilelang version: Grouped GEMM
            self.routed_kernel(
                self.stacked_expert_tokens,
                self.stacked_expert_w_gate,
                self.stacked_expert_w_up,
                self.stacked_expert_w_down,
                self.stacked_expert_weights,
                group_sizes,
                group_offset,
                group_padded_offsets,
                group_idx_for_bx,
                self.up_logits_routed,
                self.expert_output_routed,
            )

            # Scatter reduce
            self.expert_cache = torch.scatter_reduce(
                self.expert_cache,
                0,
                self.stacked_expert_tokens_idxs.view(-1, 1).repeat(1, x_flat.shape[-1]),
                self.expert_output_routed,
                reduce="sum",
            )
            routed_output = self.expert_cache.view(*orig_shape)


        torch.cuda.synchronize()

        return routed_output


def custom_kernel(data: Tuple[torch.Tensor, Dict, Dict]) -> torch.Tensor:
    """
    DeepSeek-style Mixture of Experts using Tilelang.

    Args:
        data: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, hidden_size]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters

    Returns:
        Tuple containing:
            - output: Processed tensor [batch_size, seq_len, d_model]
    """
    input_tensor, weights, config = data

    routed_kernel = RoutedMoEKernel(
        config["d_hidden"],
        config["d_expert"],
        config["n_routed_experts"],
        group_sum=config["batch_size"] * config["seq_len"] * config["n_experts_per_token"],
        group_count=config["n_routed_experts"]
    )

    moe = MoE(config, routed_kernel, weights, padding_M=128)

    output = moe(input_tensor)

    return output


def generate_input(
    dhidden: int, dexpert: int, nroutedexperts: int, nexpertspertoken: int, bs: int, seqlen: int, seed: int
) -> Tuple[torch.Tensor, Dict, Dict]:
    # Really dumb but for now _ isn't parsing correctly.
    d_hidden = dhidden
    d_expert = dexpert
    n_routed_experts = nroutedexperts
    n_experts_per_token = nexpertspertoken
    batch_size = bs
    seq_len = seqlen

    config = {
        "d_hidden": d_hidden,
        "d_expert": d_expert,
        "n_routed_experts": n_routed_experts,
        "n_experts_per_token": n_experts_per_token,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    num_experts = n_routed_experts
    expert_dim = d_expert
    weights = {}

    input_tensor = torch.randn((batch_size, seq_len, d_hidden), device="cuda", dtype=torch.float16, generator=gen).contiguous()

    # Initialize router weights
    weights["router.weight"] = torch.randn((num_experts, d_hidden), device="cuda", dtype=torch.float16, generator=gen) / math.sqrt(d_hidden)

    for i in range(num_experts):
        weights[f"experts.{i}.0.weight"] = torch.randn(
            (d_hidden, expert_dim), device="cuda", dtype=torch.float16, generator=gen
        ) / math.sqrt(expert_dim)

        weights[f"experts.{i}.1.weight"] = torch.randn(
            (d_hidden, expert_dim), device="cuda", dtype=torch.float16, generator=gen
        ) / math.sqrt(expert_dim)

        weights[f"experts.{i}.2.weight"] = torch.randn(
            (expert_dim, d_hidden), device="cuda", dtype=torch.float16, generator=gen
        ) / math.sqrt(d_hidden)

    return (input_tensor, weights, config)


def clone_data(data):
    """
    Recursively goes through data and clones all tensors.
    """
    if isinstance(data, tuple):
        return tuple(clone_data(x) for x in data)
    elif isinstance(data, list):
        return [clone_data(x) for x in data]
    elif isinstance(data, dict):
        return {k: clone_data(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.clone()
    else:
        return data

def cal_tflops(config, t_sec):
    M = config["bs"] * config["seqlen"] * config["nexpertspertoken"]
    K = config["dhidden"]
    N = config["dexpert"] * config["nroutedexperts"]
    t_sec = prof.key_averages().total_average().cuda_time_total / 1e6  # ms -> s
    flops = 2 * M * N * K
    tflops = flops / (t_sec * 1e12)
    print(f"Estimated TFLOPS: {tflops:.2f}")


def run_moe_test(config: dict, test_type: str, warm_up=10, iteration=100):
    data = generate_input(**config)

    if test_type == "functional":
        try:
            ref_output = ref_kernel(clone_data(data)).to(torch.float32)
            tilelang_output = custom_kernel(clone_data(data)).to(torch.float32)
            torch.testing.assert_close(ref_output, tilelang_output, atol=1e-2, rtol=1e-2)
            print(f"✅ Functional test passed for config: {config}")
        except AssertionError as e:
            print(f"❌ Functional test failed for config: {config}")
            print("error msg: ", str(e))
    elif test_type == "performance":
        import time
        for i in range(warm_up):
            _ = custom_kernel(clone_data(data))
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(iteration):
            _ = custom_kernel(clone_data(data))
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_ms = elapsed_ms / iteration
        print(f"⏱ Performance test: {elapsed_ms:.8f}ms for config: {config}")
        # with torch.profiler.profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     record_shapes=True,
        #     with_stack=False
        # ) as prof:
        #     with torch.profiler.record_function("routed_kernel"):
        #         _ = custom_kernel(clone_data(data))
        # prof_results = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        # print(prof_results)
    else:
        raise ValueError(f"Unknown test type {test_type}")

def run_from_config_file(config_file: str):
    with open(config_file, "r") as f:
        configs = json.load(f)

    for test_type in ["functional", "performance"]:
        print(f"\n=== Running {test_type} tests ===")
        for cfg in configs.get(test_type, []):
            run_moe_test(cfg, test_type)

def clear_caches():
    import os
    import shutil
    cache_dir = os.path.expanduser("~/.tilelang/cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    else:
        print("no tilelang cache found")

if __name__ == "__main__":
    # tilelang.disable_cache()
    clear_caches()
    config_file = "moe_test_configs.json"
    run_from_config_file(config_file)
