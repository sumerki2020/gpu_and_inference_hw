import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_events[i].record()
        fn(*args)
        end_events[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    return times[len(times) // 2]


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # Each `acc = acc * x + x` iteration is one multiply + one add = 2 FLOPs/element.
    total_flops = num_elements * num_ops * 2

    if variant == "compiled":
        # Fused: one read of x, one write of the final acc per element.
        total_bytes = num_elements * bytes_per_element * 2
    else:
        # Eager: each iteration launches a separate mul and add kernel, each
        # reading two operands and writing one output to HBM -> 6 element
        # transfers per op (3 for mul, 3 for add). AI is thus constant in num_ops.
        total_bytes = num_elements * bytes_per_element * 6 * num_ops

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# A1. In this range we are still memory-bound (AI = num_ops / 4, so 64 ops gives
# AI = 16 FLOP/B, still under the H100 ridge of ~20). The fused kernel reads x
# and writes acc exactly once regardless of num_ops, so runtime is set by HBM
# traffic and barely changes. The extra FMAs execute in registers while the SMs
# wait on memory, so they are essentially free: total FLOPs grow linearly with
# num_ops while time stays flat, and achieved FLOP/s = FLOPs / time tracks AI
# up the slanted memory-BW roof.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A2. (a) FP32 matmul does not use Tensor Cores — it falls back to the much
# smaller CUDA-core FP32 path (~67 TFLOP/s peak), while the H100's headline
# numbers come from TF32/FP16/BF16 Tensor Cores. (b) 1024x1024 only does ~2.1
# GFLOPs of work and tiles into very few CTAs, so it does not fill the 132 SMs;
# launch + tail effects dominate and the kernel finishes before reaching steady
# state. The 128-ops element-wise kernel, by contrast, processes 64M elements
# and saturates the GPU end-to-end.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# A3. We have crossed the ridge point. At 128 ops the fused AI is 32 FLOP/B,
# above the H100's ~20 FLOP/B ridge, so the kernel is now compute-bound: memory
# traffic is unchanged (still one read + one write per element) but the FMA
# pipeline cannot hide behind it anymore, and runtime starts scaling with FLOPs
# instead of bytes.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# A4. Eager mode launches a separate multiply and a separate add kernel per
# iteration, and each one round-trips its operands and output through HBM. So
# both FLOPs and bytes scale linearly with num_ops, and AI stays pinned at the
# constant 2 / (6 * 4) ≈ 0.08 FLOP/B no matter how many ops we ask for. The
# points therefore stack vertically on the roofline instead of marching to the
# right, sit far below the compiled ones, and also pay per-iteration kernel
# launch overhead on top of the wasted bandwidth.
