import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile

from utils import (
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
)


@torch.inference_mode()
def optimized_loop(model, input_ids, n_steps):
    # Prefill once, then decode one token at a time, reusing the KV cache
    # returned by the model. This turns each decode step from a full
    # re-prefill (O(prefix^2) attention work) into O(prefix) work.
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)  # [B, 1]

    # Keep generated IDs on-device; never call .item() inside the loop so
    # the CPU can keep launching kernels without waiting on the GPU.
    token_buf = [next_token]
    for _ in range(n_steps - 1):
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        token_buf.append(next_token)

    # Single CPU<->GPU sync at the very end.
    return torch.cat(token_buf, dim=1).squeeze(0).tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    trace_path = RESULTS_DIR / trace_name
    prof.export_chrome_trace(str(trace_path))
    print(f"Trace saved to {trace_path}")


def generate_optimized(optimized_trace_name: str) -> float:
    # bfloat16 cuts weight/KV traffic in half and lets the L40S use the much
    # higher BF16 throughput path. KV-cache reuse lives inside optimized_loop.
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    return time_generation(optimized_loop, model, input_ids, "Optimized")


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and why:
#
# 1. KV cache reuse (biggest single win).
#    V0 re-feeds the *entire* generated prefix into the model on every decode
#    step, so step i runs attention + MLP over (PROMPT_LEN + i) tokens. For
#    PROMPT_LEN=1024 and MAX_NEW_TOKENS=128 that is ~140k token-forwards
#    instead of the 1024 (prefill) + 127 (decode) ≈ 1151 token-forwards a real
#    KV-cached loop does. The optimized loop calls the model once with the
#    full prompt to build the cache, then feeds one token at a time and
#    appends to past_key_values. Verified in the trace: V0's GPU stream is
#    dominated by linears and SDPA over a long sequence dim that grows each
#    step; the optimized trace shows the same kernels but with a fixed,
#    tiny (T=1) sequence dim during decode.
#
# 2. bfloat16 weights/activations.
#    build_model(torch.bfloat16) halves bytes moved for weights and the KV
#    cache, and lets the L40S use its much faster BF16 matmul path instead of
#    the FP32 CUDA-core path. For this memory-bound decode workload the
#    halved HBM traffic is what actually shows up in the wall clock.
#
# 3. Drop the per-step .item() sync.
#    V0 calls next_token_id.item() every step, which forces a
#    cudaStreamSynchronize and stalls the CPU until the GPU is done. The
#    optimized loop accumulates one-element token tensors on the device and
#    only converts to Python ints once, at the end, with a single .tolist().
#    In the trace this removes per-step "sync" markers that pinned the CPU
#    between launches and starved the GPU stream.
#
# 4. Drop the per-step torch.cat on the prompt tensor.
#    V0 builds generated_ids by concatenating the new token onto the growing
#    prompt every step — allocations and a copy of an ever-larger tensor.
#    With a KV cache the model only needs the new token, so the cat goes
#    away entirely; we keep generated tokens in a small Python list of
#    [B, 1] tensors instead.
#
# 5. torch.inference_mode() on the loop.
#    Disables autograd version counters and view tracking for every op in
#    the loop. Tiny per-op savings, but the decode loop launches a *lot* of
#    small ops, so it adds up.
#
# Expected per-fix speedups on L40S (measure by toggling one change at a
# time from the V0 baseline of ~21s for 128 tokens):
#
#   + KV cache only (fp32, .item() kept):   ~8–12×   (kills the O(N^2) work)
#   + bf16 on top of KV cache:              ~1.5–2×  additional
#   + drop .item()/cat, add inference_mode: ~1.1–1.3× additional
#   Combined target:                        ≥ 4×, typically much higher
#
# Biggest impact and why:
#
# KV-cache reuse, by a wide margin. The other fixes are constant-factor
# improvements (dtype halving, removing a sync, fewer allocations); the KV
# cache changes the *asymptotic* amount of work per step from O(prefix) to
# O(1), and at PROMPT_LEN=1024 the prefix already dominates. bf16 and the
# sync removal then convert what is now a memory-bound, launch-bound decode
# loop into something close to peak — but they are only meaningful once the
# redundant prefill work is gone.
