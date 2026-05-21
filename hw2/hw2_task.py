import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    # TODO: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`
    generated_ids = input_ids.clone()
    generated_tokens = []
    for _ in range(n_steps):
        outputs = model(input_ids=generated_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_value = next_token_id.item()
        generated_tokens.append(token_value)
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return generated_tokens


def profile(loop_fn, model, input_ids, trace_name: str):
    # TODO: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    pass


def generate_optimized(optimized_trace_name: str):
    # TODO: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop
    pass


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    generate_optimized(optimized_trace_name="v1_optimized_trace.json")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
#
#
# Biggest impact and why:
#
