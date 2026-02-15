import time
import cProfile
import pstats
import io
import statistics

from litellm import Router


def make_model_list(n, model_name="gpt-test"):
    models = []
    for i in range(n):
        models.append(
            {
                "model_name": model_name,
                "litellm_params": {"model": model_name},
                "model_info": {"id": str(i), "tpm": 10, "rpm": 100},
            }
        )
    return models


def time_calls(router, model_group, calls_per_request=3):
    # Ensure cached wrapper is clear
    try:
        router._cached_get_model_group_info.cache_clear()
    except Exception:
        pass

    # Uncached: call get_model_group_info repeatedly
    t0 = time.perf_counter()
    for _ in range(calls_per_request):
        _ = router.get_model_group_info(model_group)
    t1 = time.perf_counter()
    uncached = t1 - t0

    # Cached: clear cache then call cached wrapper repeatedly
    try:
        router._cached_get_model_group_info.cache_clear()
    except Exception:
        pass
    t0 = time.perf_counter()
    for _ in range(calls_per_request):
        _ = router._cached_get_model_group_info(model_group)
    t1 = time.perf_counter()
    cached = t1 - t0

    return uncached, cached


if __name__ == "__main__":
    MODEL_COUNT = 1000
    CALLS_PER_REQUEST = 3
    REPS = 5

    print(f"Building router with {MODEL_COUNT} deployments (all same model)")
    router = Router()
    model_list = make_model_list(MODEL_COUNT, model_name="gpt-test")
    # Use the fast index builder to avoid heavy client init
    router._build_model_id_to_deployment_index_map(model_list)

    # Warm-up single call
    print("Warm-up call...")
    _ = router.get_model_group_info("gpt-test")

    uncached_times = []
    cached_times = []

    for i in range(REPS):
        u, c = time_calls(router, "gpt-test", calls_per_request=CALLS_PER_REQUEST)
        print(f"Run {i+1}: uncached={u:.6f}s, cached={c:.6f}s")
        uncached_times.append(u)
        cached_times.append(c)

    print("\nSummary (seconds)")
    print(f"uncached mean={statistics.mean(uncached_times):.6f}, stdev={statistics.stdev(uncached_times):.6f}")
    print(f"cached   mean={statistics.mean(cached_times):.6f}, stdev={statistics.stdev(cached_times):.6f}")
    if statistics.mean(uncached_times) / max(1e-12, statistics.mean(cached_times)) >= 2.0:
        print("Measured >=2x speedup using cached wrapper for repeated calls")
    else:
        print("Speedup <2x")

    # Run cProfile on one uncached call to show hotspots
    print("\nProfiling one uncached get_model_group_info() call (top 20 cumulative):")
    pr = cProfile.Profile()
    pr.enable()
    _ = router.get_model_group_info("gpt-test")
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())
