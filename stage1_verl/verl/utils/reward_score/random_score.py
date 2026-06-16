import random

def random_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Returns a random score between 0 and 1, as a sanity check."""
    return random.uniform(0, 1)