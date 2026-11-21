from tools.run_experiments import CORE_EXPERIMENTS, OPTIONAL_EXPERIMENTS


def test_core_experiment_registry_matches_task_matrix():
    assert CORE_EXPERIMENTS == (
        "rgb",
        "ir",
        "ratio1_concat",
        "ratio2_concat",
        "ratio2_simple",
        "ratio2_adapter",
        "ratio2_gated",
    )
    assert OPTIONAL_EXPERIMENTS == ("ratio1_gated", "hierafire")
