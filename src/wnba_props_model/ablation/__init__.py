"""Feature-selection / ablation study harness (leakage-safe, nested rolling-origin CV)."""
from .feature_ablation import (
    ALL_PROPS,
    COUNT_PROPS,
    MARKET_PROPS,
    AblationConfig,
    assemble_frame,
    assert_nested_cv_integrity,
    assign_groups,
    load_inputs,
    make_expanding_folds,
    run_prop,
)
from .opponent_defense import (
    OppDefConfig,
    assert_no_opponent_defense_leakage,
    build_opponent_defense_features,
)

__all__ = [
    "ALL_PROPS", "COUNT_PROPS", "MARKET_PROPS", "AblationConfig", "assemble_frame",
    "assert_nested_cv_integrity", "assign_groups", "load_inputs", "make_expanding_folds",
    "run_prop", "OppDefConfig", "assert_no_opponent_defense_leakage",
    "build_opponent_defense_features",
]
