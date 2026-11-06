"""Fixed preliminary intervention rules, reused E/C/S features and CPU LSTM inference."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from src.timeseries_features import FeatureError, PromptCounter, digest, read_json

MODES = ("no_intervention", "periodic_goal_reinjection", "entropy_only", "proposed")
DEFAULT_CHECKPOINT = "features/lstm/results/20261106T003226.747846Z_03a0d113/model.pt"


@dataclass(frozen=True)
class InterventionConfig:
    mode: str = "proposed"
    lstm_risk_threshold: float = .5
    similarity_threshold: float = .3
    context_threshold: float = .8
    entropy_threshold: float | None = None
    period: int = 5
    history_keep: int = 3

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("Unsupported intervention mode")
        for name in ("lstm_risk_threshold", "similarity_threshold", "context_threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Nonfinite {name}")
        if not 0 <= self.lstm_risk_threshold <= 1 or not -1 <= self.similarity_threshold <= 1 or self.context_threshold <= 0:
            raise ValueError("Risk/similarity/context thresholds out of range")
        if self.entropy_threshold is not None and (type(self.entropy_threshold) not in (int, float) or not math.isfinite(self.entropy_threshold) or self.entropy_threshold < 0):
            raise ValueError("entropy_threshold must be finite nonnegative nats")
        if self.mode in ("entropy_only", "proposed") and self.entropy_threshold is None:
            raise ValueError("Entropy threshold required: supply CLI value or derive from recorded training split")
        if type(self.period) is not int or self.period < 1 or type(self.history_keep) is not int or self.history_keep < 1:
            raise ValueError("period/history_keep must be positive integers")

    def public_config(self):
        return asdict(self)


def choose_intervention(config, features, risk, *, step, injection_step, fault_active):
    """At most one rule; diagnosis never uses the injected fault's known type."""
    if not fault_active or config.mode == "no_intervention":
        return None, "fault_inactive" if not fault_active else "disabled"
    if config.mode == "periodic_goal_reinjection":
        # First at injection_step + period - 1, then every period steps.
        due = (step - injection_step + 1) % config.period == 0
        return ("goal_reinjection", "periodic_schedule") if due else (None, "period_not_due")
    if config.mode == "entropy_only":
        return ("reobservation_replanning", "E_above_fixed_threshold") if features["E"] > config.entropy_threshold else (None, "E_not_abnormal")
    if risk is None:
        return None, "five_step_warmup"
    if not math.isfinite(risk) or not 0 <= risk <= 1:
        raise ValueError("Invalid LSTM risk")
    if risk < config.lstm_risk_threshold:
        return None, "risk_below_threshold"
    if features["S"] <= config.similarity_threshold:
        return "goal_reinjection", "risk_gate_and_S_at_or_below_threshold"
    if features["C"] >= config.context_threshold:
        return "context_cleaning", "risk_gate_and_C_at_or_above_threshold"
    if features["E"] >= config.entropy_threshold:
        return "reobservation_replanning", "risk_gate_and_E_at_or_above_threshold"
    return None, "risk_high_but_no_rule_abnormality"


def decision_features(decision, original_goal, config, similarity, counter):
    """Same definitions as offline extraction; prompt count comes from real generation."""
    generation = decision.generation
    entropy = generation.get("mean_entropy")
    if type(entropy) not in (int, float) or not math.isfinite(entropy) or entropy < 0 or generation.get("entropy_unit", "nats") != "nats":
        raise FeatureError("Missing/invalid mean_entropy in nats")
    if not isinstance(original_goal, str) or not original_goal.strip() or not isinstance(decision.plan, str) or not decision.plan.strip():
        raise FeatureError("Original initial goal and generated plan required; no synthetic S")
    # Live operation must never guess a historical or hypothetical prompt.
    if "prompt_token_count" not in generation:
        raise FeatureError("Live features require actual generation prompt_token_count")
    count = counter.count(generation, {"local_llm": config.public_config()})
    score = similarity.score_pairs([(original_goal, decision.plan)])[0]
    if type(score) not in (int, float) or not math.isfinite(score) or not -1.000001 <= score <= 1.000001:
        raise FeatureError("Nonfinite/out-of-range semantic cosine similarity")
    return {"E": entropy, "C": count["prompt_token_count"] / count["max_context_tokens"], "S": score,
            "prompt_token_count": count["prompt_token_count"], "max_context_tokens": count["max_context_tokens"]}


class LSTMRiskDetector:
    """Inference only. Fixed trained architecture; weights loaded on CPU, never retrained."""
    def __init__(self, checkpoint):
        import torch
        from src.lstm_classifier import FailureRiskLSTM
        self.checkpoint = Path(checkpoint).resolve()
        self.sha256 = digest(self.checkpoint)
        value = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        expected = {"input_size": 3, "hidden_size": 16, "num_layers": 1, "batch_first": True,
                    "window_size": 5, "feature_order": ["E", "C", "S"], "output_size": 1}
        if value.get("architecture") != expected:
            raise ValueError("Checkpoint architecture/feature order must match fixed five-step [E,C,S] LSTM")
        self.model = FailureRiskLSTM()
        self.model.load_state_dict(value["state_dict"], strict=True)
        self.model.eval()
        if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise ValueError("Checkpoint contains nonfinite weights")
        self.training_input_sha256 = value.get("input_sha256")

    def predict(self, window):
        import torch
        if len(window) != 5 or any(len(state) != 3 for state in window):
            raise ValueError("LSTM requires time-major (5,3) input")
        x = torch.tensor([window], dtype=torch.float32)
        with torch.inference_mode():
            risk = torch.sigmoid(self.model(x)).item()
        if not math.isfinite(risk):
            raise ValueError("Nonfinite LSTM risk")
        return risk


def training_entropy_threshold(windows_path, split_path):
    """Fixed train-only 95th percentile; de-duplicate overlapping window steps."""
    from src.lstm_training_data import check_group_leakage, load_windows
    windows_path, split_path = Path(windows_path).resolve(), Path(split_path).resolve()
    split = read_json(split_path)
    source_hash = digest(windows_path)
    if split.get("input_sha256") != source_hash:
        raise ValueError("Calibration CSV SHA256 must match recorded LSTM split; never calibrate from test results")
    samples = load_windows(windows_path)
    train_indices, validation_indices = split["train_indices"], split["validation_indices"]
    check_group_leakage(samples, train_indices, validation_indices)
    values = {}
    for index in train_indices:
        sample = samples[index]
        for offset, state in enumerate(sample.x):
            key = (*sample.group, sample.start_step + offset)
            if state[0] < 0 or (key in values and values[key] != state[0]):
                raise ValueError("Negative/conflicting training entropy values")
            values[key] = state[0]
    ordered = sorted(values.values())
    position = .95 * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    threshold = ordered[low] + (position - low) * (ordered[high] - ordered[low])
    return threshold, {"source": "recorded_training_partition_only", "quantile": .95,
                       "method": "linear_interpolation_of_unique_group_step_E", "entropy_unit": "nats",
                       "unique_training_step_count": len(ordered), "windows_path": str(windows_path),
                       "windows_sha256": source_hash, "split_path": str(split_path), "split_sha256": digest(split_path),
                       "validation_samples_used": 0}
