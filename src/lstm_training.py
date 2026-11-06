"""Small reproducible CPU-first LSTM experiment; never edits source windows."""
import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.llm_io import new_run, save_json
from src.lstm_classifier import FailureRiskLSTM
from src.lstm_metrics import binary_metrics
from src.lstm_training_data import (TrainingDataError, check_group_leakage, file_sha256,
                                    group_split, load_windows, training_pos_weight)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42
    validation_fraction: float = .2
    device: str = "cpu"
    num_threads: int = 1

    def validate(self):
        if any(type(value) is not int or value < 1 for value in (self.epochs, self.batch_size, self.num_threads)):
            raise TrainingDataError("epochs, batch_size and num_threads must be positive integers")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise TrainingDataError("learning_rate must be finite and positive")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise TrainingDataError("seed must be an integer in [0, 2**32)")
        if not 0 < self.validation_fraction < 1:
            raise TrainingDataError("validation_fraction must be between 0 and 1")
        if self.device not in ("cpu", "cuda"):
            raise TrainingDataError("device must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise TrainingDataError("CUDA was requested but is unavailable; use --device cpu")


def _write_csv(path, rows, fieldnames):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(input_path, output_dir, config=None):
    config = config or TrainingConfig()
    config.validate()
    input_path = Path(input_path).resolve()
    input_hash = file_sha256(input_path)
    samples = load_windows(input_path)
    split = group_split(samples, config.validation_fraction, config.seed)
    train_indices, validation_indices = split["train_indices"], split["validation_indices"]
    pos_weight, train_counts = training_pos_weight(samples, train_indices)
    x = torch.tensor([sample.x for sample in samples], dtype=torch.float32)
    y = torch.tensor([sample.label for sample in samples], dtype=torch.float32)
    if not torch.isfinite(x).all():
        raise TrainingDataError("Features overflowed float32; NaN/inf input is forbidden")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    torch.use_deterministic_algorithms(True)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(TensorDataset(x[train_indices], y[train_indices]),
                              batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
    validation_loader = DataLoader(TensorDataset(x[validation_indices], y[validation_indices]),
                                   batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = FailureRiskLSTM().to(config.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=config.device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    architecture = {"input_size": 3, "hidden_size": 16, "num_layers": 1, "batch_first": True,
                    "window_size": 5, "feature_order": ["E", "C", "S"], "output_size": 1}
    run = new_run(output_dir)
    split.update({"input_path": str(input_path), "input_sha256": input_hash})
    save_json(run / "split.json", split)
    save_json(run / "config.json", {**asdict(config), "architecture": architecture,
                                    "torch_version": str(torch.__version__), "input_sha256": input_hash,
                                    "normalization": "none", "checkpoint_selection": "final_epoch"})
    history = []
    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            train_loss = 0.
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(config.device), batch_y.to(config.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                if not torch.isfinite(loss):
                    raise TrainingDataError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_y)
            model.eval()
            validation_loss, final_logits = 0., []
            with torch.inference_mode():
                for batch_x, batch_y in validation_loader:
                    logits = model(batch_x.to(config.device))
                    loss = criterion(logits, batch_y.to(config.device))
                    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                        raise TrainingDataError("Nonfinite validation loss/logits")
                    validation_loss += loss.item() * len(batch_y)
                    final_logits.extend(logits.cpu().tolist())
            probabilities = torch.sigmoid(torch.tensor(final_logits)).tolist()
            labels = [samples[index].label for index in validation_indices]
            metrics = binary_metrics(labels, probabilities)
            history.append({"epoch": epoch, "train_loss": train_loss / len(train_indices),
                            "validation_loss": validation_loss / len(validation_indices),
                            **{key: metrics[key] for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc")}})

        leakage = check_group_leakage(samples, train_indices, validation_indices)
        if file_sha256(input_path) != input_hash:
            raise TrainingDataError("Input CSV changed during training; result is not reproducible")
        metrics.update({"train_class_counts": train_counts, "pos_weight": pos_weight,
                        "pos_weight_source": "training_partition_only", "train_sample_count": len(train_indices),
                        "validation_sample_count": len(validation_indices), "leakage_check": leakage,
                        "seed": config.seed, "epochs": config.epochs, "input_path": str(input_path),
                        "input_sha256": input_hash, "final_train_loss": history[-1]["train_loss"],
                        "final_validation_loss": history[-1]["validation_loss"],
                        "evaluation_subset": "validation", "checkpoint_selection": "final_epoch"})
        save_json(run / "metrics.json", metrics)
        _write_csv(run / "training_history.csv", history, list(history[0]))
        predictions = []
        for index, logit, probability in zip(validation_indices, final_logits, probabilities):
            sample = samples[index]
            predictions.append({"sample_index": index, "source_csv_line": sample.source_line,
                                "run_dir": sample.run_dir, "episode_id": sample.episode_id, "condition": sample.condition,
                                "fault_type": sample.metadata.get("fault_type", ""),
                                "fault_severity": sample.metadata.get("fault_severity", ""),
                                "start_step": sample.start_step, "end_step": sample.end_step,
                                "label": sample.label, "logit": logit, "probability": probability,
                                "prediction": int(probability >= .5), "subset": "validation"})
        _write_csv(run / "predictions.csv", predictions, list(predictions[0]))
        with (run / "model.pt").open("xb") as stream:
            torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "architecture": architecture, "training_config": asdict(config), "epoch": config.epochs,
                        "threshold": .5, "pos_weight": pos_weight, "input_sha256": input_hash}, stream)
        return run, metrics
    except Exception as exc:
        save_json(run / "error.json", {"error_type": type(exc).__name__, "message": str(exc)})
        raise
