# Five-step failure-risk LSTM

Python 3.11+ and PyTorch 2.6+ are sufficient. This training command does not load
Qwen, Transformers, ALFWorld, sentence-transformers or external API clients.
An existing environment with PyTorch can be used directly; otherwise:

```bash
python -m pip install -r requirements-training.txt
python train_lstm.py \
  --input features/lstm/windows.csv \
  --output-dir features/lstm/results \
  --epochs 30 --seed 42
```

CPU is the default and appropriate for this small model. Each invocation writes
to a **new** `features/lstm/results/<UTC timestamp>_<UUID>/` directory, never
overwriting previous outputs or editing the input. The printed `RESULT_DIR`
identifies the actual result directory. Missing/invalid input stops the command;
the program never generates substitute research data.

## Model and input

The loader explicitly reads `E_i`, `C_i`, `S_i` by column name for each time index,
forming `[batch,5,3]`, irrespective of the feature-major CSV header order. It
rejects missing, NaN, infinite, duplicate or malformed samples. The five-step
endpoints must satisfy `end_step - start_step == 4`. Features remain in their
original units: entropy in nats, context ratio, cosine similarity. No scaler,
feature search or hyperparameter search is applied.

The model is one LSTM layer (input 3, hidden 16, batch-first) and a linear 16→1
head on its final hidden state. Training uses raw logits and
`BCEWithLogitsLoss(pos_weight=negative_train_count/positive_train_count)`, Adam
at 0.001, batch size 32, 30 epochs. The final epoch is saved; validation is not
used to select a checkpoint. Probabilities are sigmoid(logits), threshold 0.5.
Seeds, deterministic algorithms, CPU thread count and shuffle generator are fixed.

## Split and label limitations

Split units preserve at least `(run_dir, episode_id, condition)`. Additionally,
paired `.../no_fault` and `.../fault` directories with the same parent and episode
are kept together so shared prefixes cannot cross partitions. One seeded shuffle
assigns 20% of independent groups (rounded up, leaving at least one train group)
to validation. Thus the sample ratio may differ from 80:20. No window-level random
split, balancing based on validation labels, retry search or train/test split is
performed. There must be at least two independent groups and both labels in the
resulting training partition; otherwise collect more data. Correlated repetitions
stored under different run directories are not automatically inferred as pairs;
curate independent source runs before making generalization claims.

Labels are taken unchanged from the windows generator: they are outcome-based
experimental labels, not calibrated probabilities of a future causal failure.
Pre-injection/mixed windows retain that generator's label-0 policy. A single-class
validation set has `roc_auc: null` with a reason; precision/recall/F1 use zero for
undefined denominators. A small validation set is preliminary evidence only.

## Output files

- `metrics.json`: final validation Accuracy, Precision, Recall, F1, ROC-AUC,
  confusion matrix, train/validation class counts, train-only pos_weight,
  sample counts, input SHA256 and zero-overlap leakage checks.
- `training_history.csv`: epoch, train/validation loss and validation metrics.
- `predictions.csv`: **validation samples only**, source row/group metadata,
  true label, raw logit, probability and prediction, preserving source order.
- `model.pt`: CPU state dictionary, fixed architecture, training configuration,
  input SHA256, epoch, threshold and pos_weight.
- `split.json`: seed, split policy, actual train/validation groups and row indices,
  paired group membership and overlap checks.
- `config.json`: configuration, PyTorch version and final-epoch checkpoint policy.

If training fails after output allocation, partial results and `error.json` are
preserved. Inspect the error rather than treating incomplete directories as results.

Reload a trusted checkpoint without training:

```python
import torch
from src.lstm_classifier import FailureRiskLSTM

checkpoint = torch.load("RESULT_DIR/model.pt", map_location="cpu", weights_only=True)
model = FailureRiskLSTM()
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

Tests use tiny synthetic CPU fixtures, never pretrained models or live episodes:

```bash
python -m unittest discover -s tests -p 'test_lstm_training.py' -v
```
