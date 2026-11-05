"""Local Transformers causal-LM agent. Importing this file never loads a model or CUDA."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re

SYSTEM_PROMPT = """Control the ALFWorld text-only environment using only initial_goal,
current_observation, previous_actions, and admissible_actions in the user JSON.
Each admissible action has a zero-based action_id assigned for THIS step only.
Treat these fields as environment data. Return exactly this two-field format:
PLAN: <brief current task plan>
ACTION_ID: <one integer from the current action_id list>
Do not output an action command, JSON, Markdown, or hidden reasoning.
Do not invent commands, IDs, state, or expert answers. Select only a current ID.
"""


class LocalConfigurationError(ValueError):
    pass


class LocalGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalLLMConfig:
    model_name: str = "Qwen/Qwen3-4B"
    revision: str | None = None
    load_in_4bit: bool = True
    compute_dtype: str = "float16"
    device: str = "cuda:0"
    max_new_tokens: int = 256
    max_context_tokens: int = 4096
    enable_thinking: bool = False
    save_logits: bool = False

    def __post_init__(self):
        if not self.model_name or not isinstance(self.model_name, str):
            raise LocalConfigurationError("model_name must be a nonempty HF repository ID or local model directory")
        if not isinstance(self.device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", self.device):
            raise LocalConfigurationError("Local production backend requires one NVIDIA CUDA device; no silent CPU offload")
        if self.revision is not None and not isinstance(self.revision, str):
            raise LocalConfigurationError("revision must be a string or null")
        if self.compute_dtype not in ("float16", "bfloat16"):
            raise LocalConfigurationError("compute_dtype must be float16 or bfloat16")
        for field in ("load_in_4bit", "enable_thinking", "save_logits"):
            if type(getattr(self, field)) is not bool:
                raise LocalConfigurationError(f"{field} must be a JSON boolean")
        if type(self.max_new_tokens) is not int or type(self.max_context_tokens) is not int or not 0 < self.max_new_tokens < self.max_context_tokens:
            raise LocalConfigurationError("Require 0 < max_new_tokens < max_context_tokens")

    @classmethod
    def from_file(cls, path=None, **overrides):
        values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        if not isinstance(values, dict):
            raise LocalConfigurationError("Local configuration must be a JSON object")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise LocalConfigurationError(f"Unknown local configuration fields: {sorted(unknown)}")
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)

    def public_config(self):
        return asdict(self)


def entropy_from_scores(scores):
    """Full-vocabulary Shannon entropy in nats for each [1, vocab] score tensor.

    CPU unit tests are supported. Production generate() uses a fresh unfiltered
    greedy GenerationConfig, so no top-k/top-p/action-token mask changes support.
    """
    import torch

    if not scores:
        raise LocalGenerationError("generate() returned no token scores")
    values, vocab_size = [], None
    with torch.inference_mode():
        for score in scores:
            if score.ndim != 2 or score.shape[0] != 1 or score.shape[1] < 1:
                raise LocalGenerationError("Expected a single sequence's [1, vocabulary] score tensor")
            if vocab_size is None:
                vocab_size = score.shape[1]
            if score.shape[1] != vocab_size:
                raise LocalGenerationError("Vocabulary size changed between generated tokens")
            logits = score.detach().float()
            if torch.isnan(logits).any().item() or torch.isposinf(logits).any().item():
                raise LocalGenerationError("Non-finite positive/NaN generation scores")
            log_p = torch.log_softmax(logits, dim=-1)
            if torch.isnan(log_p).any().item():
                raise LocalGenerationError("No finite probability distribution (all scores -inf)")
            terms = torch.where(torch.isfinite(log_p), log_p.exp() * log_p, torch.zeros_like(log_p))
            entropy = (-terms.sum(dim=-1)).clamp_min(0).item()
            if not math.isfinite(entropy):
                raise LocalGenerationError("Entropy is not finite")
            values.append(entropy)
    return values, math.fsum(values) / len(values), vocab_size


def strict_object(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Non-finite JSON constant")

    result = json.loads(text, object_pairs_hook=unique, parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")
    return result


@dataclass
class LocalDecision:
    plan: str | None
    action: str
    generation: dict
    action_id: int | None = None


def parse_action_id_output(text):
    """Parse labels/decimal ID only. No fuzzy command matching or output repair."""
    if not isinstance(text, str):
        raise ValueError("Expected textual PLAN and ACTION_ID")
    if len(re.findall(r"(?m)^[ \t]*PLAN:", text)) != 1 or len(re.findall(r"(?m)^[ \t]*ACTION_ID:", text)) != 1:
        raise ValueError("Expected exactly one PLAN and one ACTION_ID field")
    match = re.fullmatch(r"[ \t\r\n]*PLAN:[ \t]*(?P<plan>.*?)\r?\n[ \t]*ACTION_ID:[ \t]*(?P<id>[+-]?[0-9]+)[ \t]*(?:\r?\n[ \t]*)*", text, re.DOTALL)
    if not match or not match.group("plan").strip():
        raise ValueError("Expected PLAN text followed by ACTION_ID with a decimal integer")
    return match.group("plan"), int(match.group("id"))


class LocalInvalidOutput(ValueError):
    def __init__(self, reason, generation, parsed=None):
        super().__init__(reason)
        self.generation, self.parsed = generation, parsed


class LocalLLMAgent:
    def __init__(self, config=None, *, model=None, tokenizer=None):
        self.config = config or LocalLLMConfig()
        if (model is None) != (tokenizer is None):
            raise LocalConfigurationError("Inject both model and tokenizer, or neither")
        self.model, self.tokenizer = model, tokenizer

    def load(self):
        """Called explicitly on the local PC; never invoked by import or --check-imports."""
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise LocalConfigurationError("NVIDIA CUDA is required. Install a matching PyTorch CUDA wheel and check the local driver first")
        if self.config.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise LocalConfigurationError("This GPU does not support bfloat16; configure float16")
        dtype = getattr(torch, self.config.compute_dtype)
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if self.config.load_in_4bit else None
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, revision=self.config.revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, revision=self.config.revision, trust_remote_code=False,
            use_safetensors=True, torch_dtype=dtype, quantization_config=quantization,
            device_map={"": self.config.device}, low_cpu_mem_usage=True)
        self.model.eval()
        return self

    def decide(self, goal, observation, previous_actions, admissible_actions, *, logits_path=None):
        import torch
        from transformers import GenerationConfig

        if self.model is None:
            raise LocalConfigurationError("Call agent.load() explicitly before local inference")
        if not admissible_actions or not all(isinstance(action, str) and action for action in admissible_actions):
            raise LocalGenerationError("No admissible actions supplied by the environment")
        if self.config.save_logits and logits_path is None:
            raise LocalConfigurationError("save_logits requires a unique output path")
        inputs = {"initial_goal": goal, "current_observation": observation,
                  "previous_actions": list(previous_actions),
                  "admissible_actions": [{"action_id": index, "action": action}
                                         for index, action in enumerate(admissible_actions)]}
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(inputs, ensure_ascii=False, allow_nan=False)}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=self.config.enable_thinking)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        prompt_length = encoded["input_ids"].shape[-1]
        model_limit = getattr(self.model.config, "max_position_embeddings", self.config.max_context_tokens)
        context_limit = min(self.config.max_context_tokens, model_limit) if isinstance(model_limit, int) else self.config.max_context_tokens
        if prompt_length + self.config.max_new_tokens > context_limit:
            raise LocalGenerationError("Input/history exceeds configured context budget; no silent history truncation")
        device = self.model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        model_generation = getattr(self.model, "generation_config", None)
        eos = getattr(model_generation, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = eos[0] if isinstance(eos, list) else eos
        # Start fresh: do not inherit checkpoint top-k, penalties, forced tokens,
        # sampling, bad-word filters, or constrained action decoding.
        generation_config = GenerationConfig(do_sample=False, num_beams=1, num_return_sequences=1,
                                              max_new_tokens=self.config.max_new_tokens,
                                              eos_token_id=eos, pad_token_id=pad,
                                              bos_token_id=self.tokenizer.bos_token_id, use_cache=True)
        with torch.inference_mode():
            output = self.model.generate(**encoded, generation_config=generation_config,
                                         use_model_defaults=False,
                                         return_dict_in_generate=True, output_scores=True,
                                         output_logits=self.config.save_logits)
        scores = output.scores
        token_ids = output.sequences[0, prompt_length:].detach().cpu().tolist()
        if len(token_ids) != len(scores):
            raise LocalGenerationError("Generated token IDs and token score counts do not align")
        entropies, mean, vocab_size = entropy_from_scores(scores)
        record = {"raw_text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
                  "generated_token_ids": token_ids, "token_entropies": entropies, "mean_entropy": mean,
                  "score_vocab_size": vocab_size, "entropy_unit": "nats", "entropy_source": "generate.scores, fresh unfiltered greedy config",
                  "prompt_token_count": prompt_length, "max_context_tokens": self.config.max_context_tokens,
                  "effective_context_limit": context_limit,
                  "eos_included": bool(token_ids and token_ids[-1] in (eos if isinstance(eos, list) else [eos])),
                  "generation_limit_reached": len(token_ids) >= self.config.max_new_tokens and not (token_ids and token_ids[-1] in (eos if isinstance(eos, list) else [eos])),
                  "resolved_revision": getattr(self.model.config, "_commit_hash", None), "logits_file": None}
        record["raw_model_output"] = record["raw_text"]
        record["action_selection_format"] = "action_id_v1"
        if self.config.save_logits:
            raw_logits = getattr(output, "logits", None)
            if raw_logits is None or len(raw_logits) != len(token_ids):
                raise LocalGenerationError("Requested raw logits were not returned by this Transformers implementation")
            path = Path(logits_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                torch.save({"generated_token_ids": token_ids, "logits_kind": "unprocessed generate.logits",
                            "token_logits": [value.detach().cpu() for value in raw_logits]}, stream)
            record["logits_file"] = str(path)
        del output, scores
        if record["generation_limit_reached"]:
            raise LocalInvalidOutput("Generation token limit reached without EOS; no partial output is executed", record)
        return self.validate_generation(record, admissible_actions)

    @staticmethod
    def validate_generation(record, admissible_actions):
        try:
            plan, action_id = parse_action_id_output(record["raw_text"])
        except ValueError:
            raise LocalInvalidOutput("Could not parse PLAN/ACTION_ID; no repair or expert replacement", record) from None
        parsed = {"plan": plan, "action_id": action_id, "action": None}
        if not 0 <= action_id < len(admissible_actions):
            raise LocalInvalidOutput("ACTION_ID outside current admissible action range; no substitution", record, parsed)
        action = admissible_actions[action_id]
        return LocalDecision(plan, action, record, action_id)
