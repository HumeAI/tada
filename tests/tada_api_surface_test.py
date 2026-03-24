import ast
from pathlib import Path


SOURCE_PATH = Path(__file__).resolve().parents[1] / "tada/modules/tada.py"


EXPECTED_INFERENCE_OPTION_DEFAULTS = {
    "text_do_sample": True,
    "text_temperature": 0.6,
    "text_top_k": 0,
    "text_top_p": 0.9,
    "text_repetition_penalty": 1.1,
    "acoustic_cfg_scale": 1.6,
    "duration_cfg_scale": 1.0,
    "cfg_schedule": "cosine",
    "noise_temperature": 0.9,
    "num_flow_matching_steps": 10,
    "time_schedule": "logsnr",
    "num_acoustic_candidates": 1,
    "scorer": "likelihood",
    "spkr_verification_weight": 1.0,
    "speed_up_factor": None,
    "negative_condition_source": "negative_step_output",
    "text_only_logit_scale": 0.0,
}

EXPECTED_GENERATE_PARAMS = [
    "self",
    "prompt",
    "text",
    "num_transition_steps",
    "num_extra_steps",
    "system_prompt",
    "user_turn_prompt",
    "inference_options",
    "use_text_in_prompt",
    "normalize_text",
    "verbose",
]


MODULE = ast.parse(SOURCE_PATH.read_text())


def _find_class(name: str) -> ast.ClassDef:
    for node in MODULE.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name} not found in {SOURCE_PATH}")


def _literal_value(node: ast.expr):
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"Unsupported default value AST: {ast.dump(node)}")


def test_inference_options_defaults_match_readme_facing_api_surface():
    inference_options = _find_class("InferenceOptions")

    assert any(isinstance(decorator, ast.Name) and decorator.id == "dataclass" for decorator in inference_options.decorator_list)

    actual_defaults = {}
    for node in inference_options.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            actual_defaults[node.target.id] = _literal_value(node.value)

    assert actual_defaults == EXPECTED_INFERENCE_OPTION_DEFAULTS


def test_generate_accepts_inference_options_parameter_with_dataclass_default():
    tada_for_causal_lm = _find_class("TadaForCausalLM")

    generate = next(
        node for node in tada_for_causal_lm.body if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )

    actual_params = [arg.arg for arg in generate.args.args]
    assert actual_params == EXPECTED_GENERATE_PARAMS

    defaults_by_name = dict(zip(actual_params[-len(generate.args.defaults) :], generate.args.defaults))
    assert ast.unparse(defaults_by_name["inference_options"]) == "InferenceOptions()"
