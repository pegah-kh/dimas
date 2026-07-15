from lerobot.configs.eval import EvalPipelineConfig
from dataclasses import dataclass, field
from typing import List



@dataclass
class FeatureExtractionPipelineConfig(EvalPipelineConfig):
    def __post_init__(self) -> None:
        super().__post_init__()
        
    save_features: bool = True
    hook_names: List[str] = field(default_factory=list)
    modules_to_hook: List[List[str]] = field(default_factory=list)
    exact_match_modules_to_hook: bool = True
    hook_postprocessing_name: str = "save_hidden_states"
    token_idx: list[int] = field(default_factory=lambda: [-1, 0])
    shift_vector_path: list[str] = field(default_factory=list)  # list of paths to shift vectors, should be same length as hook_names
    steering_alpha: float = 1.0
    start_prompt_token_idx_steering: int = 0
    steps_to_steer: list[int] = field(default_factory=lambda: list(range(10)))

    after_step_hooks: List[str] = field(default_factory=list)
    after_env_post_hooks: str | None = None          # hook name from registry
    action_hook_dims: list[int] = field(default_factory=lambda: [0, 1, 2])
    action_hook_scale: float = 1.2
    action_hook_threshold: float = 0.05              # for adaptive
    action_hook_scale_below: float = 1.0             # for adaptive
    action_hook_deadzone: float = 0.1                # for deadzone
    action_hook_scale_pos: float = 1.2               # for signed
    action_hook_scale_neg: float = 1.0               # for signed
    action_hook_gripper_dim: int = 6                 # for gripper-gated
    action_hook_gripper_threshold: float = 0.5       # for gripper-gated



    replaced_instruction: str | None = None
    pickup_filter: bool = False
    use_grasping_filter: bool = True
