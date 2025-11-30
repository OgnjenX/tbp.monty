from __future__ import annotations

import copy
import os
from dataclasses import asdict

from benchmarks.configs.defaults import (
    min_eval_steps,
)
from benchmarks.configs.ycb_experiments import (
    lower_max_nneighbors_surf_1lm_config,
    model_path_10distinctobj,
    test_rotations_all,
)
from tbp.monty.frameworks.config_utils.config_args import (
    CSVLoggingConfig,
    MontyArgs,
    MotorSystemConfigCurInformedSurfaceGoalStateDriven,
    SurfaceAndViewSOTAMontyConfig,
)
from tbp.monty.frameworks.config_utils.make_dataset_configs import (
    EnvironmentDataloaderPerObjectArgs,
    EvalExperimentArgs,
    PredefinedObjectInitializer,
    get_object_names_by_idx,
)
from tbp.monty.frameworks.environments import embodied_data as ED
from tbp.monty.frameworks.environments.ycb import DISTINCT_OBJECTS
from tbp.monty.frameworks.experiments import MontyObjectRecognitionExperiment
from tbp.monty.frameworks.meta import MetaController
from tbp.monty.simulators.habitat.configs import (
    SurfaceViewFinderMountHabitatDatasetArgs,
)

"""
YCB experiment with MetaController for creativity/perspective shifting.

This script mirrors the surface-agent YCB evaluation but adds MetaController
to monitor entropy/slope and perturb feature weights or inject transform jitter
when stagnation is detected.

Usage:
    python -m benchmarks.configs.ycb_meta

Outputs:
    - Standard CSV/WandB logs
    - Optional simple plot of entropy/slope and intervention markers
"""

# ---------------------------- Base Config ----------------------------

base_config_10distinctobj_surf_agent = {
    "experiment_class": MontyObjectRecognitionExperiment,
    "experiment_args": EvalExperimentArgs(
        model_name_or_path=model_path_10distinctobj,
        n_eval_epochs=len(test_rotations_all),
        max_total_steps=5000,
    ),
    "logging_config": CSVLoggingConfig(
        python_log_level="INFO",
        # Uncomment to enable WandB if desired
        # wandb_group="ycb_meta",
    ),
    "monty_config": SurfaceAndViewSOTAMontyConfig(
        learning_module_configs=lower_max_nneighbors_surf_1lm_config,
        motor_system_config=MotorSystemConfigCurInformedSurfaceGoalStateDriven(),
        monty_args=MontyArgs(min_eval_steps=min_eval_steps),
    ),
    "dataset_args": SurfaceViewFinderMountHabitatDatasetArgs(),
    "eval_dataloader_class": ED.InformedEnvironmentDataLoader,
    "eval_dataloader_args": EnvironmentDataloaderPerObjectArgs(
        object_names=get_object_names_by_idx(0, 10, object_list=DISTINCT_OBJECTS),
        object_init_sampler=PredefinedObjectInitializer(rotations=test_rotations_all),
    ),
}


# ---------------------------- Experiment Wrapper ----------------------------

class MetaYcbExperiment(MontyObjectRecognitionExperiment):
    """
    Wrapper that attaches MetaController to the Monty instance before training/eval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta = None

    def setup(self):
        # Ensure the base experiment is fully initialized before attaching MetaController
        if not hasattr(self, "model"):
            # Let the experiment runner initialize; if not, call base setup
            self.setup_experiment(self.config)
        # Attach MetaController with conservative thresholds for demo
        self._meta = MetaController(
            self.model,
            entropy_min=0.2,
            slope_threshold=1e-3,
            stagnation_window=5,
            ema_alpha=0.3,
            cooldown_steps=10,
            perturb_amp=0.15,
            weight_ttl=20,
            transform_prob=0.3,
            transform_loc_sigma=0.01,
            transform_rot_sigma_deg=2.0,
        )

    def teardown(self):
        if self._meta:
            self._meta.detach()
        # No base teardown to call; MontyObjectRecognitionExperiment has none

    def post_epoch(self):
        """Override post_epoch to dump meta history after each epoch."""
        super().post_epoch()
        # Optional: dump meta history for external plotting
        if self._meta:
            import json
            history_path = os.path.join(self.output_dir, "meta_history.json")
            with open(history_path, "w") as f:
                json.dump(self._meta.get_history(), f, indent=2)


# ---------------------------- Meta Variant ----------------------------

meta_config_10distinctobj_surf_agent = copy.deepcopy(base_config_10distinctobj_surf_agent)
# Override experiment class to inject MetaController after Monty construction (defined below)
meta_config_10distinctobj_surf_agent["experiment_class"] = MetaYcbExperiment

# ---------------------------- Exposed Config ----------------------------

# Register the new config name in the YcbExperiments dataclass
from benchmarks.configs.names import YcbExperiments
import dataclasses

# Add meta_config_10distinctobj_surf_agent field if not present
if not hasattr(YcbExperiments, "meta_config_10distinctobj_surf_agent"):
    YcbExperiments = dataclasses.make_dataclass(
        "YcbExperiments",
        [(f.name, f.type, f.default) for f in dataclasses.fields(YcbExperiments)]
        + [("meta_config_10distinctobj_surf_agent", dict, dataclasses.MISSING)],
        bases=(YcbExperiments,),
    )

experiments = YcbExperiments(
    # Keep existing fields empty; we only need to add our new one
    **{f.name: None for f in dataclasses.fields(YcbExperiments) if f.name != "meta_config_10distinctobj_surf_agent"},
    meta_config_10distinctobj_surf_agent=meta_config_10distinctobj_surf_agent,
)
CONFIGS = asdict(experiments)

if __name__ == "__main__":
    # Simple manual run for quick sanity check
    from tbp.monty.frameworks.run import run

    cfg = CONFIGS["meta_config_10distinctobj_surf_agent"]
    run(cfg)
