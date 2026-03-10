# hydra_eval.py

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import socket
import actoris_harena.api as ag_ar

from tool.utils import resolve_save_root

from env.parallel import Parallel

@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):

    new_save_root = resolve_save_root(cfg.save_root)
    print(f"[tool.hydra_train] Using Save Root: {cfg.save_root}")

    # Update the config object (must unset 'struct' to modify)
    OmegaConf.set_struct(cfg, False)
    cfg.save_root = new_save_root
    OmegaConf.set_struct(cfg, True)
    # -------------------------------------

    print("[tool.hydra_train] --- Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("[tool.hydra_train] ---------------------")

    save_dir = os.path.join(cfg.save_root, cfg.exp_name)

    agent = ag_ar.build_agent(
        cfg.agent.name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir)
    
    arena = ag_ar.build_arena(
        cfg.arena.name, 
        cfg.arena,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir)

    res = ag_ar.train_and_evaluate_single(
        agent,
        arena,
        cfg.agent.validation_interval,
        cfg.agent.total_update_steps,
        eval_last_check=True,
        eval_best_check=True,
        policy_terminate=False,
        env_success_stop=False)



if __name__ == "__main__":
    main()
