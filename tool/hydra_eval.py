# hydra_eval.py
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import socket
import actoris_harena.api as ag_ar

from registration.agent import register_agents
from registration.sim_arena import register_arenas
from registration.task import build_task
from tool.utils import resolve_save_root


# 1. Update config_path to point to the root 'conf' directory
@hydra.main(config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    
    register_agents()
    register_arenas()

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
    
    # 2. Extract Names (Fallback to Exp Name if not in sub-config)
    # If your agent yaml doesn't have a 'name' field, we use cfg.exp_name or a default
    agent_name = cfg.agent.get('name', cfg.exp_name) 
    arena_name = cfg.arena.get('name', 'default_arena')

    # 3. Build Agent
    print(f"[hydra eval] Building Agent: {agent_name}")
    agent = ag_ar.build_agent(
        agent_name, 
        cfg.agent,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir,
        disable_wandb=True
    )
    
    # 4. Build Arena
    print(f"[hydra eval] Building Arena: {arena_name}")
    arena = ag_ar.build_arena(
        arena_name, 
        cfg.arena,
        project_name=cfg.project_name,
        exp_name=cfg.exp_name,
        save_dir=save_dir
    )
    
    # 5. Build Task
    # Ensure build_task can handle the DictConfig object
    task = build_task(cfg.task)
    arena.set_task(task)

    # 6. Run Evaluation
    ag_ar.evaluate(
        agent,
        arena,
        checkpoint=-2, # Load best checkpoint
        policy_terminate=False,
        env_success_stop=False
    )

if __name__ == "__main__":
    main()