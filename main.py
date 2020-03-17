from training_loops.training_loop import main
from omegaconf import OmegaConf
import wandb
import sys
import os

name = [sys.argv[1] if len(sys.argv) == 2 else None][0]
main_cnf = OmegaConf.load('configs/ant_default/ant_main_conf.yaml')
agent_cnf = OmegaConf.load('configs/ant_default/ant_agent_conf.yaml')
# Parameters of second cnf file overwrite those of first
cnf = OmegaConf.merge(main_cnf, agent_cnf)
if name:
    exp_cnf = OmegaConf.load(f'experiments/{name}.yaml')
    cnf = OmegaConf.merge(cnf, exp_cnf)
cnf.merge_with_cli()

config = {**cnf.main, **cnf.agent, **cnf.buffer, **cnf.agent.sub_model}
if cnf.main.log:
    wandb.init(project=cnf.project, entity=cnf.entity, config=config)
main(cnf)
