# play_soccer.py

import isaacgym
import torch
import yaml # 1. Importe a biblioteca YAML
from envs.soccer import Soccer

def run_visualization():
    # 2. REMOVA o dicionário 'full_cfg' hardcoded.

    # 3. ADICIONE o código para carregar o arquivo YAML.
    cfg_file = "envs/soccer.yaml"
    print(f"Carregando configuração de: {cfg_file}")
    with open(cfg_file, 'r') as f:
        # Usamos safe_load que é mais seguro
        cfg = yaml.safe_load(f)

    # O resto do código permanece o mesmo, agora usando a 'cfg' carregada do arquivo.
    env = Soccer(cfg=cfg)
    
    num_actions = env.num_dofs
    print(f"Ambiente criado. Número de ações: {num_actions}")

    while not env.gym.query_viewer_has_closed(env.viewer):
        actions = torch.zeros(env.num_envs, num_actions, device=env.device)
        env.step(actions)

    print("Fechando...")
    env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)

if __name__ == "__main__":
    run_visualization()