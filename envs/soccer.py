# envs/soccer.py

import os
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch, quat_from_euler_xyz
import torch
import numpy as np
# A herança agora vem da sua BaseTask real
from .base_task import BaseTask

assert gymtorch

class Soccer(BaseTask):

    def __init__(self, cfg):
        # 1. CHAMADA CRUCIAL: Inicializa a BaseTask primeiro.
        # Ela vai criar self.sim, self.gym, self.device, self.viewer, etc.
        super().__init__(cfg)

        # Adiciona handles para os novos objetos
        self.ball_handles = []
        self.goal_handles = []
        
        # Cria os ambientes com robô, bola e gol
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        
        # Inicializa apenas os buffers essenciais para a simulação
        self._init_simulation_buffers()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        # --- 1. UNIFICAR AS OPÇÕES DE ASSET ---
        # Vamos usar TODAS as opções de asset que o T1 usa, lendo do nosso cfg.
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT # Mantemos esforço para teste
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]
        # Adicionando as opções que faltavam para estabilidade:
        if "density" in asset_cfg: asset_options.density = asset_cfg["density"]
        if "angular_damping" in asset_cfg: asset_options.angular_damping = asset_cfg["angular_damping"]
        if "linear_damping" in asset_cfg: asset_options.linear_damping = asset_cfg["linear_damping"]
        if "max_angular_velocity" in asset_cfg: asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        if "max_linear_velocity" in asset_cfg: asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        if "armature" in asset_cfg: asset_options.armature = asset_cfg["armature"]
        if "thickness" in asset_cfg: asset_options.thickness = asset_cfg["thickness"]
        
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        
        # ... (código para criar a bola e o gol permanece o mesmo)
        ball_options = gymapi.AssetOptions()
        ball_options.disable_gravity = False
        ball_radius = 0.15
        ball_asset = self.gym.create_sphere(self.sim, ball_radius, ball_options)
        goal_options = gymapi.AssetOptions()
        goal_options.fix_base_link = True
        goal_dims = gymapi.Vec3(0.1, 1.8, 1.2)
        goal_asset = self.gym.create_box(self.sim, goal_dims.x, goal_dims.y, goal_dims.z, goal_options)
        
        # Posição inicial do robô agora é lida do cfg para consistência
        init_state_cfg = self.cfg["init_state"]
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(*init_state_cfg["pos"])
        
        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(1.0, 0.0, ball_radius)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = gymapi.Vec3(4.0, 0.0, goal_dims.z / 2.0)

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        
        self.envs = []
        self.robot_actor_handles = []

        print(f"Criando {self.num_envs} ambientes...")
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)

            # --- 2. ADICIONAR A CONFIGURAÇÃO PÓS-CRIAÇÃO ---
            # Este bloco é a correção principal.
            
            # Adicionar o Robô
            robot_handle = self.gym.create_actor(
                env_handle, robot_asset, robot_start_pose, 
                asset_cfg["name"], i, asset_cfg["self_collisions"], 0
            )
            self.robot_actor_handles.append(robot_handle)

            # Define as propriedades do corpo (massa, inércia, CoM)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            # Por enquanto, não vamos randomizar, apenas garantir que elas sejam aplicadas
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)

            # Define as propriedades de colisão (fricção, restituição)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, robot_handle)
            # Por enquanto, não vamos randomizar, apenas garantir que elas sejam aplicadas
            self.gym.set_actor_rigid_shape_properties(env_handle, robot_handle, shape_props)
            
            # Adicionar a Bola
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_start_pose, "ball", i, 0)
            self.ball_handles.append(ball_handle)

            # Adicionar o Gol
            goal_handle = self.gym.create_actor(env_handle, goal_asset, goal_start_pose, "goal", i, 0)
            self.goal_handles.append(goal_handle)

            # Configuração das juntas (DOFs)
            dof_props = self.gym.get_actor_dof_properties(env_handle, robot_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"].fill(0.0)
            dof_props["damping"].fill(5.0) # Reduzi um pouco o damping para começar
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
    
    def _init_simulation_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # 2. ATENÇÃO AQUI: O tensor `root_states` agora é criado pela BaseTask
        # e usado no método `render`. Precisamos garantir que ele tenha a forma correta
        # e que o nosso código também o referencie.
        # A BaseTask não o expõe diretamente, mas o usa internamente.
        # Nós criamos o nosso próprio wrapper para controlar os atores.
        self.root_states_tensor = gymtorch.wrap_tensor(actor_root_state)

        self.num_actors = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        self.robot_root_states = self.root_states_tensor.view(self.num_envs, self.num_actors, 13)[:, 0, :]
        self.ball_root_states = self.root_states_tensor.view(self.num_envs, self.num_actors, 13)[:, 1, :]
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)

    def step(self, actions):
        self.torques[:] = torch.clip(actions, -100.0, 100.0)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        
        self.gym.simulate(self.sim)
        self.root_states = self.root_states_tensor
        self.render()

    # As funções abaixo são exigidas pela estrutura, mas não precisam fazer nada por enquanto.
    def reset(self):
        pass

    def _compute_reward(self):
        pass

    def _compute_observations(self):
        pass

    def _check_termination(self):
        pass