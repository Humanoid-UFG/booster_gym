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
        super().__init__(cfg)

        self.ball_handles = []
        self.goal_handles = []
        
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_simulation_buffers()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        # --- 1. CARREGAR ASSETS ---
        # Asset do Robô
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]
        if "density" in asset_cfg: asset_options.density = asset_cfg["density"]
        if "angular_damping" in asset_cfg: asset_options.angular_damping = asset_cfg["angular_damping"]
        if "linear_damping" in asset_cfg: asset_options.linear_damping = asset_cfg["linear_damping"]
        if "max_angular_velocity" in asset_cfg: asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        if "max_linear_velocity" in asset_cfg: asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        if "armature" in asset_cfg: asset_options.armature = asset_cfg["armature"]
        if "thickness" in asset_cfg: asset_options.thickness = asset_cfg["thickness"]
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        
        # Asset da Bola
        ball_options = gymapi.AssetOptions()
        ball_options.disable_gravity = False
        self.ball_radius = 0.11 # Salva o raio para uso posterior
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)
        
        # Assets do Gol
        goal_options = gymapi.AssetOptions()
        goal_options.fix_base_link = True
        goal_width = 1.8
        goal_height = 1.2
        post_thickness = 0.1
        post_dims = gymapi.Vec3(post_thickness, post_thickness, goal_height)
        post_asset = self.gym.create_box(self.sim, post_dims.x, post_dims.y, post_dims.z, goal_options)
        crossbar_dims = gymapi.Vec3(post_thickness, goal_width, post_thickness)
        crossbar_asset = self.gym.create_box(self.sim, crossbar_dims.x, crossbar_dims.y, crossbar_dims.z, goal_options)

        # --- 2. DEFINIR POSES RELATIVAS (DENTRO DE UM AMBIENTE) ---
        init_state_cfg = self.cfg["init_state"]
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(*init_state_cfg["pos"])
        robot_start_pose.r = gymapi.Quat(*init_state_cfg["rot"])

        ball_start_pose = gymapi.Transform()
        ball_start_pose.p = gymapi.Vec3(1.0, 0.0, self.ball_radius)
        
        goal_x_pos = 4.0
        left_post_pose = gymapi.Transform()
        left_post_pose.p = gymapi.Vec3(goal_x_pos, -goal_width / 2, goal_height / 2)
        right_post_pose = gymapi.Transform()
        right_post_pose.p = gymapi.Vec3(goal_x_pos, goal_width / 2, goal_height / 2)
        crossbar_pose = gymapi.Transform()
        crossbar_pose.p = gymapi.Vec3(goal_x_pos, 0, goal_height)
        crossbar_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # --- 3. CRIAR AMBIENTES E ATORES EM UMA GRADE ---
        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.robot_actor_handles = []

        print(f"Criando {self.num_envs} ambientes...")
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)
            
            # Pega a origem deste ambiente específico (que é um tensor)
            env_origin_tensor = self.env_origins[i]
            # Converte o tensor para um objeto Vec3 que o Isaac Gym entende
            env_origin_vec3 = gymapi.Vec3(env_origin_tensor[0], env_origin_tensor[1], env_origin_tensor[2])

            # Agora, a soma é entre dois objetos Vec3, o que é válido
            robot_pose = gymapi.Transform(p=robot_start_pose.p + env_origin_vec3, r=robot_start_pose.r)
            ball_pose = gymapi.Transform(p=ball_start_pose.p + env_origin_vec3, r=ball_start_pose.r)
            l_post_pose = gymapi.Transform(p=left_post_pose.p + env_origin_vec3, r=left_post_pose.r)
            r_post_pose = gymapi.Transform(p=right_post_pose.p + env_origin_vec3, r=right_post_pose.r)
            c_bar_pose = gymapi.Transform(p=crossbar_pose.p + env_origin_vec3, r=crossbar_pose.r)
            
            # Adicionar Robô
            robot_handle = self.gym.create_actor(env_handle, robot_asset, robot_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            self.robot_actor_handles.append(robot_handle)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, robot_handle)
            self.gym.set_actor_rigid_shape_properties(env_handle, robot_handle, shape_props)
            dof_props = self.gym.get_actor_dof_properties(env_handle, robot_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"].fill(0.0)
            dof_props["damping"].fill(5.0)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)

            # Adicionar Bola
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", i, 0)
            self.ball_handles.append(ball_handle)
            ball_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            ball_props[0].mass = 0.43
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_props)

            # Adicionar Gol
            left_post_handle = self.gym.create_actor(env_handle, post_asset, l_post_pose, "left_post", i, 0)
            right_post_handle = self.gym.create_actor(env_handle, post_asset, r_post_pose, "right_post", i, 0)
            crossbar_handle = self.gym.create_actor(env_handle, crossbar_asset, c_bar_pose, "crossbar", i, 0)
            self.goal_handles.extend([left_post_handle, right_post_handle, crossbar_handle])

    def _get_env_origins(self):
        """Calcula as posições de origem para cada ambiente, organizando-os em uma grade."""
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        spacing = self.cfg["env"]["env_spacing"]
        num_cols = int(np.sqrt(self.num_envs))
        num_rows = int(np.ceil(self.num_envs / num_cols))
        xx, yy = torch.meshgrid(torch.arange(float(num_rows)), torch.arange(float(num_cols)), indexing="ij")
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = 0.0
    
    def _init_simulation_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states_tensor = gymtorch.wrap_tensor(actor_root_state)

        # Cada ambiente agora tem 1 robô + 1 bola + 3 partes do gol = 5 atores
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
        self.root_states = self.root_states_tensor # Garante que a BaseTask possa renderizar
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