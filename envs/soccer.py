import os
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import to_torch, quat_from_euler_xyz, torch_rand_float
import torch
import numpy as np
from .base_task import BaseTask

assert gymtorch

class Soccer(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)

        # --- Alterações Início ---
        self.add_goal = self.cfg["env"].get("add_goal", False)
        self.apply_initial_kick = self.cfg["env"].get("apply_initial_kick", False)
        self.kick_velocity = self.cfg["env"].get("kick_velocity", 10.0)  # Lido do cfg
        # --- Alterações Fim ---

        self.ball_handles = []
        self.goal_handles = []
        
        self.field_dims = gymapi.Vec2(10.0, 6.0)
        
        self.wall_thickness = 0.1
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_simulation_buffers()

        self._kick_applied = False

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        # --- 1. CARREGAR ASSETS ---
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
        
        ball_options = gymapi.AssetOptions()
        ball_options.disable_gravity = False
        self.ball_radius = 0.11
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)
        
        # Definir dimensões do gol (necessário para paredes, mesmo se o gol não for criado)
        goal_width, goal_height, post_thickness = 1.8, 1.2, 0.1

        if self.add_goal:
            goal_options = gymapi.AssetOptions()
            goal_options.fix_base_link = True
            post_asset = self.gym.create_box(self.sim, post_thickness, post_thickness, goal_height, goal_options)
            crossbar_asset = self.gym.create_box(self.sim, post_thickness, goal_width, post_thickness, goal_options)
        
        # --- CORREÇÃO 1: Geometria das Paredes ---
        self.goal_x_pos = 4.0
        back_wall_x = self.goal_x_pos - self.field_dims.x
        goal_wall_x = self.goal_x_pos + 1.0 # Posição da parede atrás do gol

        wall_options = gymapi.AssetOptions()
        wall_options.fix_base_link = True
        wall_height = goal_height * 1.5 # Agora goal_height está sempre definida

        # Novo comprimento e centro para as paredes laterais
        side_wall_length = goal_wall_x - back_wall_x
        side_wall_center_x = (goal_wall_x + back_wall_x) / 2

        side_wall_asset = self.gym.create_box(self.sim, side_wall_length, self.wall_thickness, wall_height, wall_options)
        back_wall_asset = self.gym.create_box(self.sim, self.wall_thickness, self.field_dims.y, wall_height, wall_options)

        # --- 2. DEFINIR POSES RELATIVAS ---
        self.robot_initial_z = self.cfg["init_state"]["pos"][2]
        self.initial_robot_quat = gymapi.Quat(*self.cfg["init_state"]["rot"])
        
        if self.add_goal:
            goal_physical_x = self.goal_x_pos + post_thickness / 2 # post_thickness está sempre definida
            left_post_pose = gymapi.Transform(p=gymapi.Vec3(goal_physical_x, -goal_width / 2, goal_height / 2))
            right_post_pose = gymapi.Transform(p=gymapi.Vec3(goal_physical_x, goal_width / 2, goal_height / 2))
            crossbar_pose = gymapi.Transform(p=gymapi.Vec3(goal_physical_x, 0, goal_height), r=gymapi.Quat(0,0,0,1))
        
        left_wall_y = -self.field_dims.y / 2
        right_wall_y = self.field_dims.y / 2
        
        back_wall_pose = gymapi.Transform(p=gymapi.Vec3(back_wall_x, 0, wall_height / 2))
        left_wall_pose = gymapi.Transform(p=gymapi.Vec3(side_wall_center_x, left_wall_y, wall_height / 2))
        right_wall_pose = gymapi.Transform(p=gymapi.Vec3(side_wall_center_x, right_wall_y, wall_height / 2))
        goal_wall_pose = gymapi.Transform(p=gymapi.Vec3(goal_wall_x, 0, wall_height / 2))

        # --- 3. CRIAR AMBIENTES E ATORES ---
        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.robot_actor_handles = []

        field_min_x = back_wall_x
        field_max_x = self.goal_x_pos
        field_min_y = left_wall_y
        field_max_y = right_wall_y

        print(f"Criando {self.num_envs} ambientes...")
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)
            
            env_origin_vec3 = gymapi.Vec3(self.env_origins[i,0], self.env_origins[i,1], self.env_origins[i,2])

            start_pos_robot = gymapi.Vec3(np.random.uniform(field_min_x, field_max_x), np.random.uniform(field_min_y, field_max_y), self.robot_initial_z)
            start_pos_ball = gymapi.Vec3(np.random.uniform(field_min_x, field_max_x), np.random.uniform(field_min_y, field_max_y), self.ball_radius)
            
            robot_pose = gymapi.Transform(p=start_pos_robot + env_origin_vec3, r=self.initial_robot_quat)
            ball_pose = gymapi.Transform(p=start_pos_ball + env_origin_vec3)
            
            if self.add_goal:
                l_post_pose = gymapi.Transform(p=left_post_pose.p + env_origin_vec3, r=left_post_pose.r)
                r_post_pose = gymapi.Transform(p=right_post_pose.p + env_origin_vec3, r=right_post_pose.r)
                c_bar_pose = gymapi.Transform(p=crossbar_pose.p + env_origin_vec3, r=crossbar_pose.r)

            b_wall_pose = gymapi.Transform(p=back_wall_pose.p + env_origin_vec3, r=back_wall_pose.r)
            l_wall_pose = gymapi.Transform(p=left_wall_pose.p + env_origin_vec3, r=left_wall_pose.r)
            r_wall_pose = gymapi.Transform(p=right_wall_pose.p + env_origin_vec3, r=right_wall_pose.r)
            g_wall_pose = gymapi.Transform(p=goal_wall_pose.p + env_origin_vec3, r=goal_wall_pose.r)
            
            collision_group = 0
            collision_filter = -1

            # Criando atores com os filtros corretos
            robot_handle = self.gym.create_actor(env_handle, robot_asset, robot_pose, "robot", collision_group, collision_filter)
            ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", collision_group, collision_filter)

            if self.add_goal:
                left_post_handle = self.gym.create_actor(env_handle, post_asset, l_post_pose, "left_post", collision_group, collision_filter)
                right_post_handle = self.gym.create_actor(env_handle, post_asset, r_post_pose, "right_post", collision_group, collision_filter)
                crossbar_handle = self.gym.create_actor(env_handle, crossbar_asset, c_bar_pose, "crossbar", collision_group, collision_filter)

            self.gym.create_actor(env_handle, back_wall_asset, b_wall_pose, "back_wall", collision_group, collision_filter)
            self.gym.create_actor(env_handle, side_wall_asset, l_wall_pose, "left_wall", collision_group, collision_filter)
            self.gym.create_actor(env_handle, side_wall_asset, r_wall_pose, "right_wall", collision_group, collision_filter)
            self.gym.create_actor(env_handle, back_wall_asset, g_wall_pose, "goal_wall", collision_group, collision_filter)
            
            # Configurações dos atores
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
            
            self.ball_handles.append(ball_handle)
            ball_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            ball_props[0].mass = 0.43
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_props)
            
            if self.add_goal:
                self.goal_handles.extend([left_post_handle, right_post_handle, crossbar_handle])

    def _get_env_origins(self):
        """
        Calcula as origens para cada ambiente, garantindo que eles não se sobreponham.
        O espaçamento é baseado nas dimensões totais de um único ambiente (campo + paredes).
        """
        # --- 1. Calcular as dimensões totais de um ambiente ---
        self.goal_x_pos = 4.0
        back_wall_x = self.goal_x_pos - self.field_dims.x
        goal_wall_x = self.goal_x_pos + 1.0

        # Encontra os limites extremos em X e Y, incluindo a espessura das paredes
        min_x = back_wall_x - self.wall_thickness / 2
        max_x = goal_wall_x + self.wall_thickness / 2
        min_y = -self.field_dims.y / 2 - self.wall_thickness / 2
        max_y = self.field_dims.y / 2 + self.wall_thickness / 2

        # Largura e profundidade total de um ambiente
        env_width = max_x - min_x
        env_depth = max_y - min_y

        # Define um buffer para dar um espaço extra entre os ambientes
        spacing_buffer = 2.0  # Você pode ajustar este valor

        spacing_x = env_width + spacing_buffer
        spacing_y = env_depth + spacing_buffer
        
        print(f"Espaçamento dinâmico calculado: X={spacing_x:.2f}, Y={spacing_y:.2f}")

        # --- 2. Criar a grade de origens com o espaçamento calculado ---
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        
        num_cols = int(np.sqrt(self.num_envs))
        
        # Otimização para criar a grade de forma eficiente
        env_ids = torch.arange(self.num_envs, device=self.device)
        env_rows = torch.div(env_ids, num_cols, rounding_mode='floor')
        env_cols = torch.remainder(env_ids, num_cols)

        self.env_origins[:, 0] = env_cols * spacing_x
        self.env_origins[:, 1] = env_rows * spacing_y
        self.env_origins[:, 2] = 0.0
    
    def _init_simulation_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states_tensor = gymtorch.wrap_tensor(actor_root_state)

        self.num_actors = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        self.robot_root_states = self.root_states_tensor.view(self.num_envs, self.num_actors, 13)[:, 0, :]
        self.ball_root_states = self.root_states_tensor.view(self.num_envs, self.num_actors, 13)[:, 1, :]
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)

    def step(self, actions):
        if self.apply_initial_kick and not self._kick_applied:
            goal_center = torch.tensor([self.goal_x_pos, 0, self.ball_radius], device=self.device)
            ball_positions_relative = self.ball_root_states[:, :3] - self.env_origins
            
            direction_to_goal = goal_center - ball_positions_relative
            direction_to_goal = torch.nn.functional.normalize(direction_to_goal, p=2, dim=1)
            
            kick_vec = direction_to_goal * self.kick_velocity
            
            self.ball_root_states[:, 7:10] = kick_vec
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_tensor))
            self._kick_applied = True

        self.torques[:] = torch.clip(actions, -100.0, 100.0)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        
        self.gym.simulate(self.sim)
        self.root_states = self.root_states_tensor
        self.render()

    def reset(self):
        pass

    def _compute_reward(self):
        pass

    def _compute_observations(self):
        pass

    def _check_termination(self):
        pass