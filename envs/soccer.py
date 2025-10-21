import os
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    to_torch, 
    quat_rotate_inverse, 
    quat_from_euler_xyz, 
    torch_rand_float,
    get_axis_params,
    get_euler_xyz,
    quat_rotate,
)
import torch
import numpy as np
from .base_task import BaseTask
from .utils.camera import VirtualCamera 
from .utils.utils import TerminalVelocityControl

assert gymtorch

class Soccer(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.is_training_mode = self.cfg["env"].get("enable_training_mode", False)
        
        self.num_envs = self.cfg["env"]["num_envs"]

        if not self.is_training_mode:
            print("Modo de Operação: [PLAYER/DEPLOY]")
            self.camera = VirtualCamera(self.cfg, self.num_envs, self.device)
            self.enable_terminal_control = self.cfg["env"].get("enable_terminal_control", False)
            self.terminal_control = None

            if self.enable_terminal_control:
                print("Controle de velocidade pelo terminal [ATIVADO].")
                self.terminal_control = TerminalVelocityControl(self.device)
                self.terminal_control.start()
            else:
                print("Controle de velocidade pelo terminal [DESATIVADO].")

            policy_file = self.cfg["env"].get("stand_pose_file", "deploy/models/T1.pt")
            
            try:
                self.stand_policy = torch.load(policy_file, map_location=self.device)
                self.stand_policy.eval() 
                print(f"Política para ficar em pé '{policy_file}' carregada com sucesso.")
            except Exception as e:
                print(f"ERRO: Falha ao carregar a política '{policy_file}'. O robô não ficará em pé.")
                print(e)
                self.stand_policy = None 
        else:
            print("Modo de Operação: [TRAINER]")
            self.camera = None
            self.terminal_control = None
            self.stand_policy = None
        
        self.add_goal = self.cfg["env"].get("add_goal", False)
        self.apply_initial_kick = self.cfg["env"].get("apply_initial_kick", False)
        self.kick_velocity = self.cfg["env"].get("kick_velocity", 10.0)  

        self.randomize_init_pos = self.cfg["env"].get("randomize_init_pos", False) 
        
        self.dof_stiffness = None
        self.dof_damping = None
        
        control_cfg = self.cfg.get("control", {}) 
        self.action_scale = control_cfg.get("action_scale", 1.0) 
        print(f"Action Scale: {self.action_scale}")
        
        norm_cfg = self.cfg.get("normalization", {}) 
        self.norm_gravity_scale = norm_cfg.get("gravity", 1.0)
        self.norm_ang_vel_scale = norm_cfg.get("ang_vel", 1.0)
        self.norm_lin_vel_scale = norm_cfg.get("lin_vel", 1.0) 
        self.norm_dof_pos_scale = norm_cfg.get("dof_pos", 1.0)
        self.norm_dof_vel_scale = norm_cfg.get("dof_vel", 1.0)
        self.action_clip = norm_cfg.get("clip_actions", 100.0) 
        self.commands_scale = torch.tensor(
            [self.norm_lin_vel_scale, self.norm_lin_vel_scale, self.norm_ang_vel_scale],
            device=self.device, dtype=torch.float32
        )
        print("Normalization Scales Loaded.") 
        print(f"Action Clip: +/- {self.action_clip}")
 
        self.ball_handles = []
        self.goal_handles = []
        
        if not self.is_training_mode:
            self.field_dims = gymapi.Vec2(10.0, 6.0)
            self.wall_thickness = 0.1
        
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        
        self.step_count = 0 

        self._init_simulation_buffers()

        if self.is_training_mode:
            self._init_training_buffers()
            self._prepare_reward_function()
        else:
            self._kick_applied = False

    def _create_envs(self):
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
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        # CARREGAR stiffness e damping específicos por junta (como em T1.py)
        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        
        if "control" not in self.cfg or "stiffness" not in self.cfg["control"] or "damping" not in self.cfg["control"]:
             raise ValueError("cfg['control']['stiffness'] and cfg['control']['damping'] must be defined in the config file.")
        
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
                    break 
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} was not defined in cfg['control']['stiffness/damping']")
        print("Per-joint stiffness and damping loaded from CFG.")
        
        # Encontra o índice do link da cabeça/câmera... 
        self.head_link_name = self.cfg["camera"].get("link_name", "ERROR_NO_LINK_NAME_IN_CFG") 
        # ... (resto do código de encontrar link da câmera) ...
        self.head_link_index = self.gym.find_asset_rigid_body_index(robot_asset, self.head_link_name)
        if self.head_link_index == -1: print(f"ALERTA: Link da câmera '{self.head_link_name}' NÃO ENCONTRADO...")
        else: print(f"Link da câmera '{self.head_link_name}' encontrado no índice {self.head_link_index}.")

        if not self.is_training_mode:
            self._create_envs_player(robot_asset)
        else:
            self._create_envs_trainer(robot_asset)

    def _get_env_origins(self):
        if not self.is_training_mode:
            self._get_env_origins_player()
        else:
            self._get_env_origins_trainer()
            
    def step(self, actions):
        if self.is_training_mode:
            return self.step_train(actions)
        else:
            self.step_play(actions)
            return None, None, None, None 

    def _create_envs_player(self, robot_asset):
        print("Criando ambientes: Modo [PLAYER/DEPLOY]")
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

            # Posições iniciais (randomizadas ou fixas)
            if self.randomize_init_pos:
                start_pos_robot = gymapi.Vec3(np.random.uniform(field_min_x, field_max_x), np.random.uniform(field_min_y, field_max_y), self.robot_initial_z)
                start_pos_ball = gymapi.Vec3(np.random.uniform(field_min_x, field_max_x), np.random.uniform(field_min_y, field_max_y), self.ball_radius)
            else:
                center_x = (field_min_x + field_max_x) / 2.0
                center_y = (field_min_y + field_max_y) / 2.0
                start_pos_robot = gymapi.Vec3(center_x, center_y, self.robot_initial_z)
                start_pos_ball = gymapi.Vec3(center_x + 1.0, center_y, self.ball_radius)
            
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
            dof_props["stiffness"].fill(0.0) # Stiffness do motor é 0, o controle é feito pelo PD calculado
            dof_props["damping"].fill(0.0) # Damping do motor é 0, o controle é feito pelo PD calculado
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            
            self.ball_handles.append(ball_handle)
            ball_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
            ball_props[0].mass = 0.43
            self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_props)
            
            if self.add_goal:
                self.goal_handles.extend([left_post_handle, right_post_handle, crossbar_handle])

    def _get_env_origins(self):
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

    def step_play(self, actions):
        # Esta é a sua função step() original
        if self.enable_terminal_control:
            self.commands[:] = self.terminal_control.get_commands()
            
        if self.apply_initial_kick and not self._kick_applied:
            goal_center = torch.tensor([self.goal_x_pos, 0, self.ball_radius], device=self.device)
            ball_positions_relative = self.ball_root_states[:, :3] - self.env_origins
            
            direction_to_goal = goal_center - ball_positions_relative
            direction_to_goal = torch.nn.functional.normalize(direction_to_goal, p=2, dim=1)
            
            kick_vec = direction_to_goal * self.kick_velocity
            
            self.ball_root_states[:, 7:10] = kick_vec
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_tensor))
            self._kick_applied = True

        # --- Alterações (Correção PD Gains / Normalização) ---
        policy_actions_clipped = torch.zeros_like(self.dof_pos) 

        if self.stand_policy is not None:
            with torch.no_grad():
                projected_gravity = quat_rotate_inverse(self.robot_root_states[:, 3:7], self.gravity_vec)
                base_ang_vel = quat_rotate_inverse(self.robot_root_states[:, 3:7], self.robot_root_states[:, 10:13])

                cos_gait = (torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1)
                sin_gait = (torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1)
                
                last_actions_for_obs = self.dof_pos_targets 

                obs_batch = torch.cat([(projected_gravity * self.norm_gravity_scale),     
                                       (base_ang_vel * self.norm_ang_vel_scale),         
                                       (self.commands * self.commands_scale),            
                                       cos_gait,                                         
                                       sin_gait,                                         
                                       ((self.dof_pos - self.default_dof_pos) * self.norm_dof_pos_scale), 
                                       (self.dof_vel * self.norm_dof_vel_scale),         
                                       last_actions_for_obs                              
                                      ], dim=-1)
                
                policy_actions_raw = self.stand_policy(obs_batch) 
                
                policy_actions_clipped = torch.clip(policy_actions_raw, -self.action_clip, self.action_clip)

        # 6. O alvo é a pose default + (escala * offset CLIPADO da política)
        self.dof_pos_targets[:] = self.default_dof_pos + self.action_scale * policy_actions_clipped

        # 7. Calcular torques com os ganhos CORRETOS por junta
        torques = self.dof_stiffness * (self.dof_pos_targets - self.dof_pos) - self.dof_damping * self.dof_vel
        
        self.torques[:] = torch.clip(torques, -100.0, 100.0) 
        
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        
        self.gym.simulate(self.sim)

        self.gym.refresh_rigid_body_state_tensor(self.sim) 
        self.gym.refresh_dof_state_tensor(self.sim) 
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = self.root_states_tensor
        self.render()

    def _create_envs_trainer(self, robot_asset):
        print("Criando ambientes: Modo [TRAINER]")
        
        # Lógica de _get_env_origins do T1
        self._get_env_origins() # Chama o roteador, que chamará _get_env_origins_trainer
        
        # Lógica de base_init_state do T1
        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.robot_actor_handles = []
        
        print(f"Criando {self.num_envs} ambientes (Modo Trainer)...")
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            # T1 cria apenas um ator (robô)
            # Usando collision group 0, filter 0 (como em T1)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot", i, 0, 0) 
            
            dof_props = self.gym.get_actor_dof_properties(env_handle, actor_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"].fill(0.0) 
            dof_props["damping"].fill(0.0) 
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            
            self.envs.append(env_handle)
            self.robot_actor_handles.append(actor_handle)

        # Lógica de índices de contato do T1
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        if "rewards" in self.cfg and "terminate_contacts_on" in self.cfg["rewards"]:
            termination_contact_names = []
            for name in self.cfg["rewards"]["terminate_contacts_on"]:
                termination_contact_names.extend([s for s in body_names if name in s])
            self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
            for i in range(len(termination_contact_names)):
                self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])
        else:
            print("ALERTA: cfg['rewards']['terminate_contacts_on'] não definido. Terminador de colisão desativado.")
            self.termination_contact_indices = torch.empty(0, dtype=torch.long, device=self.device)

    def _get_env_origins_trainer(self):
        # Lógica de T1 (terreno plano)
        print("Calculando espaçamento (Trainer): Modo Grade Fixa (env_spacing)")
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
        
        spacing = self.cfg["env"].get("env_spacing", 5.0) # Pega do cfg, com fallback
        
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = 0.0 # Assumindo terreno plano

    def _init_training_buffers(self):
        # Buffers específicos de RL (copiados do T1)
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}
        
        # Buffers de estado do T1
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.robot_root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Buffers de estado computados (T1)
        self.base_lin_vel = quat_rotate_inverse(self.robot_root_states[:, 3:7], self.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.robot_root_states[:, 3:7], self.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.robot_root_states[:, 3:7], self.gravity_vec)
        
        print("Buffers específicos do modo [TRAINER] inicializados.")

    def _prepare_reward_function(self):
        # Lógica de T1 (a ser preenchida com as funções de recompensa)
        print("Preparando funções de recompensa (Modo Trainer)...")
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            # Precisamos verificar se a função existe ANTES de adicioná-la
            if hasattr(self, name):
                self.reward_functions.append(getattr(self, name))
            else:
                print(f"ALERTA: Função de recompensa '{name}' definida em cfg mas NÃO IMPLEMENTADA na classe.")
        print(f"Funções de recompensa carregadas: {self.reward_names}")


    def step_train(self, actions):
        # Lógica de step() do T1
        
        # 1. Aplicar ações e calcular alvos
        self.actions[:] = torch.clip(actions, -self.action_clip, self.action_clip)
        self.dof_pos_targets[:] = self.default_dof_pos + self.action_scale * self.actions

        # 2. Simular (T1 usa 'decimation' - simulação multi-passo)
        self.torques.zero_()
        decimation = self.cfg["control"].get("decimation", 1) # Pega do cfg
        for i in range(decimation):
            # (A lógica de torque do T1 é mais complexa, mas usamos a nossa por enquanto)
            torques = self.dof_stiffness * (self.dof_pos_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            
            # (A lógica de clip do T1 usa 'torque_limits', que não carregamos. Usando 100.0)
            self.torques[:] = torch.clip(torques, -100.0, 100.0) 
            
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            
            self.gym.refresh_dof_state_tensor(self.sim)
            # (T1 também refresca 'dof_force_tensor')
        
        # 3. Refresh dos buffers principais (pós-simulação)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # 4. Atualizar estados computados (T1)
        self.episode_length_buf += 1
        # (Atualiza estados como base_lin_vel, base_ang_vel, projected_gravity...)
        self.base_lin_vel[:] = quat_rotate_inverse(self.robot_root_states[:, 3:7], self.robot_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.robot_root_states[:, 3:7], self.robot_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.robot_root_states[:, 3:7], self.gravity_vec)
        
        # 5. Chamar o ciclo de RL
        self._check_termination()
        self._compute_reward()
        self._compute_observations()

        # 6. Lidar com resets
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids) 
        
        
        # 8. Retornar buffers para o algoritmo
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _init_simulation_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.is_training_mode:
            net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_states_tensor = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.num_actors = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        
        if not self.is_training_mode:
            # Modo Player: Múltiplos atores (Robô=0, Bola=1)
            self.robot_root_states = self.root_states_tensor.view(self.num_envs, self.num_actors, 13)[:, 0, :]
            self.ball_root_states = self.root_states_tensor.view(self.num_envs, self.num_actors, 13)[:, 1, :]
        else:
            # Modo Trainer: Apenas 1 ator (Robô)
            self.robot_root_states = self.root_states_tensor # View 2D
            self.ball_root_states = None 
        
        if self.head_link_index != -1:
            self.head_states = self.rigid_body_states[:, self.head_link_index, :]
        else:
            self.head_states = torch.empty((self.num_envs, 13), device=self.device) 

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)

        self.dof_pos_targets = torch.zeros_like(self.dof_pos)
        self.gravity_vec = to_torch([0, 0, -1.0], device=self.device).repeat((self.num_envs, 1))
        self.commands = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32) 
        self.gait_frequency = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.gait_process = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.default_dof_pos = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        if "init_state" not in self.cfg or "default_joint_angles" not in self.cfg["init_state"]:
            print("ALERTA: 'default_joint_angles' não encontrado... Usando 0.0.")
            default_angles_cfg = {"default": 0.0}
        else:
            default_angles_cfg = self.cfg["init_state"]["default_joint_angles"]

        for i in range(self.num_dofs):
            found = False
            for name in default_angles_cfg.keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = default_angles_cfg[name]
                    found = True
                    break 
            if not found:
                default_val = default_angles_cfg.get("default", 0.0)
                if "default" not in default_angles_cfg:
                     print(f"ALERTA: Posição default para {self.dof_names[i]} não encontrada. Usando 0.0.")
                self.default_dof_pos[:, i] = default_val
        print("Pose estática (default_dof_pos) carregada do CFG.")


    def reset(self, env_ids=None):
        if self.is_training_mode:
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            if len(env_ids) == 0:
                return

            self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
            self.dof_vel[env_ids] = 0.0
            
            self.robot_root_states[env_ids] = self.base_init_state
            self.robot_root_states[env_ids, :2] += self.env_origins[env_ids, :2]
            
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
            )
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, gymtorch.unwrap_tensor(self.root_states_tensor), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
            )
            
            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0 
            
        else:
            pass

    def _compute_reward(self):
        pass

    def _compute_observations(self):
        pass

    def _check_termination(self):
        pass