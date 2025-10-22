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
        
        self.num_envs = self.cfg["env"]["num_envs"]

        self.enable_camera = self.cfg["env"].get("enable_camera", False)
        if self.enable_camera:
            print("Modo de Operação: [VISUALIZATION]")
            self.camera = VirtualCamera(self.cfg, self.num_envs, self.device)
        else:
            print("Modo de Operação: [HEADLESS]")
            self.camera = None

        self.enable_terminal_control = self.cfg["env"].get("enable_terminal_control", False)
        self.terminal_control = None
        if self.enable_terminal_control:
            print("Controle de velocidade pelo terminal [ATIVADO].")
            self.terminal_control = TerminalVelocityControl(self.device)
            self.terminal_control.start()
        else:
            print("Controle de velocidade pelo terminal [DESATIVADO].")

        policy_file = self.cfg["env"].get("policy_file", None)
        if policy_file is not None:
            try:
                self.stand_policy = torch.load(policy_file, map_location=self.device)
                self.stand_policy.eval() 
                print(f"Política pré-carregada '{policy_file}' carregada com sucesso.")
            except Exception as e:
                print(f"ERRO: Falha ao carregar a política '{policy_file}'.")
                print(e)
                self.stand_policy = None 
        else:
            print("Nenhuma política pré-carregada. O robô aceitará comandos de step(actions).")
            self.stand_policy = None
        
        self.add_goal = self.cfg["env"].get("add_goal", False)
        self.add_ball = self.cfg["env"].get("add_ball", False)
        self.add_walls = self.cfg["env"].get("add_walls", False)
        self.apply_initial_kick = self.cfg["env"].get("apply_initial_kick", False)
        self.kick_velocity = self.cfg["env"].get("kick_velocity", 10.0)  
        self.randomize_init_pos = self.cfg["env"].get("randomize_init_pos", False) 
        self._kick_applied = False
        
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
        
        self.field_dims = gymapi.Vec2(10.0, 6.0)
        self.wall_thickness = 0.1
        
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        
        self.step_count = 0 

        self.test_camera = self.cfg["env"].get("test_camera", False)
        if self.test_camera and (not self.enable_camera or not self.add_ball):
            print("ALERTA: 'test_camera' está True, mas 'enable_camera' ou 'add_ball' está False. O teste não será executado.")
            self.test_camera = False

        self._init_simulation_buffers()

    def _create_envs(self):
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

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
        
        self.head_link_name = self.cfg["camera"].get("link_name", "ERROR_NO_LINK_NAME_IN_CFG") 
        self.head_link_index = self.gym.find_asset_rigid_body_index(robot_asset, self.head_link_name)
        if self.head_link_index == -1: print(f"ALERTA: Link da câmera '{self.head_link_name}' NÃO ENCONTRADO...")
        else: print(f"Link da câmera '{self.head_link_name}' encontrado no índice {self.head_link_index}.")

        if self.add_ball:
            ball_options = gymapi.AssetOptions()
            ball_options.disable_gravity = False
            self.ball_radius = 0.11
            ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_options)
        
        goal_width, goal_height, post_thickness = 1.8, 1.2, 0.1

        if self.add_goal:
            goal_options = gymapi.AssetOptions()
            goal_options.fix_base_link = True
            post_asset = self.gym.create_box(self.sim, post_thickness, post_thickness, goal_height, goal_options)
            crossbar_asset = self.gym.create_box(self.sim, post_thickness, goal_width, post_thickness, goal_options)
        
        self.goal_x_pos = 4.0
        back_wall_x = self.goal_x_pos - self.field_dims.x
        goal_wall_x = self.goal_x_pos + 1.0 

        if self.add_walls:
            wall_options = gymapi.AssetOptions()
            wall_options.fix_base_link = True
            wall_height = goal_height * 1.5 
            side_wall_length = goal_wall_x - back_wall_x
            side_wall_center_x = (goal_wall_x + back_wall_x) / 2
            side_wall_asset = self.gym.create_box(self.sim, side_wall_length, self.wall_thickness, wall_height, wall_options)
            back_wall_asset = self.gym.create_box(self.sim, self.wall_thickness, self.field_dims.y, wall_height, wall_options)

        self.robot_initial_z = self.cfg["init_state"]["pos"][2]
        self.initial_robot_quat = gymapi.Quat(*self.cfg["init_state"]["rot"])
        
        if self.add_goal:
            goal_physical_x = self.goal_x_pos + post_thickness / 2 
            left_post_pose = gymapi.Transform(p=gymapi.Vec3(goal_physical_x, -goal_width / 2, goal_height / 2))
            right_post_pose = gymapi.Transform(p=gymapi.Vec3(goal_physical_x, goal_width / 2, goal_height / 2))
            crossbar_pose = gymapi.Transform(p=gymapi.Vec3(goal_physical_x, 0, goal_height), r=gymapi.Quat(0,0,0,1))
        
        if self.add_walls:
            left_wall_y = -self.field_dims.y / 2
            right_wall_y = self.field_dims.y / 2
            back_wall_pose = gymapi.Transform(p=gymapi.Vec3(back_wall_x, 0, wall_height / 2))
            left_wall_pose = gymapi.Transform(p=gymapi.Vec3(side_wall_center_x, left_wall_y, wall_height / 2))
            right_wall_pose = gymapi.Transform(p=gymapi.Vec3(side_wall_center_x, right_wall_y, wall_height / 2))
            goal_wall_pose = gymapi.Transform(p=gymapi.Vec3(goal_wall_x, 0, wall_height / 2))

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.robot_actor_handles = []

        field_min_x = back_wall_x
        field_max_x = self.goal_x_pos
        field_min_y = -self.field_dims.y / 2
        field_max_y = self.field_dims.y / 2

        print(f"Criando {self.num_envs} ambientes...")
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            self.envs.append(env_handle)
            
            env_origin_vec3 = gymapi.Vec3(self.env_origins[i,0], self.env_origins[i,1], self.env_origins[i,2])

            if self.randomize_init_pos and self.add_walls:
                start_pos_robot = gymapi.Vec3(np.random.uniform(field_min_x, field_max_x), np.random.uniform(field_min_y, field_max_y), self.robot_initial_z)
                start_pos_ball = gymapi.Vec3(np.random.uniform(field_min_x, field_max_x), np.random.uniform(field_min_y, field_max_y), self.ball_radius if self.add_ball else 0.0)
            else:
                if self.randomize_init_pos and not self.add_walls:
                    print("ALERTA: randomize_init_pos=True, mas add_walls=False. Usando posições fixas.")
                center_x = (field_min_x + field_max_x) / 2.0
                center_y = (field_min_y + field_max_y) / 2.0
                start_pos_robot = gymapi.Vec3(center_x, center_y, self.robot_initial_z)
                start_pos_ball = gymapi.Vec3(center_x + 1.0, center_y, self.ball_radius if self.add_ball else 0.0)
            
            robot_pose = gymapi.Transform(p=start_pos_robot + env_origin_vec3, r=self.initial_robot_quat)
            
            if self.add_ball:
                ball_pose = gymapi.Transform(p=start_pos_ball + env_origin_vec3)
            
            if self.add_goal:
                l_post_pose = gymapi.Transform(p=left_post_pose.p + env_origin_vec3, r=left_post_pose.r)
                r_post_pose = gymapi.Transform(p=right_post_pose.p + env_origin_vec3, r=right_post_pose.r)
                c_bar_pose = gymapi.Transform(p=crossbar_pose.p + env_origin_vec3, r=crossbar_pose.r)

            if self.add_walls:
                b_wall_pose = gymapi.Transform(p=back_wall_pose.p + env_origin_vec3, r=back_wall_pose.r)
                l_wall_pose = gymapi.Transform(p=left_wall_pose.p + env_origin_vec3, r=left_wall_pose.r)
                r_wall_pose = gymapi.Transform(p=right_wall_pose.p + env_origin_vec3, r=right_wall_pose.r)
                g_wall_pose = gymapi.Transform(p=goal_wall_pose.p + env_origin_vec3, r=goal_wall_pose.r)
            
            collision_group = 0
            collision_filter = -1

            robot_handle = self.gym.create_actor(env_handle, robot_asset, robot_pose, "robot", collision_group, collision_filter)
            
            if self.add_ball:
                ball_handle = self.gym.create_actor(env_handle, ball_asset, ball_pose, "ball", collision_group, collision_filter)

            if self.add_goal:
                left_post_handle = self.gym.create_actor(env_handle, post_asset, l_post_pose, "left_post", collision_group, collision_filter)
                right_post_handle = self.gym.create_actor(env_handle, post_asset, r_post_pose, "right_post", collision_group, collision_filter)
                crossbar_handle = self.gym.create_actor(env_handle, crossbar_asset, c_bar_pose, "crossbar", collision_group, collision_filter)

            if self.add_walls:
                self.gym.create_actor(env_handle, back_wall_asset, b_wall_pose, "back_wall", collision_group, collision_filter)
                self.gym.create_actor(env_handle, side_wall_asset, l_wall_pose, "left_wall", collision_group, collision_filter)
                self.gym.create_actor(env_handle, side_wall_asset, r_wall_pose, "right_wall", collision_group, collision_filter)
                self.gym.create_actor(env_handle, back_wall_asset, g_wall_pose, "goal_wall", collision_group, collision_filter)
            
            self.robot_actor_handles.append(robot_handle)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, robot_handle)
            self.gym.set_actor_rigid_shape_properties(env_handle, robot_handle, shape_props)
            dof_props = self.gym.get_actor_dof_properties(env_handle, robot_handle)
            dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"].fill(0.0) 
            dof_props["damping"].fill(0.0) 
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            
            if self.add_ball:
                self.ball_handles.append(ball_handle)
                ball_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
                ball_props[0].mass = 0.43
                self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_props)
            
            if self.add_goal:
                self.goal_handles.extend([left_post_handle, right_post_handle, crossbar_handle])

    def _get_env_origins(self):
        # Esta função agora combina as duas lógicas
        
        spacing_x = self.cfg["env"].get("env_spacing", 5.0) 
        spacing_y = spacing_x

        if self.add_walls:
            # Se estamos adicionando paredes, usamos a lógica dinâmica
            # (A mesma lógica de _get_env_origins_player original)
            print("Calculando espaçamento: Modo Dinâmico (baseado nas paredes)")
            self.goal_x_pos = 4.0
            back_wall_x = self.goal_x_pos - self.field_dims.x
            goal_wall_x = self.goal_x_pos + 1.0

            min_x = back_wall_x - self.wall_thickness / 2
            max_x = goal_wall_x + self.wall_thickness / 2
            min_y = -self.field_dims.y / 2 - self.wall_thickness / 2
            max_y = self.field_dims.y / 2 + self.wall_thickness / 2

            env_width = max_x - min_x
            env_depth = max_y - min_y

            spacing_buffer = 2.0 
            spacing_x = env_width + spacing_buffer
            spacing_y = env_depth + spacing_buffer
            
            print(f"Espaçamento dinâmico calculado: X={spacing_x:.2f}, Y={spacing_y:.2f}")

        else:
            # Se não há paredes, usamos o espaçamento estático do YAML
            print(f"Calculando espaçamento: Modo Estático (env_spacing: {spacing_x})")

        # Criar a grade de origens com o espaçamento calculado
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        
        num_cols = int(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        
        env_ids = torch.arange(self.num_envs, device=self.device)
        env_rows = torch.div(env_ids, num_cols, rounding_mode='floor')
        env_cols = torch.remainder(env_ids, num_cols)

        self.env_origins[:, 0] = env_cols * spacing_x
        self.env_origins[:, 1] = env_rows * spacing_y
        self.env_origins[:, 2] = 0.0
            
    def step(self, actions):
        if self.enable_terminal_control:
            self.commands[:] = self.terminal_control.get_commands()
            
        if self.apply_initial_kick and not self._kick_applied and self.add_ball:
            goal_center = torch.tensor([self.goal_x_pos, 0, self.ball_radius], device=self.device)
            ball_positions_relative = self.ball_root_states[:, :3] - self.env_origins
            
            direction_to_goal = goal_center - ball_positions_relative
            direction_to_goal = torch.nn.functional.normalize(direction_to_goal, p=2, dim=1)
            
            kick_vec = direction_to_goal * self.kick_velocity
            
            self.ball_root_states[:, 7:10] = kick_vec
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states_tensor))
            self._kick_applied = True

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
        else:
            policy_actions_clipped = torch.clip(actions, -self.action_clip, self.action_clip)

        self.dof_pos_targets[:] = self.default_dof_pos + self.action_scale * policy_actions_clipped

        torques = self.dof_stiffness * (self.dof_pos_targets - self.dof_pos) - self.dof_damping * self.dof_vel
        
        self.torques[:] = torch.clip(torques, -100.0, 100.0) 
        
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        
        self.gym.simulate(self.sim)

        self.gym.refresh_rigid_body_state_tensor(self.sim) 
        self.gym.refresh_dof_state_tensor(self.sim) 
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = self.root_states_tensor
        
        if self.enable_camera:
            self.render()
        
        return None, None, None, None 

    def _init_simulation_buffers(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.root_states_tensor = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.num_actors = self.gym.get_sim_actor_count(self.sim) // self.num_envs
        root_states_view = self.root_states_tensor.view(self.num_envs, self.num_actors, 13)
        
        self.robot_root_states = root_states_view[:, 0, :]
        
        actor_index = 1
        if self.add_ball:
            self.ball_root_states = root_states_view[:, actor_index, :]
            actor_index += 1
        else:
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
        pass