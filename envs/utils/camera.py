import torch
import numpy as np
from isaacgym.torch_utils import quat_rotate_inverse

class VirtualCamera:
    """
    Implementa o "Virtual Camera Model" descrito no paper Dribble Master.
    
    Este modelo calcula matematicamente se um ponto (a bola) está dentro 
    do campo de visão (FOV) de uma câmera com pose e FOV definidos.
    Ele também gerencia o currículo de 2 estágios para o FOV, conforme o paper.
    """

    def __init__(self, cfg, num_envs, device):
        """
        Inicializa o modelo da câmera virtual.
        
        Args:
            cfg (dict): Dicionário de configuração, esperando por cfg["camera"].
            num_envs (int): Número de ambientes paralelos.
            device (str): Dispositivo torch (ex: "cuda:0" ou "cpu").
        """
        cam_cfg = cfg["camera"]
        self.device = device
        self.num_envs = num_envs

        # Carrega o FOV realista (Estágio 2) do robô RealSense D455 [cite: 151, 173]
        # Usamos 87, 58 como padrões se não especificado, baseado no D455
        hfov_real_deg = cam_cfg.get("hfov_real_deg", 87.0) 
        vfov_real_deg = cam_cfg.get("vfov_real_deg", 58.0)
        
        self.hfov_real = torch.tensor(hfov_real_deg * np.pi / 180.0, device=self.device)
        self.vfov_real = torch.tensor(vfov_real_deg * np.pi / 180.0, device=self.device)

        # Calcula o FOV aumentado (Estágio 1), que é 2x o FOV real [cite: 150]
        self.hfov_stage1 = self.hfov_real * 2.0
        self.vfov_stage1 = self.vfov_real * 2.0
        
        # Buffers para os *meios* ângulos de FOV atuais (usados para checagem)
        # Inicializa todos no Estágio 1 por padrão
        self.current_half_hfov = torch.full((num_envs,), self.hfov_stage1 / 2.0, device=self.device, dtype=torch.float32)
        self.current_half_vfov = torch.full((num_envs,), self.vfov_stage1 / 2.0, device=self.device, dtype=torch.float32)

        print(f"VirtualCamera inicializada. FOV Estágio 1 (H,V): {hfov_real_deg * 2:.1f}, {vfov_real_deg * 2:.1f} graus.")
        print(f"VirtualCamera inicializada. FOV Estágio 2 (H,V): {hfov_real_deg:.1f}, {vfov_real_deg:.1f} graus.")

    def set_stage(self, stage_indices, env_ids):
        """
        Define o FOV da câmera para ambientes específicos com base em seu estágio.
        
        Args:
            stage_indices (torch.Tensor): Tensor de índices de estágio (1 ou 2) para os env_ids.
            env_ids (torch.Tensor): Tensor de IDs dos ambientes que serão atualizados.
        """
        if env_ids.numel() == 0:
            return

        # Estágio 1: FOV aumentado [cite: 150]
        stage1_mask = (stage_indices == 1)
        stage1_env_ids = env_ids[stage1_mask]
        if stage1_env_ids.numel() > 0:
            self.current_half_hfov[stage1_env_ids] = self.hfov_stage1 / 2.0
            self.current_half_vfov[stage1_env_ids] = self.vfov_stage1 / 2.0

        # Estágio 2: FOV realista [cite: 151]
        stage2_mask = (stage_indices == 2)
        stage2_env_ids = env_ids[stage2_mask]
        if stage2_env_ids.numel() > 0:
            self.current_half_hfov[stage2_env_ids] = self.hfov_real / 2.0
            self.current_half_vfov[stage2_env_ids] = self.vfov_real / 2.0

    @torch.no_grad()
    def check_ball_in_view(self, camera_pose, ball_pos_global):
        """
        Verifica se a bola está dentro do FOV da câmera virtual para todos os ambientes.
        
        Args:
            camera_pose (torch.Tensor): Pose (pos[3], quat[4]) do link da câmera (cabeça do robô).
                                        Shape: [num_envs, 7].
            ball_pos_global (torch.Tensor): Posição global da bola. 
                                            Shape: [num_envs, 3].
                                            
        Returns:
            is_in_view (torch.Tensor): Tensor Booleano, True se a bola está visível. 
                                       Shape: [num_envs].
            yaw_error (torch.Tensor): Erro de Yaw (radians) para centralizar a bola. 
                                      Shape: [num_envs].
            pitch_error (torch.Tensor): Erro de Pitch (radians) para centralizar a bola. 
                                        Shape: [num_envs].
        """
        # 1. Obter a pose da câmera e a posição da bola
        camera_pos = camera_pose[:, 0:3]
        camera_quat = camera_pose[:, 3:7]

        # 2. Calcular o vetor da câmera para a bola no frame global
        ball_vec_global = ball_pos_global - camera_pos

        # 3. Transformar o vetor para o frame local da câmera
        # (Onde X é para frente, Y é para esquerda, Z é para cima)
        ball_vec_local = quat_rotate_inverse(camera_quat, ball_vec_global)

        # 4. Calcular os ângulos de Yaw (erro horizontal) e Pitch (erro vertical)
        
        # Yaw: ângulo no plano X-Y local. 
        # torch.atan2(y, x)
        # Positivo = bola à esquerda
        yaw_error = torch.atan2(ball_vec_local[:, 1], ball_vec_local[:, 0])
        
        # Pitch: ângulo em relação ao plano X-Y local.
        # torch.atan2(z, dist_horizontal)
        # Positivo = bola acima
        horizontal_dist = torch.sqrt(ball_vec_local[:, 0]**2 + ball_vec_local[:, 1]**2)
        pitch_error = torch.atan2(ball_vec_local[:, 2], horizontal_dist)

        # 5. Verificar se os erros de ângulo estão dentro dos limites de FOV
        in_hfov = (torch.abs(yaw_error) < self.current_half_hfov)
        in_vfov = (torch.abs(pitch_error) < self.current_half_vfov)

        # A bola está visível apenas se estiver dentro de AMBOS os campos de visão
        is_in_view = in_hfov & in_vfov
        
        return is_in_view, yaw_error, pitch_error