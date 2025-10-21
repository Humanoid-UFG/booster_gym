# utils.py
import threading
import torch
import sys

class TerminalVelocityControl:
    """
    Inicia uma thread separada para ler comandos de velocidade (x, y, yaw)
    do terminal sem bloquear o loop de simulação principal do Isaac Gym.
    """
    def __init__(self, device):
        self.commands = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        self.lock = threading.Lock()
        self.device = device
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.stopped = False

    def start(self):
        """Inicia a thread de escuta."""
        print("=" * 50)
        print("Controle de Velocidade pelo Terminal Ativado.")
        print("Digite os comandos no formato: x y yaw (ex: 1.0 0.0 0.5)")
        print("Pressione 'Enter' para enviar.")
        print("Digite 'stop' ou 'q' para encerrar.")
        print("=" * 50)
        self.thread.start()

    def stop(self):
        """Sinaliza para a thread parar."""
        self.stopped = True

    def run(self):
        """O loop principal da thread que lê a entrada do usuário."""
        while not self.stopped:
            try:
                cmd_str = input("Comando (x y yaw) ou 'q'/'stop': ")
                
                cmd_str_lower = cmd_str.lower()
                if cmd_str_lower == 'stop' or cmd_str_lower == 'q':
                    self.stopped = True
                    print("Encerrando thread de comando...")
                    break
                    
                parts = cmd_str.split()
                if len(parts) == 3:
                    x = float(parts[0])
                    y = float(parts[1])
                    yaw = float(parts[2])
                    
                    with self.lock:
                        self.commands[0] = x
                        self.commands[1] = y
                        self.commands[2] = yaw
                    print(f"--> Comando atualizado: [x={x}, y={y}, yaw={yaw}]")
                else:
                    print("Formato inválido. Use: x y yaw (três números separados por espaço)")

            except ValueError:
                print("Entrada inválida. Por favor, digite números.")
            except (EOFError, KeyboardInterrupt):
                self.stopped = True
                print("\nEncerrando thread de comando.")
                break
        
        print("Thread de controle do terminal finalizada.")

    def get_commands(self):
        """Retorna o tensor de comando atual de forma segura."""
        with self.lock:
            return self.commands