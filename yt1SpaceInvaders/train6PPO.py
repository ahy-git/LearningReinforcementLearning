import os
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
import datetime

# Criar ambiente com X processos paralelos para maior eficiência
n_env = 64
env = make_atari_env("SpaceInvaders-v4", n_envs=n_env, seed=42)
env = VecMonitor(env)
env = VecFrameStack(env, n_stack=4)

# Criar pasta para salvar checkpoints
checkpoint_dir = "./dqn_checkpoints_6/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Criar um prefixo dinâmico com timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
name_prefix_timestamp = f"dqn_space_invaders_train6_{timestamp}"

save_freq_corrected = max(300_000 // n_env, 1)

# Criar callback para salvar a cada 500.000 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq_corrected,  # Salvar com mais frequência para evitar perdas
    save_path=checkpoint_dir,
    name_prefix=name_prefix_timestamp
)

# Caminho do modelo salvo anteriormente
model_path = os.path.join(
    checkpoint_dir, "dqn_space_invaders_train6_latest.zip")

# Nome fixo para os logs do TensorBoard
# 🌟 Define um nome fixo para manter o log contínuo
log_dir = "./dqn_space_invaders/train6/"
os.makedirs(log_dir, exist_ok=True)  # Criar diretório se não existir

# Se houver um modelo salvo, carregar para continuar treinamento
if os.path.exists(model_path):
    print("✅ Modelo salvo encontrado! Continuando treinamento...")
    model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
else:
    print("🚀 Nenhum modelo encontrado, iniciando novo treinamento...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir
    )
    model.save(os.path.join(checkpoint_dir,
               "dqn_space_invaders_train6_latest.zip"))

# Treinar o modelo e salvar periodicamente
model.learn(total_timesteps=160_000_000, callback=checkpoint_callback)

# Salvar última versão do modelo ao final do treinamento
model.save(os.path.join(checkpoint_dir, "dqn_space_invaders_train6_latest.zip"))
print("✅ Treinamento concluído e modelo salvo!")
