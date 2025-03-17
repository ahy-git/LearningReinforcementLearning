# seguindo o treinamento
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3 import DQN
import ale_py
import warnings
warnings.filterwarnings('ignore')

# Processar ambiente para RL
env = make_atari_env("SpaceInvaders-v4", n_envs=7, seed=42)
env = VecFrameStack(env, n_stack=4)

# Criar pasta para salvar checkpoints
checkpoint_dir = "./dqn_checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Criar callback para salvar a cada 100.000 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,  # Salvar a cada 100.000 timesteps
    save_path=checkpoint_dir,
    name_prefix="dqn_space_invaders_train1"
)
# Carregar o modelo treinado e continuar treinamento
model = DQN.load("dqn_space_invaders.zip", env=env,
                 tensorboard_log="./dqn_space_invaders/")

# Continuar o treinamento por mais 500.000 passos
model.learn(total_timesteps=50000000, callback=checkpoint_callback)

# Salvar última versão do modelo ao final do treinamento
model.save(os.path.join(checkpoint_dir, "dqn_space_invaders_train1_latest.zip"))
print("✅ Treinamento concluído e modelo salvo!")
