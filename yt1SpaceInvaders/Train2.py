
import os
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
import ale_py

# Criar ambiente
env = make_atari_env("SpaceInvaders-v4", n_envs=8, seed=42)
env = VecFrameStack(env, n_stack=4)

# Criar pasta para salvar checkpoints
checkpoint_dir = "./dqn_checkpoints_2/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Criar callback para salvar a cada 100.000 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,  # Salvar a cada 100.000 timesteps
    save_path=checkpoint_dir,  
    name_prefix="dqn_space_invaders_train2"
)

# Nome fixo para os logs do TensorBoard
log_dir = "./dqn_space_invaders/train2"  # 🌟 Define um nome fixo para manter o log contínuo
os.makedirs(log_dir, exist_ok=True)  # Criar diretório se não existir

# Tentar carregar um modelo salvo
model_path = os.path.join(checkpoint_dir, "dqn_space_invaders_train2_latest.zip")

if os.path.exists(model_path):
    print("✅ Modelo salvo encontrado! Continuando treinamento...")
    model = DQN.load(model_path, env=env, tensorboard_log=log_dir)
else:
    print("🚀 Nenhum modelo encontrado, iniciando novo treinamento...")
    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=500000,  
        learning_rate=7e-4,  # Aprendizado um pouco mais rápido
        batch_size=128,  # Processar mais amostras por atualização
        exploration_fraction=0.15,  # Mais exploração inicial
        exploration_final_eps=0.02,  
        target_update_interval=4000,  # Atualizar alvo com menor frequência
        train_freq=4,  # Atualizar a cada 4 passos
        gradient_steps=4,  # Mais estabilidade no aprendizado
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.save(os.path.join(checkpoint_dir, "dqn_space_invaders_train2_latest.zip"))
    model = DQN.load(model_path, env=env, tensorboard_log=log_dir)
    


# Treinar o modelo e salvar periodicamente
model.learn(total_timesteps=2000000, callback=checkpoint_callback)

# Salvar última versão do modelo ao final do treinamento
model.save(os.path.join(checkpoint_dir, "dqn_space_invaders_train2_latest.zip"))
print("✅ Treinamento concluído e modelo salvo!")
