import os
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

# Criar ambiente com 21 processos paralelos para maior eficiência
env = make_atari_env("SpaceInvaders-v4", n_envs=8, seed=42)
env = VecFrameStack(env, n_stack=4)

# Criar pasta para salvar checkpoints
checkpoint_dir = "./dqn_checkpoints_5/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Criar callback para salvar a cada 500.000 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=128_000,  # Salvar com mais frequência para evitar perdas
    save_path=checkpoint_dir,
    name_prefix="dqn_space_invaders_train5"
)

# Caminho do modelo salvo anteriormente
model_path = os.path.join(
    checkpoint_dir, "dqn_space_invaders_train5_latest.zip")

# Nome fixo para os logs do TensorBoard
# 🌟 Define um nome fixo para manter o log contínuo
log_dir = "./dqn_space_invaders/train5/"
os.makedirs(log_dir, exist_ok=True)  # Criar diretório se não existir

# Se houver um modelo salvo, carregar para continuar treinamento
if os.path.exists(model_path):
    print("✅ Modelo salvo encontrado! Continuando treinamento...")
    model = DQN.load(model_path, env=env, tensorboard_log=log_dir)
else:
    print("🚀 Nenhum modelo encontrado, iniciando novo treinamento...")
    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=1000_000,  # Replay buffer grande
        learning_rate=0.00025,  # Taxa de aprendizado correta
        batch_size=64,  # Aumentado para melhor aprendizado
        gamma=0.99,  # Fator de desconto
        exploration_fraction=0.3,  # Epsilon decaindo de 1 → 0.1
        exploration_final_eps=0.15,  # Mantém um pouco mais de exploração contínua
        target_update_interval=5_000,  # Atualização da rede-alvo mais frequente
        train_freq=4,  # Atualizar a cada 4 passos
        gradient_steps=2,  # Mais estabilidade no aprendizado
        # policy_kwargs={"dueling": True},  # **Double DQN ativado**
        # **Prioritized Experience Replay (PER) pode ser ativado aqui**
        replay_buffer_class=None,
        verbose=1,
        tensorboard_log=log_dir
    )
    model.save(os.path.join(checkpoint_dir,
               "dqn_space_invaders_train5_latest.zip"))

# Treinar o modelo e salvar periodicamente
model.learn(total_timesteps=48_000_000, callback=checkpoint_callback)

# Salvar última versão do modelo ao final do treinamento
model.save(os.path.join(checkpoint_dir, "dqn_space_invaders_train5_latest.zip"))
print("✅ Treinamento concluído e modelo salvo!")
