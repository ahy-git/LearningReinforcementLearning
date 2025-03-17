# #seguindo o treinamento
# import warnings
# warnings.filterwarnings('ignore')
# import ale_py
# from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_atari_env
# from stable_baselines3.common.vec_env import VecFrameStack
# import torch.optim as optim


# # Criar ambiente na outra máquina
# env = make_atari_env("SpaceInvaders-v4", n_envs=8, seed=99)  # Alteramos seed para diversificação
# env = VecFrameStack(env, n_stack=4)

# # Criar modelo normalmente
# model = DQN(
#     "CnnPolicy",
#     env,
#     buffer_size=500000,
#     learning_rate=5e-4,
#     batch_size=64,
#     train_freq=2,
#     gradient_steps=2,
#     verbose=1,
#     tensorboard_log="./dqn_space_invaders/"
# )

# # Substituir o otimizador manualmente
# model.policy.optimizer = optim.RMSprop(model.policy.parameters(), lr=5e-4, eps=1e-5)

# # Continuar o treinamento
# model.learn(total_timesteps=500000000)

# # Salvar o modelo treinado
# model.save("dqn_space_invaders_rmsprop.zip")




# import os
# from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_atari_env
# from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.callbacks import CheckpointCallback
# import ale_py

# # Criar ambiente
# env = make_atari_env("SpaceInvaders-v4", n_envs=4, seed=42)
# env = VecFrameStack(env, n_stack=4)

# # Criar pasta para salvar checkpoints
# checkpoint_dir = "./dqn_checkpoints/"
# os.makedirs(checkpoint_dir, exist_ok=True)

# # Criar callback para salvar a cada 100.000 timesteps
# checkpoint_callback = CheckpointCallback(
#     save_freq=100000,  # Salvar a cada 100.000 timesteps
#     save_path=checkpoint_dir,  
#     name_prefix="dqn_space_invaders"
# )

# # Tentar carregar um modelo salvo
# model_path = os.path.join(checkpoint_dir, "dqn_space_invaders_latest.zip")

# if os.path.exists(model_path):
#     print("✅ Modelo salvo encontrado! Continuando treinamento...")
#     model = DQN.load(model_path, env=env, tensorboard_log="./dqn_space_invaders/")
# else:
#     print("🚀 Nenhum modelo encontrado, iniciando novo treinamento...")
#     model = DQN(
#         "CnnPolicy",
#         env,
#         buffer_size=500000,
#         learning_rate=5e-4,
#         batch_size=64,
#         train_freq=4,
#         gradient_steps=2,
#         verbose=1,
#         tensorboard_log="./dqn_space_invaders/"
#     )

# # Treinar o modelo e salvar periodicamente
# model.learn(total_timesteps=2000000, callback=checkpoint_callback)

# # Salvar última versão do modelo ao final do treinamento
# model.save(os.path.join(checkpoint_dir, "dqn_space_invaders_latest.zip"))
# print("✅ Treinamento concluído e modelo salvo!")

import os
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

# Criar ambiente com 15 processos paralelos
env = make_atari_env("SpaceInvaders-v4", n_envs=8, seed=42)
env = VecFrameStack(env, n_stack=4)

# Criar pasta para salvar checkpoints
checkpoint_dir = "./dqn_checkpoints_3/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Criar callback para salvar a cada 500.000 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,  # Salvar com mais frequência para evitar perdas
    save_path=checkpoint_dir,  
    name_prefix="dqn_space_invaders_train3"
)

# Nome fixo para os logs do TensorBoard
log_dir = "./dqn_space_invaders/train3"  # 🌟 Define um nome fixo para manter o log contínuo
os.makedirs(log_dir, exist_ok=True)  # Criar diretório se não existir

# Caminho do modelo salvo anteriormente
model_path = os.path.join(checkpoint_dir, "dqn_space_invaders_train3_latest.zip")

# Se houver um modelo salvo, carregar para continuar treinamento
if os.path.exists(model_path):
    print("✅ Modelo salvo encontrado! Continuando treinamento...")
    model = DQN.load(model_path, env=env, tensorboard_log=log_dir)
else:
    print("🚀 Nenhum modelo encontrado, iniciando novo treinamento...")
    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=500000,  # Replay buffer grande
        learning_rate=0.00025,  # Taxa de aprendizado correta
        batch_size=32,  # Minibatch para estabilidade
        gamma=0.99,  # Fator de desconto
        exploration_fraction=0.1,  # Epsilon decaindo de 1 → 0.1
        exploration_final_eps=0.1,  # Epsilon final em 0.1
        target_update_interval=10000,  # Atualização da rede-alvo
        train_freq=4,  # Atualizar a cada 4 passos
        gradient_steps=1,  # Um update por passo
        verbose=1,
        tensorboard_log=log_dir
    )
    model.save(os.path.join(checkpoint_dir, "dqn_space_invaders_train3_latest.zip"))
   

# Treinar o modelo e salvar periodicamente
model.learn(total_timesteps=6000000, callback=checkpoint_callback)

# Salvar última versão do modelo ao final do treinamento
model.save(os.path.join(checkpoint_dir, "dqn_space_invaders_train3_latest.zip"))
print("✅ Treinamento concluído e modelo salvo!")
