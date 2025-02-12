import gym
import pygame

# Inicializar o ambiente CartPole
env = gym.make("CartPole-v1", render_mode="human")

# Inicializar pygame para capturar teclado
pygame.init()

# Definir tamanho da janela (JANELA SEPARADA PARA HUD)
HUD_HEIGHT = 100  # Altura do HUD
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 300
full_screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT + HUD_HEIGHT))  # Criando espaço para o HUD
pygame.display.set_caption("Controle o CartPole")

# Definir cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Variável global para armazenar a última tecla pressionada
last_pressed_key = None

# Função para capturar entrada do usuário


def get_action(last_action, waiting_for_start=False):
    global last_pressed_key

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            env.close()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                last_pressed_key = "Esq"
                return 0  # Esquerda
            elif event.key == pygame.K_RIGHT:
                last_pressed_key = "Dir"
                return 1  # Direita
            elif waiting_for_start:  # Se estamos esperando para começar, qualquer tecla inicia
                last_pressed_key = "Dir"
                return 1

    return None if waiting_for_start else last_action

# Função para exibir tela inicial antes de começar


def wait_for_start_screen():
    font = pygame.font.Font(None, 36)
    text = font.render("Pressione qualquer tecla para começar", True, BLACK)

    full_screen.fill(WHITE)
    full_screen.blit(text, (50, 130))
    pygame.display.flip()

    while True:
        action = get_action(None, waiting_for_start=True)
        if action is not None:
            return action

# Função para desenhar o HUD (separado do jogo)


def draw_hud(score):
    font = pygame.font.Font(None, 40)

    # Fundo do HUD em branco
    full_screen.fill(WHITE, (0, SCREEN_HEIGHT, SCREEN_WIDTH, HUD_HEIGHT))

    # Texto da pontuação em preto
    score_text = font.render(f"Pontuação: {int(score)}", True, BLACK)
    full_screen.blit(score_text, (20, SCREEN_HEIGHT + 20))

    # Texto da tecla pressionada em preto
    if last_pressed_key:
        key_text = font.render(f"Tecla: {last_pressed_key}", True, BLACK)
        full_screen.blit(key_text, (250, SCREEN_HEIGHT + 20))

    # Atualiza apenas o HUD
    pygame.display.update((0, SCREEN_HEIGHT, SCREEN_WIDTH, HUD_HEIGHT))

# Função para exibir tela de Game Over com pontuação


def game_over_screen(score):
    font = pygame.font.Font(None, 36)
    text1 = font.render("Game Over!", True, BLACK)
    text2 = font.render(f"Pontuação: {score}", True, BLACK)
    text3 = font.render("Pressione R para Retry ou Q para Sair", True, BLACK)

    full_screen.fill(WHITE)
    full_screen.blit(text1, (140, 80))
    full_screen.blit(text2, (140, 130))
    full_screen.blit(text3, (20, 180))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                env.close()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True
                elif event.key == pygame.K_q:
                    pygame.quit()
                    env.close()
                    exit()
        pygame.time.delay(100)


# **Exibir tela de espera e aguardar primeira tecla**
first_action = wait_for_start_screen()

# Loop principal do jogo
while True:
    state = env.reset()[0]
    done = False
    score = 0
    action = first_action

    while not done:
        env.render()
        action = get_action(action)
        draw_hud(score)

        next_state, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = bool(terminated or truncated)

    # Exibir tela de Game Over com pontuação e perguntar se o jogador quer tentar novamente
    if not game_over_screen(int(score)):
        break
