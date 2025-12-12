import random
import pyspiel
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt


# Ambiente de juego
game = pyspiel.load_game("connect_four")

# Parametros del juego
num_players = game.num_players()
num_rows = game.get_parameters()["rows"]
num_cols = game.get_parameters()["columns"]

# Q-table, se usara un diccionario para asociar estado con accion 
q_table = defaultdict(lambda: defaultdict(float))

# Hiperparámetros para Q-learning
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99995
epsilon_min = 0.01

# Contadores de resultados
agent_wins = 0
agent_losses = 0
agent_draws = 0
#Almacenamiento para segmentos de 1000 juegos
recent_results = deque(maxlen=1000)
recent_wins = 0
recent_losses = 0
recent_draws = 0

#Guardado de estadisticas para plotting
episode_stats = []


# Función para obtener la representación del estado como string, usado para el mapeo en la Q-table
def state_to_string(state):
    # Si el juego va a acabar, se retorna un string especial 
    if state.is_terminal():
        return "terminal"
    # String para el mapeo de estado -> accion
    return str(state.observation_string(state.current_player()))

def select_action_epsilon_greedy(state, q_table, epsilon):
    legal_actions = state.legal_actions()

    #Situacion donde no tiene acciones disponibles 
    if not legal_actions:
        return None

    # Transformacion del estado actual a la llave usada en el mapeo
    state_key = state_to_string(state)

    # Epsilon es el valor que indica la preferencia entre exploracion y explotacion, se reduce con el tiempo
    # Exploracion 
    if random.random() < epsilon:
        return random.choice(legal_actions)
    # Explotacion, hace referencia a los valores guardados en la Q-table
    else:
        q_values = {action: q_table[state_key].get(action, 0.0) for action in legal_actions}

        # Busca la mejor o mejores acciones para el estado actual dependiendo de los valores q obtenidos al usar la llave de estado actual
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]

        return random.choice(best_actions)

#Funcion para actualizar los valores guardados en la Q-table usando: Q(s, a) ← Q(s, a) + α * [R + γ * max(Q(s', a')) - Q(s, a)]
def update_q_value(state, action, recompensa, next_state, q_table, alpha, gamma):
    state_key = state_to_string(state)
    next_state_key = state_to_string(next_state) if next_state else "terminal"

    # Q-value actual
    q_actual = q_table[state_key].get(action, 0.0)

    # Max Q-value del siguiente estado
    if next_state and not next_state.is_terminal():
        next_legal_actions = next_state.legal_actions()
        if next_legal_actions:
            next_q = max(q_table[next_state_key].get(a, 0.0) for a in next_legal_actions)
        else:
            next_q = 0.0
    else:
        next_q = 0.0

    # Ecuacion para calcular el valor de Q
    nuevo_q = q_actual + alpha * (recompensa + gamma * next_q - q_actual)
    q_table[state_key][action] = nuevo_q

#Funcion para obtener la recompensa de victoria, la funcion del ambiente retorna un valor dependiendo del jugador elegido
#(Aun que solo importara para el jugador que aprende)
# +1 si gana, -1 si pierde, 0 si empata
def get_agent_recompensa(state, agent_player):
    returns = state.returns()
    if len(returns) > agent_player:
        return returns[agent_player]
    return 0.0

#Funcion para guardar los valores del entrenamiento
def update_recent_results(result):
    global recent_wins, recent_losses, recent_draws
    
    recent_results.append(result)
    
    # Se usa el deque para guardar los ultimos 1000 resultados
    recent_wins = recent_results.count(1)
    recent_losses = recent_results.count(-1)
    recent_draws = recent_results.count(0)

# Función principal para el entrenamiento, usa los datos para calcular Q y guarda los avances
def train_q_learning(num_episodes):
    global epsilon, agent_wins, agent_losses, agent_draws
    global recent_wins, recent_losses, recent_draws

    # El jugador agente sera el primero en jugar, el primero siempre tiene una ventaja sobre el segundo
    # Uno de los objetivos es encontrar la solucion optima investigada por estudios sobre el juego
    # (El jugador 1 siempre puede ganar o empatar si empieza en el espacio del medio y juega perfectamente)
    for episode in range(num_episodes):
        state = game.new_initial_state()
        agent_player = 0

        # Guardado de variables del episodio
        episode_history = []

        # Jugar un episodio completo, hasta que no puedan jugar o uno haya ganado
        while not state.is_terminal():
            current_player = state.current_player()

            #Turno del agente
            if current_player == agent_player:
                
                # Selecciona una accion usando e greedy, dentro de la seleccion se actualizan los valores Q y la tabla
                action = select_action_epsilon_greedy(state, q_table, epsilon)

                if action is None:
                    break

                prev_state = state.clone()
                state.apply_action(action)

                if not state.is_terminal():
                    episode_history.append((prev_state, action, state))
                else:
                    episode_history.append((prev_state, action, state))
            else:
                # Turno del oponente, realiza una accion aleatoria
                legal_actions = state.legal_actions()
                if legal_actions:
                    action = random.choice(legal_actions)
                    state.apply_action(action)

        # Calcular recompensa al final del episodio
        recompensa = get_agent_recompensa(state, agent_player)

        # Actualizar Q-values para cada transición
        for prev_state, action, next_state in episode_history:
            update_q_value(prev_state, action, recompensa, next_state, q_table, alpha, gamma)

        # Dependiendo de la victoria/perdida/empate, añade los valores a los resultados
        if recompensa == 1.0:
            agent_wins += 1
            update_recent_results(1)
        elif recompensa == -1.0:
            agent_losses += 1
            update_recent_results(-1)
        else:
            agent_draws += 1
            update_recent_results(0)

        # Epsilon es el valor de decision de exploracion/explotacion, 
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # progreso cada 1000 juegos
        if (episode + 1) % 1000 == 0:
            # Calcular winrate de los últimos 1000 juegos
            total_recent = recent_wins + recent_losses + recent_draws
            if total_recent > 0:
                recent_win_rate = recent_wins / total_recent * 100
                recent_loss_rate = recent_losses / total_recent * 100
                recent_draw_rate = recent_draws / total_recent * 100
            else:
                recent_win_rate = recent_loss_rate = recent_draw_rate = 0
            
            # Valores acumulados
            total_global = agent_wins + agent_losses + agent_draws
            if total_global > 0:
                global_win_rate = agent_wins / total_global * 100
            else:
                global_win_rate = 0

            episode_stats.append({
                'episode': episode + 1,
                'recent_win_rate': recent_win_rate,
            })
            
            print(f"Episodio {episode + 1}: Epsilon={epsilon:.4f}")
            print(f"  Segmento de 1000 juegos - Victorias: {recent_wins} ({recent_win_rate:.1f}%), "
                  f"Derrotas: {recent_losses} ({recent_loss_rate:.1f}%), "
                  f"Empates: {recent_draws} ({recent_draw_rate:.1f}%)")
            print(f"  Acumulado - Victorias: {agent_wins}, Derrotas: {agent_losses}, "
                  f"Empates: {agent_draws}, Tasa victorias: {global_win_rate:.1f}%")
            print(f"  Estados aprendidos: {len(q_table)}")
            print("-" * 80)

    # Tabla de % de victorias contra la cantidad de juegos
    if episode_stats:
        episodes = [s['episode'] for s in episode_stats]
        winrates = [s['recent_win_rate'] for s in episode_stats]
            
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, winrates, 'b-', linewidth=2)
        plt.xlabel('Episodio')
        plt.ylabel('% de victorias')
        plt.title(f'Progreso del Entrenamiento ({num_episodes} juegos)')
        plt.grid(True, alpha=0.3)
            
        # Líneas de referencia para el % de victorias
        plt.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (Malo)')
        plt.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='75% (Bueno)')
        plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% (Excelente)')
            
        plt.legend()
        plt.tight_layout()
        plt.show()


# Función para evaluar el agente, toma la Q table calculada y solo realiza explotacion
def evaluate_agent(num_games):
    wins = 0
    losses = 0
    draws = 0
    estados_primer_movimiento = []
    estados_ultimo_movimiento = []

    for game_num in range(num_games):
        state = game.new_initial_state()
        agent_player = 0 
        game_states = []

        game_states.append(state.clone())

        while not state.is_terminal():
            current_player = state.current_player()

            if current_player == agent_player:
                action = select_action_epsilon_greedy(state, q_table, 0.0)
            else:
                legal_actions = state.legal_actions()
                action = random.choice(legal_actions) if legal_actions else None

            if action is None:
                break

            state.apply_action(action)

        estados_ultimo_movimiento.append(state.clone())
        returns = state.returns()
        if returns[agent_player] == 1.0:
            wins += 1
        elif returns[agent_player] == -1.0:
            losses += 1
        else:
            draws += 1

    print(f"\nEvaluación ({num_games} juegos):")
    print(f"Victorias: {wins} ({wins/num_games*100:.1f}%)")
    print(f"Derrotas: {losses} ({losses/num_games*100:.1f}%)")
    print(f"Empates: {draws} ({draws/num_games*100:.1f}%)")
    return estados_primer_movimiento, estados_ultimo_movimiento, wins, losses, draws



print("Entrenamiento por Q learning")
train_q_learning(num_episodes=500000)

## Evaluar el agente entrenado en 100 juegos contra un rival aleatorio
#evaluate_agent(num_games=100)

# Realizar un juego de ejemplo
print("\nJuego de ejemplo:")
state = game.new_initial_state()
step = 0

while not state.is_terminal():
    step += 1
    print(f"\nPaso {step}")
    print(f"Tablero:\n{state}")

    legal_actions = state.legal_actions()
    print(f"Acciones legales: {legal_actions}")

    #Agente
    if state.current_player() == 0:
        action = select_action_epsilon_greedy(state, q_table, 0.0)
        print(f"Agente juega columna: {action}")
    # Oponente aleatorio
    else:  
        action = random.choice(legal_actions) if legal_actions else None
        print(f"Oponente juega columna: {action}")

    if action is not None:
        state.apply_action(action)

print(f"\nJuego terminado! Resultado: {state.returns()}")