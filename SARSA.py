import random
from collections import defaultdict
import numpy as np
import pyspiel
import time
import pickle
import os


def state_to_key(state, player):
    obs = np.array(state.observation_tensor(player), dtype=np.int8)
    return b"p:" + bytes([player]) + b"obs:" + obs.tobytes()

#epsilon-greedy como politica
def epsilon_greedy_action(Q, state_key, legal_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(legal_actions)

    best_val = -np.inf
    best_actions = []
    for a in legal_actions:
        val = Q.get((state_key, a), 0.0)
        if val > best_val:
            best_val = val
            best_actions = [a]
        elif val == best_val:
            best_actions.append(a)

    return random.choice(best_actions)



#ENTRENAMIENTO VS RANDOM
def train_sarsa_vs_random(num_episodes=5000,
                          alpha=0.1,
                          gamma=0.99,
                          epsilon_start=0.3,
                          epsilon_end=0.05,
                          epsilon_decay_episodes=4000,
                          agent_player=0,
                          Q=None):

    game = pyspiel.load_game("connect_four")

    if Q is None:
        Q = defaultdict(float)

    def get_epsilon(ep):
        if ep >= epsilon_decay_episodes:
            return epsilon_end
        frac = ep / float(max(1, epsilon_decay_episodes))
        return epsilon_start * (1 - frac) + epsilon_end * frac

    stats = {"wins": 0, "losses": 0, "draws": 0}

    for ep in range(1, num_episodes + 1):
        epsilon = get_epsilon(ep)
        state = game.new_initial_state()

        # El oponente puede empezar
        while not state.is_terminal() and state.current_player() != agent_player:
            opp_legal = state.legal_actions(state.current_player())
            state.apply_action(random.choice(opp_legal))

        if state.is_terminal():
            r = state.returns()[agent_player]
            if r > 0: stats["wins"] += 1
            elif r < 0: stats["losses"] += 1
            else: stats["draws"] += 1
            continue

        # Primer estado del agente
        s_key = state_to_key(state, agent_player)
        legal = state.legal_actions(agent_player)
        a = epsilon_greedy_action(Q, s_key, legal, epsilon)

        while True:
            # turno del agente
            #print("\n===== Turno del AGENTE (Player {}) =====".format(agent_player))
            #print("Acción elegida por el agente:", a)

            state.apply_action(a)

            #print("Estado después de la acción del agente:")
            #print(state)
            #print("---------------------------------------------")


            if state.is_terminal():
                reward = state.returns()[agent_player]
                old = Q[(s_key, a)]
                Q[(s_key, a)] = old + alpha * (reward - old)

                #print("\n>>> El juego terminó después del turno del AGENTE")
                #print(state)
                #print("Recompensa final:", reward)
                break

            # turno del oponente random
            opp_pid = state.current_player()
            opp_legal = state.legal_actions(opp_pid)
            opp_action = random.choice(opp_legal)

            #print("\n===== Turno del OPONENTE (Player {}) =====".format(opp_pid))
            #print("Acción del oponente:", opp_action)

            state.apply_action(opp_action)

            #print("Estado después de la acción del oponente:")
            #print(state)
            #print("---------------------------------------------")

            if state.is_terminal():
                reward = state.returns()[agent_player]
                old = Q[(s_key, a)]
                Q[(s_key, a)] = old + alpha * (reward - old)
                #print("\n>>> El juego terminó después del turno del OPONENTE")
                #print(state)
                #print("Recompensa final:", reward)
                break

            # SARSA paso intermedio
            s_prime_key = state_to_key(state, agent_player)
            legal_prime = state.legal_actions(agent_player)
            a_prime = epsilon_greedy_action(Q, s_prime_key, legal_prime, epsilon)

            old = Q[(s_key, a)]
            q_next = Q[(s_prime_key, a_prime)]
            Q[(s_key, a)] = old + alpha * (gamma * q_next - old)

            s_key = s_prime_key
            a = a_prime

        if ep % 200 == 0:
            print(f"EP {ep}")

    return Q, stats



#SELF-PLAY (AGENTE ENTRENANDO CONTRA SÍ MISMO)
def train_selfplay_sarsa(num_episodes=5000,
                         alpha=0.1,
                         gamma=0.99,
                         epsilon_start=0.3,
                         epsilon_end=0.05,
                         epsilon_decay_episodes=4000,
                         Q0=None,
                         Q1=None):

    game = pyspiel.load_game("connect_four")

    if Q0 is None:
        Q0 = defaultdict(float)
    if Q1 is None:
        Q1 = defaultdict(float)

    Q = [Q0, Q1]   # Q[0] para player 0, Q[1] para player 1

    def get_epsilon(ep):
        if ep >= epsilon_decay_episodes:
            return epsilon_end
        frac = ep / float(max(1, epsilon_decay_episodes))
        return epsilon_start * (1 - frac) + epsilon_end * frac

    for ep in range(1, num_episodes + 1):
        #print("\n\n==============================")
        #print(f"EPISODIO {ep}")
        #print("==============================\n")

        state = game.new_initial_state()
        epsilon = get_epsilon(ep)

        # Acción inicial
        p = state.current_player()
        s_key = state_to_key(state, p)
        legal = state.legal_actions(p)
        a = epsilon_greedy_action(Q[p], s_key, legal, epsilon)

        #print(f"Jugador inicial: Player {p}")
        #print("Estado inicial del tablero:")
        #print(state)
        #print("---------------------------------------------")
        #print(f"Acción inicial elegida por Player {p}: {a}")
        #print("---------------------------------------------")

        while not state.is_terminal():

            # TURNO DEL JUGADOR p
            #print(f"\n===== Turno del JUGADOR {p} =====")
            #print(f"Acción tomada: {a}")

            state.apply_action(a)

            #print("Estado después de su acción:")
            #print(state)
            #print("---------------------------------------------")

            # Si el juego terminó con la jugada de p
            if state.is_terminal():
                reward = state.returns()
                old = Q[p][(s_key, a)]
                Q[p][(s_key, a)] = old + alpha * (reward[p] - old)

                #print(">>> El juego terminó después del turno del jugador", p)
                #print("Recompensas:", reward)
                #print(f"Actualización final SARSA para Player {p}: Q[{(s_key,a)}] = {Q[p][(s_key,a)]}")
                break

            # TURNO DEL OTRO JUGADOR
            next_p = state.current_player()
            s_prime_key = state_to_key(state, next_p)
            legal_prime = state.legal_actions(next_p)
            a_prime = epsilon_greedy_action(Q[next_p], s_prime_key, legal_prime, epsilon)

            #print(f"\n===== Turno del SIGUIENTE JUGADOR {next_p} =====")
            #print(f"Acción elegida: {a_prime}")

            # UPDATE SARSA
            old = Q[p][(s_key, a)]
            q_next = Q[next_p][(s_prime_key, a_prime)]
            Q[p][(s_key, a)] = old + alpha * (gamma * q_next - old)

            #print("\n[Actualización SARSA]")
            #print(f"Old Q: {old}")
            #print(f"q_next (del otro jugador): {q_next}")
            #print("---------------------------------------------")

            # Avanzar
            s_key = s_prime_key
            a = a_prime
            p = next_p

        if ep % 200 == 0:
            print(f"EP {ep}")

    return Q[0], Q[1]


def evaluate_policy_random(Q, games=500):
    """Evalúa Player 0 vs oponente aleatorio usando Q (greedy)."""
    game = pyspiel.load_game("connect_four")
    results = {"wins": 0, "losses": 0, "draws": 0}

    for _ in range(games):
        state = game.new_initial_state()

        while not state.is_terminal():
            if state.current_player() == 0:
                s_key = state_to_key(state, 0)
                legal = state.legal_actions(0)
                a = max(legal, key=lambda x: Q.get((s_key, x), 0.0))
                state.apply_action(a)
            else:
                opp_legal = state.legal_actions(1)
                state.apply_action(random.choice(opp_legal))

            if state.is_terminal():
                r = state.returns()[0]
                if r > 0: results["wins"] += 1
                elif r < 0: results["losses"] += 1
                else: results["draws"] += 1

    return results

def evaluate_policy_self(Q0, Q1):
    """Evalúa Player 0 usando Q0 (greedy) vs Player 1 usando Q1 (greedy)."""
    game = pyspiel.load_game("connect_four")
    results = {"wins": 0, "losses": 0, "draws": 0}

    Q = [Q0, Q1] 

    state = game.new_initial_state()
    #print(state)

    while not state.is_terminal():
        p = state.current_player()
        s_key = state_to_key(state, p)
        legal = state.legal_actions(p)
        
        # Seleccionar mejor acción entre las legales
        #antes solo tenia max y jugador 0 siempre ganaba y 1 practicamente no daba competencia
        #ahora que jugador 1 minimiza este siempre gana aunque al ver el tablero parece que ahora ambos si ponen mayor competencia
        if p == 0:
            a = max(legal, key=lambda act: Q[p].get((s_key, act), 0.0))
        else:
            a = min(legal, key=lambda act: Q[p].get((s_key, act), 0.0)) 
        state.apply_action(a)
        #print(f"\nPlayer {p} juega acción: {a}")
        #print(state)

        # Resultado final
        if state.is_terminal():
            r = state.returns()[0]   # recompensa desde perspectiva del jugador 0
            if r > 0: results["wins"] += 1
            elif r < 0: results["losses"] += 1
            else: results["draws"] += 1

    return results



if __name__ == "__main__":
    start = time.time()
    games = 1000
    num_episodes = 10000

    #ESCOGER MODO DE ENTRENAMIENTO
    mode = "vs_random"       #"selfplay" o "vs_random"

    if mode == "vs_random":
        # Intentar cargar Q existente
        if os.path.exists("q_table_sarsa.pkl"):
            print("Cargando q_table_sarsa.pkl...")
            with open("q_table_sarsa.pkl", "rb") as f:
                data = pickle.load(f)
            Q = defaultdict(float, data)
        else:
            Q = defaultdict(float)
            Q, stats = train_sarsa_vs_random(num_episodes=num_episodes,Q=Q)
            print("Guardando Q...")
            with open("q_table_sarsa.pkl", "wb") as f:
                pickle.dump(dict(Q), f)

        print("Eval:", evaluate_policy_random(Q, games=games))


    #para el self-play hacen falta dos Q para evitar sobreescritura cuando indeseada
    else:  # SELF-PLAY
        # Intentar cargar Q0 y Q1
        if os.path.exists("q0_tabla_sarsa.pkl") and os.path.exists("q1_tabla_sarsa.pkl"):
            print("Cargando q0_tabla_sarsa.pkl y q1_tabla_sarsa.pkl...")
            Q0 = defaultdict(float, pickle.load(open("q0_tabla_sarsa.pkl", "rb")))
            Q1 = defaultdict(float, pickle.load(open("q1_tabla_sarsa.pkl", "rb")))
            
            results = {"wins": 0, "losses": 0, "draws": 0}

            #Se cambio a asi que se reevalue politica luego de cada juego por que si no todos los juegos eran iguales y eran {"wins": 1000, "losses": 0, "draws": 0}
            #o {"wins": 0, "losses": 1000, "draws": 0} o {"wins": 0, "losses": 0, "draws": 1000}
            for i in range(games):
                res = evaluate_policy_self(Q0, Q1)
                results["wins"] += res["wins"]
                results["losses"] += res["losses"]
                results["draws"] += res["draws"]
                Q0, Q1 = train_selfplay_sarsa(
                    num_episodes= int(num_episodes/1000), # menos episodios por evaluación
                    Q0=Q0,
                    Q1=Q1
                )
            print("Resultados de evaluación tras cargar Q0/Q1:", results)
        else:
            Q0 = defaultdict(float)
            Q1 = defaultdict(float)

            Q0, Q1 = train_selfplay_sarsa(
                num_episodes=num_episodes,
                Q0=Q0,
                Q1=Q1
            )

            print("Guardando Q0/Q1...")
            pickle.dump(dict(Q0), open("q0_tabla_sarsa.pkl", "wb"))
            pickle.dump(dict(Q1), open("q1_tabla_sarsa.pkl", "wb"))

            print("Eval:", evaluate_policy_self(Q0,Q1))

    print("Elapsed:", time.time() - start)


