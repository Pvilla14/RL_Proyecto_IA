import random
from collections import defaultdict
import numpy as np
import pyspiel
import time
import pickle
import os

def state_to_key(state, player):
    obs = np.array(state.observation_tensor(player), dtype=np.int8)
    return b"obs:" + obs.tobytes()

# epsilon-greedy como politica
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

def train_sarsa_connect4(num_episodes=5000,
                         alpha=0.1,
                         gamma=0.99,
                         epsilon_start=0.3,
                         epsilon_end=0.05,
                         epsilon_decay_episodes=4000,
                         agent_player=0,
                         Q=None):

    game = pyspiel.load_game("connect_four")

    # Si no se entrega una Q, crearla
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

        while not state.is_terminal() and state.current_player() != agent_player:
            opp_legal = state.legal_actions(state.current_player())
            state.apply_action(random.choice(opp_legal))

        if state.is_terminal():
            returns = state.returns()
            r = returns[agent_player]
            if r > 0: stats["wins"] += 1
            elif r < 0: stats["losses"] += 1
            else: stats["draws"] += 1
            continue

        s_key = state_to_key(state, agent_player)
        legal = state.legal_actions(agent_player)
        a = epsilon_greedy_action(Q, s_key, legal, epsilon)

        while True:
            #print("\n===== Turno del AGENTE (Player {}) =====".format(agent_player))
            #print("Acción elegida por el agente:", a)

            state.apply_action(a)

            #print("Estado después de la acción del agente:")
            #print(state)
            #print("---------------------------------------------")

            if state.is_terminal():
                returns = state.returns()
                reward = returns[agent_player]
                old = Q[(s_key, a)]
                Q[(s_key, a)] = old + alpha * (reward - old)

                #print("\n>>> El juego terminó después del turno del AGENTE")
                #print(state)
                #print("Recompensa final:", reward)
                break

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
                returns = state.returns()
                reward = returns[agent_player]
                old = Q[(s_key, a)]
                Q[(s_key, a)] = old + alpha * (reward - old)

                #print("\n>>> El juego terminó después del turno del OPONENTE")
                #print(state)
                #print("Recompensa final:", reward)
                break

            s_prime_key = state_to_key(state, agent_player)
            legal_prime = state.legal_actions(agent_player)
            a_prime = epsilon_greedy_action(Q, s_prime_key, legal_prime, epsilon)

            reward = 0.0
            old = Q[(s_key, a)]
            q_next = Q[(s_prime_key, a_prime)]
            Q[(s_key, a)] = old + alpha * (reward + gamma * q_next - old)

            s_key = s_prime_key
            a = a_prime

        if (ep % 100 == 0):
            tot = stats["wins"] + stats["losses"] + stats["draws"]
            print(f"Ep {ep}/{num_episodes} eps={epsilon:.3f} "
                  f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']}")

    return Q, stats


def evaluate_policy(Q, games=200, agent_player=0):
    game = pyspiel.load_game("connect_four")
    results = {"wins": 0, "losses": 0, "draws": 0}
    for _ in range(games):
        state = game.new_initial_state()

        while not state.is_terminal() and state.current_player() != agent_player:
            la = state.legal_actions(state.current_player())
            state.apply_action(random.choice(la))

        while not state.is_terminal():
            s_key = state_to_key(state, agent_player)
            legal = state.legal_actions(agent_player)
            best_a = max(legal, key=lambda a: Q.get((s_key, a), 0.0))
            state.apply_action(best_a)

            if state.is_terminal():
                break

            opp_legal = state.legal_actions(state.current_player())
            state.apply_action(random.choice(opp_legal))

        r = state.returns()[agent_player]
        if r > 0: results["wins"] += 1
        elif r < 0: results["losses"] += 1
        else: results["draws"] += 1

    return results


if __name__ == "__main__":
    start = time.time()

    # CARGA AUTOMÁTICA DE Q TABLE SI EXISTE
    if os.path.exists("./q_table_sarsa.pkl"):
        print("Cargando Q desde q_table_sarsa.pkl...")
        with open("q_table_sarsa.pkl", "rb") as f:
            data = pickle.load(f)
        Q = defaultdict(float, data)
    else:
        print("No existe q_table_sarsa.pkl, iniciando Q desde cero...")
        Q = defaultdict(float)

    # ENTRENAMIENTO
    Q, stats = train_sarsa_connect4(
        num_episodes=10000,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=0.3,
        epsilon_end=0.05,
        epsilon_decay_episodes=1500,
        agent_player=0,
        Q=Q   # <--- carga Q previa
    )

    # GUARDADO DE Q TABLE
    print("Guardando tabla Q en q_table_sarsa.pkl...")
    with open("q_table_sarsa.pkl", "wb") as f:
        pickle.dump(dict(Q), f)

    print("Training finished.")
    res = evaluate_policy(Q, games=1000, agent_player=0)
    print("Eval results:", res)
    print("Elapsed:", time.time() - start)
