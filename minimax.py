import random
import numpy as np
import pyspiel
import time

def rollout_evaluation(state, maximizing_player, n_rollouts=20):
    """
    Ejecuta n_rollouts partidas aleatorias desde `state` y devuelve
    el promedio de la utilidad para `maximizing_player`.
    """
    total = 0.0
    for _ in range(n_rollouts):
        sim_state = state.clone()
        # jugar aleatoriamente hasta terminal
        while not sim_state.is_terminal():
            action = random.choice(sim_state.legal_actions(sim_state.current_player()))
            sim_state.apply_action(action)
        returns = sim_state.returns()
        total += returns[maximizing_player]
    return total / float(n_rollouts)

# En Conecta 4, la columna central suele ser la mejor jugada inicial y también una de las más fuertes durante la partida, asi le ayudamos a la busqueda
def action_center_priority(action):
    center = 3  # en tablero de 7 columnas, columna central es 3
    return -abs(center - action)  # más cerca del centro -> mayor prioridad

def alpha_beta(state, depth, alpha, beta, maximizing_player, rollout_at_leaf=30):
    """
    Minimax con poda alfa-beta:
      - depth: profundidad restante
      - alpha, beta: parámetros de poda
      - maximizing_player: índice del jugador cuya utilidad maximizamos
      - rollout_at_leaf: número de rollouts si depth == 0 (evaluación heurística)
    Devuelve (valor_est, mejor_accion) donde mejor_accion es None para nodos internos
    si solo queremos el valor.
    """
    # Caso terminal: devolver utilidad real
    if state.is_terminal():
        returns = state.returns()
        return returns[maximizing_player], None

    # Caso profundidad límite: evaluación heurística por rollouts
    if depth == 0:
        value = rollout_evaluation(state, maximizing_player, n_rollouts=rollout_at_leaf)
        return value, None

    current = state.current_player()
    legal = state.legal_actions(current)

    # Si no hay acciones (raro fuera de terminal), devolver evaluación por rollouts
    if not legal:
        value = rollout_evaluation(state, maximizing_player, n_rollouts=rollout_at_leaf)
        return value, None

    best_action = None

    # Si es el turno del jugador maximizador
    if current == maximizing_player:
        value = -float('inf')
        # ordenar movimientos por heurística simple: priorizar el centro (opcional)
        ordered_actions = sorted(legal, key=lambda a: action_center_priority(a), reverse=True)
        for action in ordered_actions:
            child = state.clone()
            child.apply_action(action)

            # si la acción termina el juego inmediatamente, podemos leer el resultado
            if child.is_terminal():
                child_val = child.returns()[maximizing_player]
            else:
                child_val, _ = alpha_beta(child, depth - 1, alpha, beta, maximizing_player, rollout_at_leaf)

            if child_val > value:
                value = child_val
                best_action = action

            alpha = max(alpha, value)
            if alpha >= beta:
                # poda
                break
        return value, best_action

    # Turno del jugador minimizador (el rival)
    else:
        value = float('inf')
        ordered_actions = sorted(legal, key=lambda a: action_center_priority(a), reverse=False)
        for action in ordered_actions:
            child = state.clone()
            child.apply_action(action)

            if child.is_terminal():
                child_val = child.returns()[maximizing_player]
            else:
                child_val, _ = alpha_beta(child, depth - 1, alpha, beta, maximizing_player, rollout_at_leaf)

            if child_val < value:
                value = child_val
                best_action = action

            beta = min(beta, value)
            if alpha >= beta:
                break
        return value, best_action




if __name__ == "__main__":
    game = pyspiel.load_game("connect_four")

    search_depth=6
    rollout_at_leaf=8
    max_print_depth=5

    state = game.new_initial_state()

    turn = 0
    print("=== GAME START ===")
    print(state) 

    while not state.is_terminal():
        print("\n--- TURN", turn, "player", state.current_player(), "---")
        #start = time.time()
        value, best_action = alpha_beta(
            state,
            depth=search_depth,
            alpha=-float('inf'),
            beta=float('inf'),
            maximizing_player=state.current_player(),
            rollout_at_leaf=rollout_at_leaf,
        )
        #end = time.time()

        print(f"\nBest action at root: {best_action} -> {state.action_to_string(state.current_player(), best_action)}")
        print(f"Estimated value (for player {state.current_player()}): {value:.4f}")
        #print(f"Nodes visited: {nodes}, time: {end-start:.2f}s")

        # Aplicar la mejor acción y mostrar estado
        state.apply_action(best_action)
        print("\nEstado después de aplicar mejor acción:")
        print(state)

        turn += 1

    # Juego terminado: mostrar recompensas
    returns = state.returns()
    for pid in range(game.num_players()):
        print(f"Utility for player {pid} is {returns[pid]}")



