import pyspiel
import random
import numpy as np
import time
import os
import pickle
from collections import defaultdict
from open_spiel.python.algorithms import mcts
from SARSA import state_to_key

def evaluate_agent_sarsa(Q_table, opponent_type="random", num_games=100, mcts_bot=None):
    
    game = pyspiel.load_game("connect_four")
    # Ya no hay agent_net.eval() porque es un diccionario
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"--- Iniciando Evaluación vs {opponent_type.upper()} ({num_games} partidas) ---")
    
    for i in range(num_games):
        state = game.new_initial_state()
        
        # Alternar quién empieza:
        # Partidas pares: Agente es Player 0
        # Partidas impares: Agente es Player 1
        agent_player_id = i % 2 
        
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            current_player = state.current_player()
            
            if current_player == agent_player_id:
                # --- JUEGA TU AGENTE (SARSA TABULAR) ---
                # Estrategia GREEDY: Buscar la acción con mayor valor en Q
                
                s_key = state_to_key(state, current_player)
                
                # Buscamos la acción legal que tenga el valor Q más alto.
                # Si la acción no existe en la tabla, asume valor 0.0 (o un valor bajo si prefieres)
                action = max(legal_actions, key=lambda a: Q_table.get((s_key, a), 0.0))
                
            else:
                # --- JUEGA EL OPONENTE ---
                if opponent_type == "random":
                    action = random.choice(legal_actions)
                elif opponent_type == "mcts":
                    action = mcts_bot.step(state)
            
            state.apply_action(action)
        
        # Fin del juego
        returns = state.returns()
        agent_reward = returns[agent_player_id]
        
        if agent_reward > 0:
            wins += 1
        elif agent_reward < 0:
            losses += 1
        else:
            draws += 1
            
    win_rate = (wins / num_games) * 100
    print(f"Resultados vs {opponent_type}:")
    print(f"Victorias: {wins} | Derrotas: {losses} | Empates: {draws}")
    print(f"Win Rate: {win_rate:.2f}%")
    print("---------------------------------------------------")
    return win_rate

def play_vs_human(Q_table):
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()
    
    print("Tú eres Player 1 (Turnos pares). El Agente es Player 0.")
    
    while not state.is_terminal():
        print("\n" + str(state)) # Imprime el tablero en texto
        legal_actions = state.legal_actions()
        current_player = state.current_player()
        
        if current_player == 0: # Turno del AGENTE
            print("Pensando agente...")
            
            s_key = state_to_key(state, current_player)
            
            # --- Debugging: Ver valores Q ---
            print("Valores Q del agente:")
            for action in legal_actions:
                val = Q_table.get((s_key, action), 0.0)
                print(f"  Col {action}: {val:.4f}")
            # -------------------------------

            # Selección Greedy
            action = max(legal_actions, key=lambda a: Q_table.get((s_key, a), 0.0))
            
            print(f"Agente elige columna: {action}")
            state.apply_action(action)
            
        else: # Turno HUMANO
            while True:
                try:
                    action_input = input(f"Elige columna {legal_actions}: ")
                    action = int(action_input)
                    if action in legal_actions:
                        break
                    print("Movimiento inválido (columna llena o fuera de rango).")
                except ValueError:
                    print("Por favor ingresa un número válido.")
            state.apply_action(action)
            
    print("\n" + str(state))
    print("Juego terminado. Returns:", state.returns())
    if state.returns()[0] > 0:
        print("¡El Agente Gana!")
    elif state.returns()[0] < 0:
        print("¡Tú Ganas!")
    else:
        print("Empate.")


if __name__ == "__main__":    
    
    # Parámetros de entrenamiento
    NUM_EPISODES = 10000
    EVAL_GAMES = 100
    # -----------------------------------------------


    filename = "q1_table_sarsa.pkl"
    
    # 1. CARGA DE DATOS
    if os.path.exists(filename):
        print(f"Cargando Q-table existente desde {filename}...")
        with open(filename, "rb") as f:
            loaded_data = pickle.load(f)
        # Convertimos de dict normal a defaultdict para evitar errores de clave
        Q = defaultdict(float, loaded_data)
    else:
        print("No se encontró archivo guardado. Se iniciará con una tabla Q vacía.")
        Q = defaultdict(float)        

    # 4. evaluamos contra un random 
    evaluate_agent_sarsa(Q, opponent_type="random", num_games=EVAL_GAMES)
    evaluate_agent_sarsa(Q, opponent_type="mcts", num_games=EVAL_GAMES, mcts_bot=mcts.MCTSBot(pyspiel.load_game("connect_four"), uct_c=2, max_simulations=20, evaluator=mcts.RandomRolloutEvaluator()))

    # 5. jugamos con el bot
    #input("\nPresiona Enter para jugar contra el agente...")
    #play_vs_human(Q)

