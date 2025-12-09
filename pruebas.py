import pyspiel
import numpy as np

def jugar_conecta_4():
    # 1. Cargar Connect Four
    # El string interno para este juego es "connect_four"
    game = pyspiel.load_game("connect_four")
    state = game.new_initial_state()

    print(f"=== Iniciando {game.get_type().long_name} ===")
    print("Tablero vacío:")
    print(state)

    # 2. Bucle principal
    while not state.is_terminal():
        # El estado actual nos dice de quién es el turno
        current_player = state.current_player()
        
        # Obtener columnas válidas (donde no estén llenas)
        legal_actions = state.legal_actions()
        
        # --- AQUÍ IRÍA TU IA ---
        # Por ahora, elegimos una columna al azar
        action = np.random.choice(legal_actions)
        # -----------------------
        
        print(f"\nEl Jugador {current_player} suelta ficha en la columna {action}")
        state.apply_action(action)
        
        # Imprimimos el tablero gráfico (OpenSpiel lo hace con texto)
        print(state)

    # 3. Fin del juego y resultados
    print("\n" + "="*30)
    print("       JUEGO TERMINADO")
    print("="*30)
    
    returns = state.returns()
    print(f"Recompensas finales: {returns}")
    
    # Interpretación del resultado
    if returns[0] > 0:
        print("🏆 ¡Ganó el Jugador 0 (x)!")
    elif returns[1] > 0:
        print("🏆 ¡Ganó el Jugador 1 (o)!")
    else:
        print("🤝 ¡Es un empate!")

if __name__ == "__main__":
    jugar_conecta_4()