import gym
import random
import numpy as np
# matplotlib para hacer las gráficas
import matplotlib as plt
# parta visualizar el muñeco y las acciones
from IPython.display import clear_output
import time 


# Inicializar el entorno no resbaladizo de Frozen Lake
environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")
environment.reset()
environment.render()


# Inicializar la tabla Q con ceros
# Nuestra tabla tiene las siguientes dimensiones:
# (filas x columnas) = (estados x acciones) = (16 x 4)
qtable = np.zeros((16, 4))

# Alternativamente, la biblioteca gym también puede proporcionarnos directamente
# el número de estados y acciones utilizando "env.observation_space.n" y "env.action_space.n"
nb_states = environment.observation_space.n  # = 16
nb_actions = environment.action_space.n      # = 4
qtable = np.zeros((nb_states, nb_actions))

# Veamos cómo se ve
print('Tabla Q =')
print(qtable)


# Elegimos una acción aleatoria
# left -- 0
# down --  1
# right -- 2
# up -- 3
accion = environment.action_space.sample()

# implementamos la acción y movemos el personaje en la dirección
nuevo_estado, recompensa, terminado, info = environment.step(accion)

# Mostrar los resultados (recompensa y mapa)
environment.render()
print(f'Recompensa = {recompensa}')

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

# volvemos a inicializar la tabla q
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hiperparámetros
episodes = 1000        # numero total de episodios
alpha = 0.5            # Tasa de aprendizaje
gamma = 0.9            # Factor de descuento

# lista deresultados
outcomes = []

print('Q-table antes del entrenamiento:')
print(qtable)

# Entrenamiento
for _ in range(episodes):
    state = environment.reset()
    done = False

    # Por defecto, ponemos que el resultado ha salido mal
    outcomes.append("Failure")

    # sefguimos entrenando hasta que llegue a la salida o hasta que se quede en un agujero
    while not done:
        # escogemos la accion con el valor más alto en el estado actuaql
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])

        # si no hay una mejor opcion, es todo 0, escogemos una aleatoria
        else:
          action = environment.action_space.sample()        
    
        # implementamos la acción y movemos al monigote
        new_state, reward, done, info = environment.step(action)

        # actualizamos la tabla q
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
             
        # actualizamos el estado actual
        state = new_state

        # si tenemos una recompensa, entonces ha sido un éxito
        if reward:
          outcomes[-1] = "Success"

print()
print('===========================================')
print('Q-table después del entrenamiento:')
print(qtable)

# graficamos el resultado
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
ax = plt.gca()
plt.bar(range(len(outcomes)), outcomes, width=1.0)
plt.show()

# después de entrenarlo, lo evaluamos en 100 episodios
episodes = 100
nb_success = 0

# Evaluacion
for _ in range(100):
    state = environment.reset()
    done = False
    
    # lo entrenamos hasta que se atasque o hasta que gane
    while not done:
        # cogemos la accion con el valor más alto
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])

        # si no hay valor más alto, cogemos aleatoriamente
        else:
          action = environment.action_space.sample()
          
        # implementamos la accion y movemos el muñeco
        new_state, reward, done, info = environment.step(action)
       
        # actualizamos el estado
        state = new_state

        # cuando tenemos una recompensa, hemos ganadoo
        nb_success += reward

# Vemos la tasa de éxito
print (f"TASA DE ÉXITO = {nb_success/episodes*100}%")


# vamos a visualizar el muñeco moviendose e imprimir la secuencia de acciones

state = environment.reset()
done = False
sequence = []

while not done:
    # cogemos la accion con el valor más alto
    if np.max(qtable[state]) > 0:
      action = np.argmax(qtable[state])

    # si no hay valor más alto, cogemos aleatoriamente
    else:
      action = environment.action_space.sample()
      
    # añadimos la accion a la secuencia de acciones
    sequence.append(action)

    # implementamos la accion y movemos el muñeco
    new_state, reward, done, info = environment.step(action)

    # actualizamos el estado
    state = new_state

    # actualizamos el render
    clear_output(wait=True)
    environment.render()
    time.sleep(1)

print(f"Sequence = {sequence}")

# Re-inicializamos la tabla Q
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hiperparámetros
episodios = 1000        # Número total de episodios
alfa = 0.5              # Tasa de aprendizaje
gamma = 0.9            # Factor de descuento
epsilon = 1.0          # Cantidad de aleatoriedad en la selección de acciones
epsilon_decaimiento = 0.001  # Cantidad fija para disminuir

# Lista de resultados para graficar
resultados = []

print('Tabla Q antes del entrenamiento:')
print(qtable)

# Entrenamiento
for _ in range(episodios):
    estado = environment.reset()
    hecho = False

    # Por defecto, consideramos que nuestro resultado es un fracaso
    resultados.append("Fracaso")
    
    # Hasta que el agente quede atrapado en un agujero o alcance la meta, sigue entrenándolo
    while not hecho:
        # Generar un número aleatorio entre 0 y 1
        rnd = np.random.random()
        # Si el número aleatorio < epsilon, tomar una acción aleatoria
        if rnd < epsilon:
          accion = environment.action_space.sample()
        # De lo contrario, tomar la acción con el valor más alto en el estado actual
        else:
          accion = np.argmax(qtable[estado])
        
        # Implementar esta acción y mover al agente en la dirección deseada
        nuevo_estado, recompensa, hecho, info = environment.step(accion)

        # Actualizar Q(s,a)
        qtable[estado, accion] = qtable[estado, accion] + \
                                alfa * (recompensa + gamma * np.max(qtable[nuevo_estado]) - qtable[estado, accion])
        
        # Actualizar nuestro estado actual
        estado = nuevo_estado

        # Si tenemos una recompensa, significa que nuestro resultado es un éxito
        if recompensa:
          resultados[-1] = "Éxito"

    # Actualizar epsilon
    epsilon = max(epsilon - epsilon_decaimiento, 0)

print()
print('===========================================')
print('Tabla Q después del entrenamiento:')
print(qtable)

# Graficar resultados
plt.figure(figsize=(12, 5))
plt.xlabel("Número de ejecución")
plt.ylabel("Resultado")
ax = plt.gca()
plt.bar(range(len(resultados)), resultados, width=1.0)
plt.show()

#vemos el éxito
episodios = 100
num_exitos = 0

# Evaluación
for _ in range(100):
    estado = environment.reset()
    hecho = False
   
    # Hasta que el agente quede atrapado o alcance la meta, sigue entrenándolo
    while not hecho:
        # Elegir la acción con el valor más alto en el estado actual
        accion = np.argmax(qtable[estado])

        # Implementar esta acción y mover al agente en la dirección deseada
        nuevo_estado, recompensa, hecho, info = environment.step(accion)

        # Actualizar nuestro estado actual
        estado = nuevo_estado

        # Cuando obtenemos una recompensa, significa que resolvimos el juego
        num_exitos += recompensa

# ¡Veamos nuestra tasa de éxito!
print(f"Tasa de éxito = {num_exitos/episodios*100}%")

# hacemos lo mismo con el suelo resbaladizo
# Inicializar el Frozen Lake resbaladizo
environment = gym.make("FrozenLake-v1", is_slippery=True)
environment.reset()

# Re-inicializamos la tabla Q
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Hiperparámetros
episodios = 1000        # Número total de episodios
alfa = 0.5              # Tasa de aprendizaje
gamma = 0.9            # Factor de descuento
epsilon = 1.0          # Cantidad de aleatoriedad en la selección de acciones
epsilon_decaimiento = 0.001  # Cantidad fija para disminuir

# Lista de resultados para graficar
resultados = []

print('Tabla Q antes del entrenamiento:')
print(qtable)

# Entrenamiento
for _ in range(episodios):
    estado = environment.reset()
    hecho = False

    # Por defecto, consideramos que nuestro resultado es un fracaso
    resultados.append("Fracaso")
    
    # Hasta que el agente quede atrapado en un agujero o alcance la meta, sigue entrenándolo
    while not hecho:
        # Generar un número aleatorio entre 0 y 1
        rnd = np.random.random()

        # Si el número aleatorio < epsilon, tomar una acción aleatoria
        if rnd < epsilon:
          accion = environment.action_space.sample()

        # De lo contrario, tomar la acción con el valor más alto en el estado actual
        else:
          accion = np.argmax(qtable[estado])
       
        # Implementar esta acción y mover al agente en la dirección deseada
        nuevo_estado, recompensa, hecho, info = environment.step(accion)

        # Actualizar Q(s,a)
        qtable[estado, accion] = qtable[estado, accion] + \
                                alfa * (recompensa + gamma * np.max(qtable[nuevo_estado]) - qtable[estado, accion])

       
         # Actualizar nuestro estado actual
        estado = nuevo_estado

        # Si tenemos una recompensa, significa que nuestro resultado es un éxito
        if recompensa:
          resultados[-1] = "Éxito"

    # Actualizar epsilon
    epsilon = max(epsilon - epsilon_decaimiento, 0)

print()
print('===========================================')
print('Tabla Q después del entrenamiento:')
print(qtable)

# Graficar resultados
plt.figure(figsize=(12, 5))
plt.xlabel("Número de ejecución")
plt.ylabel("Resultado")
ax = plt.gca()
plt.bar(range(len(resultados)), resultados, width=1.0)
plt.show()

episodios = 100
num_exitos = 0

# Evaluación
for _ in range(100):
    estado = environment.reset()
    hecho = False
    
    # Hasta que el agente quede atrapado o alcance la meta, sigue entrenándolo
    while not hecho:
        # Elegir la acción con el valor más alto en el estado actual
        accion = np.argmax(qtable[estado])

        # Implementar esta acción y mover al agente en la dirección deseada
        nuevo_estado, recompensa, hecho, info = environment.step(accion)

        # Actualizar nuestro estado actual
        estado = nuevo_estado

        # Cuando obtenemos una recompensa, significa que resolvimos el juego
        num_exitos += recompensa

# ¡Veamos nuestra tasa de éxito!
print(f"Tasa de éxito = {num_exitos/episodios*100}%")
