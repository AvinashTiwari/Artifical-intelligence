import numpy as np
gamma = 0.75
alpha = 0.9

location_to_state = {'A' : 0,
                     'B' : 1,
                     'C' : 2,
                     'D' : 3,
                     'E' : 4,
                     'F' : 5,
                     'G' : 6,
                     'H' : 7,
                     'I' : 8,
                     'J' : 9,
                     'K' : 10,
                     'L' : 11}

actions = [0,1,2,3,4,5,6,7,8,9,10,11]

R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])

Q = np.array(np.zeros([12,12]))

for i in range(1000):
    current_state = np.random.randint(0,12)
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
    TD = R[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])]- Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha*TD
            