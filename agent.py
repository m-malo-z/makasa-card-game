"""

Reinforcement learning agent for the Makasa card game.
Uses a linear value function V(b) weight updates and an epsilon-greedy action selection policy.

"""

import numpy as np
import json

#hyperparameters 
ALPHA     = 0.001    #learning rate, how fast weights update
GAMMA     = 0.9     #discount factor, how much future rewards matter instead of immediate rewards
EPSILON   = 0.5     #initial exploration rate
EPS_MIN   = 0.05    #minimum exploration 
EPS_DECAY = 0.9995   #how fast exploration sinks

N_FEATURES = 6      #features x1 through x6

class MakasaAgent:
    def __init__(self):
        #initialise small random values so weights are not all equal
        self.weights = np.random.uniform(-0.1, 0.1, N_FEATURES + 1)
        self.epsilon = EPSILON

    def value(self, state):
        #1.0 for the bias weight w0
        features = np.concatenate(([1.0], state))
        return float(np.dot(self.weights, features)) #calculate linear value function V(b)

    def gradient(self, state):
        return np.concatenate(([1.0], state))

    def choose_action(self, state, legal_moves, env):
        if not legal_moves:
            return None  #no moves at all 

        #Explore random move
        if np.random.random() < self.epsilon:
            return np.random.choice(len(legal_moves))

        # Exploit, evaluate each legal move and pick best
        best_action = None
        best_value  = float('-inf')

        for i, action in enumerate(legal_moves):
            # Simulate: what state would this action lead to
            next_state = env.simulate_move(action)
            v = self.value(next_state)
            if v > best_value:
                best_value  = v
                best_action = i

        return best_action

    def update(self, state, reward, next_state, done):
        v_current = self.value(state)

        if done:
            #no future value if game is over
            td_target = reward
        else:
            td_target = reward + GAMMA * self.value(next_state)

        td_error = td_target - v_current
        td_error = np.clip(td_error, -10.0, 10.0)
        
        grad     = self.gradient(state)

        #weight update
        self.weights += ALPHA * td_error * grad

        if np.any(np.isnan(self.weights)) or np.any(np.abs(self.weights) > 1000):
            print("Warning: weights reset due to overflow")
            self.weights = np.random.uniform(-0.1, 0.1, N_FEATURES + 1)

    def decay_epsilon(self):
        #reduce exploration rate after each episode
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)

    def save_weights(self, path="weights.json"):
        with open(path, 'w') as f:
            json.dump(self.weights.tolist(), f)
        print(f"Weights saved to {path}")

    def load_weights(self, path="weights.json"):
        with open(path, 'r') as f:
            self.weights = np.array(json.load(f))
        print(f"Weights loaded from {path}")

if __name__ == '__main__':
    from makasa_env import MakasaEnv

    env   = MakasaEnv()
    agent = MakasaAgent()
    state = env.reset()

    print("Initial weights:", agent.weights)
    print("Initial V(state):", round(agent.value(state), 4))

    # Test one move
    moves      = env.get_legal_moves()
    action_idx = agent.choose_action(state, moves, env)
    action     = moves[action_idx] if action_idx is not None else None

    next_state, reward, done = env.step(action)
    agent.update(state, reward, next_state, done)

    print("After 1 update:")
    print("  Action taken:", action)
    print("  Reward:", reward)
    print("  New weights:", np.round(agent.weights, 4))
    print("  Weight changed:", not np.allclose(agent.weights,
          np.random.uniform(-0.1, 0.1, N_FEATURES + 1)))
