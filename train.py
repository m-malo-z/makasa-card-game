from makasa_env import MakasaEnv
from agent import MakasaAgent
import numpy as np
import csv
import os

N_EPISODES     = 5000   #total training games
LOG_EVERY      = 100    #log metrics every N episodes
SNAPSHOT_EVERY = 500    #save weights every N episodes
LOG_FILE       = "training_log.csv"

#run one full game, return True if agent wins
def run_episode(env, agent):
    state = env.reset()
    done  = False
    turns = 0

    while not done:
        turns += 1
        if turns > 500:         
            break
        moves      = env.get_legal_moves()
        action_idx = agent.choose_action(state, moves, env)
        action     = moves[action_idx] if action_idx is not None else None

        next_state, reward, done = env.step(action)

        scaled_reward = reward / 100.0
        
        agent.update(state, reward, next_state, done)
        state = next_state

    won         = reward == 100.0
    final_score = 0.0 if won else abs(reward)
    return won, final_score

def train():
    env   = MakasaEnv()
    agent = MakasaAgent()

    wins        = []   # 1=win, 0=loss for each episode
    scores      = []   # final score each episode
    log_records = []   # for CSV

    print(f"Training for {N_EPISODES} episodes...")
    print(f"{'Episode':>8} {'Win rate':>10} {'Avg score':>10} {'Epsilon':>8}")
    print("-" * 42)

    for ep in range(1, N_EPISODES + 1):
        if ep <= 5:
            print(f"Starting episode {ep}...")
        won, score = run_episode(env, agent)
        wins.append(1 if won else 0)
        scores.append(score)
        agent.decay_epsilon()

        #Log every LOG_EVERY episodes
        if ep % LOG_EVERY == 0:
            window_wins  = wins[-LOG_EVERY:]
            window_scores = scores[-LOG_EVERY:]
            win_rate     = sum(window_wins) / len(window_wins)
            avg_score    = sum(window_scores) / len(window_scores)

            print(f"{ep:>8} {win_rate:>10.2%} {avg_score:>10.2f} {agent.epsilon:>8.4f}")
            log_records.append({
                'episode':   ep,
                'win_rate':  round(win_rate, 4),
                'avg_score': round(avg_score, 2),
                'epsilon':   round(agent.epsilon, 4)
            })

    #Save weight snapshot
        if ep % SNAPSHOT_EVERY == 0:
            agent.save_weights(f"weights_{ep:04d}.json")

    # Save final weights and log
    agent.save_weights("weights_final.json")

    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode','win_rate','avg_score','epsilon'])
        writer.writeheader()
        writer.writerows(log_records)

    print(f"\nDone. Log saved to {LOG_FILE}")
    print(f"Final weights: {np.round(agent.weights, 4)}")
    return log_records

if __name__ == '__main__':
    results = train()

    #Quick summary
    first_100 = [r for r in results if r['episode'] <= 500]
    last_100  = [r for r in results if r['episode'] > N_EPISODES - 500]

    if first_100 and last_100:
        early_wr = sum(r['win_rate'] for r in first_100) / len(first_100)
        late_wr  = sum(r['win_rate'] for r in last_100)  / len(last_100)
        print(f"\nLearning summary:")
        print(f"  Early win rate (ep 1-500):    {early_wr:.2%}")
        print(f"  Late  win rate (ep 2500-3000): {late_wr:.2%}")
        if late_wr > early_wr:
            print("Agent improved over training")
        else:
            print("No clear improvement")
