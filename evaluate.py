"""
Plot results from training the agent on Makasa
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import json
from makasa_env import MakasaEnv, card_score
from agent import MakasaAgent

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.family'] = 'sans-serif'

def plot_win_rate(log_file="training_log.csv"):
    df = pd.read_csv(log_file)

    plt.figure(figsize=(9, 4))
    plt.plot(df['episode'], df['win_rate'] * 100,
             color='#534AB7', linewidth=1.5, label='Win rate (per 100 episodes)')

    #Trend line
    z = np.polyfit(df['episode'], df['win_rate'] * 100, 1)
    p = np.poly1d(z)
    plt.plot(df['episode'], p(df['episode']),
             color='#1D9E75', linewidth=1.5, linestyle='--', label='Trend')

    plt.xlabel('Training episode')
    plt.ylabel('Win rate (%)')
    plt.title('Agent win rate over training (Makasa RL)')
    plt.legend()
    plt.ylim(0, 60)
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.show()
    print("Saved: learning_curve.png")

def plot_avg_score(log_file="training_log.csv"):
    df = pd.read_csv(log_file)

    plt.figure(figsize=(9, 4))
    plt.plot(df['episode'], df['avg_score'],
             color='#D85A30', linewidth=1.5, label='Avg score when losing')

    z = np.polyfit(df['episode'], df['avg_score'], 1)
    p = np.poly1d(z)
    plt.plot(df['episode'], p(df['episode']),
             color='#BA7517', linewidth=1.5, linestyle='--', label='Trend')

    plt.xlabel('Training episode')
    plt.ylabel('Average final score (points)')
    plt.title('Average final score over training (lower = better)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('score_curve.png')
    plt.show()
    print("Saved: score_curve.png")

#compare win rate of trained agent and random opponent
def compare_vs_random(n_games=200):
    env          = MakasaEnv()
    trained      = MakasaAgent()
    trained.load_weights("weights_final.json")
    trained.epsilon = 0.0   # no exploration, pure exploitation

    trained_wins = 0
    random_wins  = 0

    for i in range(n_games):
        state = env.reset()
        done  = False
        turns = 0

        while not done and turns < 500:
            turns += 1
            #Trained agent chooses best move
            moves      = env.get_legal_moves()
            action_idx = trained.choose_action(state, moves, env)
            action     = moves[action_idx] if action_idx is not None else None
            state, reward, done = env.step(action)

        if reward == 100.0:
            trained_wins += 1
        else:
            random_wins += 1

    trained_wr = trained_wins / n_games * 100
    random_wr  = random_wins  / n_games * 100

    print(f"\nResults over {n_games} games:")
    print(f"  Trained agent win rate: {trained_wr:.1f}%")
    print(f"  Random opponent win rate: {random_wr:.1f}%")

    #Bar chart
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Trained agent', 'Random opponent'],
                   [trained_wr, random_wr],
                   color=['#534AB7', '#888780'], width=0.4)
    plt.ylabel('Win rate (%)')
    plt.title(f'Trained agent vs random opponent ({n_games} games)')
    plt.ylim(0, 100)
    for bar, val in zip(bars, [trained_wr, random_wr]):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1.5,
                 f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_bar.png')
    plt.show()
    print("Saved: comparison_bar.png")
    return trained_wr, random_wr

def interpret_weights():
    with open("weights_final.json") as f:
        w = json.load(f)

    features = [
        ("w0 (bias)",              "baseline value — general optimism/pessimism"),
        ("w1 (agent hand size)",   "expected: negative — fewer cards is better"),
        ("w2 (opponent hand size)","expected: positive — opponent having more cards is good"),
        ("w3 (top card encoded)",  "expected: near zero — card identity matters less than suit"),
        ("w4 (current suit)",      "expected: near zero — suit alone has weak signal"),
        ("w5 (penalty stack)",     "expected: negative — active penalty is bad"),
        ("w6 (special cards)",     "expected: positive — more options is better"),
    ]

    print("\nFinal weight interpretation:")
    print(f"{'Weight':<10} {'Value':>8}   Feature + expected direction")
    print("-" * 65)
    for i, (name, desc) in enumerate(features):
        direction = "Good" if (
            (i == 1 and w[i] < 0) or
            (i == 2 and w[i] > 0) or
            (i == 5 and w[i] < 0) or
            (i in [0,3,4,6])
        ) else "unexpected"
        print(f"{name:<10} {w[i]:>8.4f}   {desc}  {direction}")

if __name__ == '__main__':
    print("Graph 1: Win rate learning curve")
    plot_win_rate()

    print("\nGraph 2: Average score curve")
    plot_avg_score()

    print("\nExperiment: Trained vs random")
    trained_wr, random_wr = compare_vs_random(200)

    print("\nWeight interpretation")
    interpret_weights()
