"""
Game: Makasa 2-player 
Deck: 54 cards (4 suits x 13 ranks + 2 Jokers)
"""

import random
import numpy as np

# Cards
SUITS = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
RANKS = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']
SUIT_IDX = {s: i for i, s in enumerate(SUITS)} 
RANK_IDX = {r: i for i, r in enumerate(RANKS)}

# Scoring points for cards remaining in hand at the end
SCORE = {
    '3':3, '4':4, '5':5, '6':6, '7':7, '8':8,
    '9':9, '10':10, 'J':10, 'Q':10, 'K':10,
    '2':20, 'A':100, 'JOKER':50
}

# Special card 
SKIP_RANKS    = {'J'} #Jack: next player loses turn
REVERSE_RANKS = {'Q'} #Queen: reverse direction
EXTRA_RANKS   = {'K'} #King: current player extra turn
WILD_RANKS    = {'A'} #Ace: choose suit OR cancel penalty
PENALTY_RANKS = {'2', 'JOKER'} #Penalty cards

#build a shuffled deck as an array of suit and ranks
def build_deck():
    deck = [(suit, rank) for suit in SUITS for rank in RANKS]
    deck += [('JOKER', 'JOKER'), ('JOKER', 'JOKER')]
    random.shuffle(deck)
    return deck

#get the score of a certain card
def card_score(card):
    suit, rank = card
    return SCORE.get(rank, int(rank) if rank.isdigit() else 10)

#encode cards with an integer value between 0 and 52
def encode_card(card):
    suit, rank = card
    if suit == 'JOKER':
        return 52  # both Jokers map to 52
    return SUIT_IDX[suit] * 13 + RANK_IDX[rank]

#returns true if card is playable on top card and false if not
def is_playable(card, top_card, current_suit, penalty_stack=0):
    suit, rank = card
    top_suit, top_rank = top_card

    #Jokers are always playable
    if suit == 'JOKER':
        return True
    
    #Ace cancels penalties and is always playable
    if rank == 'A':
        return True

    #If penalty is stacking, only penalty cards or Ace can be played
    if penalty_stack > 0:
        if rank == '2' and (top_rank == '2'):
            return True  #2 stacks on 2
        if rank == 'JOKER':
            return True  #Joker stacks on 2 or Joker
        return False     #2 cannot stack on Joker

    #card playable when a match to top card's suit or rank
    if suit == current_suit:
        return True
    if rank == top_rank:
        return True

    return False

#Makasa game environment in a class
class MakasaEnv:
    def __init__(self):
        self.reset()

    #initialise new game
    def reset(self):
        self.deck = build_deck()
        self.hands = [[], []] #hands[0] = agent, hands[1] = opponent
        for _ in range(5):
            self.hands[0].append(self.deck.pop())
            self.hands[1].append(self.deck.pop())
        
        #Flip top card, must not be a special card
        while True:
            top = self.deck.pop()
            if top[1] not in (SKIP_RANKS | REVERSE_RANKS | WILD_RANKS | PENALTY_RANKS):
                break
            self.deck.insert(0, top) #put top card at the bottom

        self.discard = [top]
        self.current_suit = top[0]
        self.penalty_stack = 0
        self.current_player = 0 #agent always starts
        self.done = False
        return self.get_state_vector()

    #Get the 6 feature vectors of the current player
    def get_state_vector(self):
        top = self.discard[-1]
        return np.array([
            len(self.hands[0]), #x1: agent hand size
            len(self.hands[1]), #x2: opponent hand size
            encode_card(top) / 53.0, #x3: top card as an integer between 0-53
            SUIT_IDX.get(self.current_suit, 0) / 3.0, #x4: current suit
            min(self.penalty_stack, 10) / 10.0, #x5: penalty stack
            sum(1 for c in self.hands[0] #x6: special cards in hand
                if c[1] in (SKIP_RANKS | REVERSE_RANKS |
                            WILD_RANKS | PENALTY_RANKS)
                or c[0] == 'JOKER') / 5.0
        ], dtype=np.float32)

    #get playable cards from agent's hand
    def get_legal_moves(self):
        top = self.discard[-1]
        moves = [c for c in self.hands[0]
                 if is_playable(c, top, self.current_suit, self.penalty_stack)]
        moves.append(None)  #None = draw from pile
        return moves

    #with the curent player action, return the state feature vector, reward and whether it is a win or loss 
    def step(self, action):
        if action is None:
            if self.deck:
                drawn = self.deck.pop()
                self.hands[0].append(drawn)
                if self.penalty_stack > 0:
                    for _ in range(self.penalty_stack - 1):
                        if self.deck:
                            self.hands[0].append(self.deck.pop())
                            self.penalty_stack = 0
            else:
                #Deck is empty and no legal moves and end the game
                self.done = True
                penalty = sum(card_score(c) for c in self.hands[0])
                return self.get_state_vector(), -float(penalty), True
        else:
            #Play the card
            self.hands[0].remove(action)
            self.discard.append(action)
            suit, rank = action

            #Update suit when Ace is played. Agent can choose suit or default to most common suit
            if rank == 'A':
                from collections import Counter
                suits_in_hand = [c[0] for c in self.hands[0] if c[0] != 'JOKER']
                self.current_suit = Counter(suits_in_hand).most_common(1)[0][0] if suits_in_hand else suit
                self.penalty_stack = 0  #Ace cancels penalty
            elif suit != 'JOKER':
                self.current_suit = suit

            #Penalty stacking
            if rank == '2':
                self.penalty_stack += 2
            elif suit == 'JOKER':
                self.penalty_stack += 5

            #Check win and reward agent
            if len(self.hands[0]) == 0:
                self.done = True
                return self.get_state_vector(), 100.0, True
            
        #Opponent plays a random legal move
        self._opponent_turn()

        #Check if opponent won and penalise agent
        if len(self.hands[1]) == 0:
            self.done = True
            penalty = sum(card_score(c) for c in self.hands[0])
            return self.get_state_vector(), -float(penalty), True

        if not self.deck and not self.get_legal_moves():
            self.done = True
            penalty = sum(card_score(c) for c in self.hands[0])
            return self.get_state_vector(), -float(penalty), True

        return self.get_state_vector(), 0.0, False

    #random opponent plays a legal card or draws cards
    def _opponent_turn(self):
        top = self.discard[-1]
        legal = [c for c in self.hands[1]
                 if is_playable(c, top, self.current_suit, self.penalty_stack)]
        if legal:
            card = random.choice(legal)
            self.hands[1].remove(card)
            self.discard.append(card)
            suit, rank = card
            if suit != 'JOKER':
                self.current_suit = suit
            if rank == '2':
                self.penalty_stack += 2
            elif suit == 'JOKER':
                self.penalty_stack += 5
            elif rank == 'A':
                self.penalty_stack = 0
        elif self.deck:
            self.hands[1].append(self.deck.pop())
        else:
            self.done = True

    def simulate_move(self, action):
        """
        Return the state vector that would result from playing action,
        WITHOUT modifying the actual game state.
        Used by the agent to evaluate moves before committing.
        """
        import copy
        sim = copy.deepcopy(self)

        # Apply the action on the simulation
        if action is None:
            if sim.deck:
                sim.hands[0].append(sim.deck.pop())
        else:
            if action in sim.hands[0]:
                sim.hands[0].remove(action)
                sim.discard.append(action)
                suit, rank = action
                if rank == 'A':
                    sim.penalty_stack = 0
                    from collections import Counter
                    suits = [c[0] for c in sim.hands[0] if c[0] != 'JOKER']
                    sim.current_suit = Counter(suits).most_common(1)[0][0] if suits else suit
                elif suit != 'JOKER':
                    sim.current_suit = suit
                if rank == '2':
                    sim.penalty_stack += 2
                elif suit == 'JOKER':
                    sim.penalty_stack += 5

        return sim.get_state_vector()
    

#test
if __name__ == '__main__':
    env = MakasaEnv()
    state = env.reset()
    print(f"Starting state: {state}")
    print(f"Agent hand: {env.hands[0]}")
    print(f"Top card: {env.discard[-1]}")
    
    for turn in range(20):
        moves = env.get_legal_moves()
        # Pick first legal card, or draw
        action = moves[0] if moves[0] is not None else None
        state, reward, done = env.step(action)
        print(f"Turn {turn+1}: played {action}, reward={reward:.1f}, done={done}")
        if done:
            print("Game over!")
            break
