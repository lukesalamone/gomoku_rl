from agent import GomokuAgent
from environment import GomokuEnvironment

NUM_ITERATIONS = 100                            # alpha go used 80 here
NUM_EPISODES = 250                              # alpha go used 25000 here
THRESHOLD = 0.55                                # same as alpha go paper
BOARD_SIZE = 9


def policy_iteration():
    env = GomokuEnvironment(size=BOARD_SIZE)
    
    # initialise random neural network
    agent = GomokuAgent(size=BOARD_SIZE)

    examples = []
    for _ in range(NUM_ITERATIONS):
        for _ in range(NUM_EPISODES):
            # collect examples from this game
            examples += exec_episode(env, agent)
        new_agent = GomokuAgent().train(examples)
        
        # compare new net with previous net
        frac_win = pit(agent, new_agent, env, num_games=200)
        if frac_win > THRESHOLD:
            # replace with new net
            agent = new_agent
    return agent



def exec_episode(env, agent):
    examples = []
    state = env.reset()
    done = False
    reward = None

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        
        # reward = None b/c we don't know it yet
        examples.append([state, action, next_state, reward])

    examples = assign_rewards(examples, reward)
    return examples

def assign_rewards(examples, reward):
    reward0 = reward if len(examples)%2 == 1 else -reward
    reward1 = -reward0

    for i in range(len(examples)):
        examples[i][3] = reward0 if i%2 == 0 else reward1

    return examples


# play agent1 against agent2 and return win percentage for agent2
def pit(agent1, agent2, env, num_games):
    wins = 0

    for game in range(num_games):

        # agents take turn moving first
        player1,player2 = agent1,agent2 if game%2 == 0 else agent2,agent1

        # play one game
        result = play_game(player1, player2, env)

        if game%2 == 0 and result == -1:
            wins += 1
        if game%2 == 1 and result == 1:
            wins += 1
    
    return wins/200



def play_game(player1, player2, env):
    env.reset()
    done = False
    move_counter = 0
    reward = 0

    while not done:
        available_moves = env.available_moves()
        state = env.board_state()

        if move_counter%2 == 0:
            move = player1.select_move(state, available_moves)
        else:
            move = player2.select_move(state, available_moves)
        assert move in available_moves, 'agent selected illegal move!'

        _, reward, done, info =  env.step(move)

    # reward = 1 if player1 wins, -1 if player2 wins, 0 if draw
    return reward