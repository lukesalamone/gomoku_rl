from environment import GomokuEnvironment
import wandb
from tqdm.autonotebook import tqdm

from agent import GomokuAgent
from minimax_agent import MinimaxAgent

NUM_ITERATIONS = 100  # alpha go used 80 here
NUM_EPISODES = 250  # alpha go used 25000 here
THRESHOLD = 0.55  # same as alpha go paper
NUM_PIT_GAMES = 400  # same as alpha go paper
NUM_MINIMAX_GAMES = 100
BOARD_SIZE = 9

# NUM_ITERATIONS = 1  # alpha go used 80 here
# NUM_EPISODES = 2  # alpha go used 25000 here
# THRESHOLD = 0.55  # same as alpha go paper
# NUM_PIT_GAMES = 10  # same as alpha go paper
# NUM_MINIMAX_GAMES = 10
# BOARD_SIZE = 9


def policy_iteration():
    wandb.init(project="oh-yeah", entity="gomokoolaid")

    wandb.config = {
        "num_iterations": NUM_ITERATIONS,
        "num_episodes": NUM_EPISODES,
        "threshold": THRESHOLD,
        "num_pit_games": NUM_PIT_GAMES,
        "num_minimax_games": NUM_MINIMAX_GAMES,
        "board_size": BOARD_SIZE 
    }
        
    env = GomokuEnvironment(size=BOARD_SIZE)

    # initialise random neural network
    agent = GomokuAgent(size=BOARD_SIZE, wandb=wandb)
    wandb.watch(agent.net)

    examples = []
    for iter_num in range(NUM_ITERATIONS):
        for _ in tqdm(range(NUM_EPISODES)):
            # collect examples from this game
            examples += exec_episode(env, agent)
        new_agent = GomokuAgent(size=BOARD_SIZE, wandb=wandb)
        wandb.watch(new_agent.net)
        new_agent.train(examples)

        # compare new net with previous net
        frac_win = pit(agent, new_agent, env, num_games=NUM_PIT_GAMES)
        wandb.log({"pit_win_pct": frac_win})
        if frac_win > THRESHOLD:
            # save old agent for posterity

            # replace with new net
            agent = new_agent
            wandb.log({"agent_swap": iter_num})
            print(f"new agent won {100*frac_win:0.2f}% of games, updating")

        play_minimax_games(NUM_MINIMAX_GAMES, agent, env)

    return agent


# play one game
def exec_episode(env, agent):
    examples = []
    env.reset()

    state = env.board()
    done = False
    reward = None

    # start with white
    color = 1

    while not done:
        action = agent.select_move(state, color, env.available_moves())
        next_state, reward, done = env.step(action)

        # reward = None b/c we don't know it yet
        examples.append(state)
        state = next_state

        color = 1 if color == -1 else -1

    examples = [(x, reward) for x in examples]
    return examples


# play agent1 against agent2 and return win percentage for agent2
def pit(agent1, agent2, env, num_games):
    wins = 0

    for game in tqdm(range(num_games), desc="playing pit games"):

        # agents take turn moving first
        if game % 2 == 0:
            player1 = agent1
            player2 = agent2
        else:
            player1 = agent2
            player2 = agent1

        # play one game
        result = play_game(player1, player2, env)

        if game % 2 == 0 and result == -1:
            wins += 1
        if game % 2 == 1 and result == 1:
            wins += 1

    return wins / num_games


def play_game(player1, player2, env):
    env.reset()
    done = False
    move_counter = 0
    reward = 0

    while not done:
        available_moves = env.available_moves()
        state = env.board()

        if move_counter % 2 == 0:
            move = player1.select_move(state, 1, available_moves)
        else:
            move = player2.select_move(state, -1, available_moves)
        assert move in available_moves, "agent selected illegal move!"

        _, reward, done = env.step(move)

    # reward = 1 if player1 wins, -1 if player2 wins, 0 if draw
    return reward


def play_minimax_games(num_games, agent, env):
    depth = 0
    win_pct = 1
    while win_pct > 0.5:
        # play some games, record win % of agent
        opponent = MinimaxAgent(agent.size, depth=depth)
        wins = 0
        for game in tqdm(range(num_games), desc='playing minimax games'):
            if game % 2 == 0:
                result = play_game(agent, opponent, env)
            else:
                result = -1 * play_game(opponent, agent, env)

            wins += 1 if result == 1 else 0
        win_pct = wins / num_games
        print(f"win % at minimax depth {depth} was {win_pct*100:0.1f}")
        depth += 1


if __name__ == "__main__":
    policy_iteration()
