import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy

class ChessPPO(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChessPPO, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        #print("forwardstate: ", state)
        policy_probs = self.policy(state)
        value = self.value(state)
        return policy_probs, value

class PPOAgent:
    def __init__(self, input_size, output_size, learning_rate=0.001, gamma=0.99, epsilon=0.2):
        self.policy_network = ChessPPO(input_size, 64, output_size)

        self.policy_network.load_state_dict(torch.load("trained.pth"))

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        policy_probs, _ = self.policy_network(state)
        action_distribution = torch.distributions.Categorical(policy_probs)
        action = action_distribution.sample()
        #print("action: ", action)
        #return action.item()
        m = Categorical(policy_probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        advantage = 0.0
        rewards= numpy.array(rewards)

        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            mask = 1.0 - done
            advantage = advantage * self.gamma * mask + reward + self.gamma * value * mask
            advantages.append(advantage)

        advantages.reverse()
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def update_policy(self, statess, actionss, old_probss, advantagess, rewards):
        #print("paramstatess",statess, "actionss",actionss, "rewards",rewards, "log_probs",old_probss, "advantages",advantagess)
        states = torch.from_numpy(numpy.array(statess)).float()
        actions = torch.tensor(actionss, dtype=torch.int64)
        old_probs = torch.tensor(old_probss, dtype=torch.float32)
        advantages = torch.tensor(advantagess, dtype=torch.float32)
        #print("inn------states",states, "actions",actions, "rewards",rewards, "log_probs",old_probs, "advantages",advantages)


        new_probs, values = self.policy_network(states)
        action_masks = torch.nn.functional.one_hot(actions, num_classes=new_probs.shape[-1])
        chosen_action_probs = torch.sum(action_masks * new_probs, dim=-1)

        ratio = chosen_action_probs / old_probs

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages

        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        #print("values", values, "rewards", rewards)
        rewards= numpy.array(rewards)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        value_loss = nn.MSELoss()(values.view(-1), rewards)

        total_loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    env = chess.Board()

    state_size = 64  # Chess board size
    action_size = 4096  # Number of possible chess moves (64 squares * 64 squares)
    hidden_size = 128

    agent = ChessPPOAgent(state_size, hidden_size, action_size)

    max_episodes = 1000

    for episode in range(max_episodes):
        state = env.fen()

        done = False
        states, actions, rewards, values, old_probs, dones = [], [], [], [], [], []

        while not done:
            # Encode the state (FEN notation) into a one-hot vector or other suitable representation
            # For simplicity, we're using a flattened one-hot encoding of the board positions here
            state_array = np.array([int(c) for c in state if c.isdigit() or c.isalpha()])
            action = agent.select_action(state_array)

            states.append(state_array)
            actions.append(action)

            old_probs.append(agent.policy_network(torch.from_numpy(state_array).float())[0][action].item())

            # Apply the action to the environment
            env.push(chess.Move.from_uci(chess.SQUARE_NAMES[action]))

            # Simulate the chess engine playing a move
            engine_move = chess.engine.SimpleEngine.popen_uci("stockfish").play(env, chess.engine.Limit(time=0.1))
            env.push(engine_move.move)

            state = env.fen()

            # Reward function can be customized based on the task
            reward = 1.0 if env.is_checkmate() else 0.0

            rewards.append(reward)
            values.append(agent.policy_network(torch.from_numpy(state_array).float())[1].item())
            dones.append(env.is_game_over())

        next_value = 0.0 if env.is_checkmate() else agent.policy_network(torch.from_numpy(state_array).float())[1].item()
        advantages = agent.compute_advantages(rewards, values + [next_value], dones)

        agent.update_policy(states, actions, old_probs, advantages, rewards)

        print(f"Episode: {episode}, Total Reward: {np.sum(rewards)}")

    env.close()
