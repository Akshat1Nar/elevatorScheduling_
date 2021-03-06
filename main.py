import gym, time
from network.Agent import Agent as Agent
# from utils import plotLearning
from game.Building import Building
import numpy as np

total_elevator_num = 1
max_floor = 7
max_passengers_in_floor = 3
max_passengers_in_elevator = 5
passenger_max_num = 5
max_time = 20

if __name__ == '__main__':
    env = Building(total_elevator_num = total_elevator_num, max_floor = max_floor,
                   max_passengers_in_floor = max_passengers_in_floor,
                   max_passengers_in_elevator = max_passengers_in_elevator,
                   max_time = max_time)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64,
                  n_actions=4**total_elevator_num, eps_end=0.01,
                  input_dims=[2,], lr=0.001)

    scores, eps_history = [], []
    n_games = 10

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        env.generate_passengers(0.7, passenger_max_num  = passenger_max_num)
        step = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                        observation_, done)
            agent.learn()
            observation = observation_

            env.render(env.time_taken)
            time.sleep(0.3)
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)

    # x = [i+1 for i in range(n_games)]
    # filename = 'lunar_lander.png'
    # plotLearning(x, scores, eps_history, filename)
