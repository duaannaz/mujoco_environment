import gymnasium as gym
import numpy as np

env = gym.make("HalfCheetah-v5", render_mode="human")

def initialize_population(p_size,k):
    population = []
    for i in range(p_size):
        chromosome=[]
        for j in range(k):
            torques=np.random.uniform(-1.0,1.0,6)
            duration=np.array([np.random.randint(10,50)])
            keyframe=np.concatenate([torques,duration])
            chromosome.append(keyframe)
        population.append(chromosome)
    return population

def evaluate_fitness(chromosome,env):
    K=len(chromosome)
    keyframes=chromosome
    observation,info=env.reset(seed=None)

    keyframe_index=0
    remaining_steps = int(keyframes[keyframe_index][-1])

    episode_reward = 0

    while True:
        # Select action from the current keyframe
        action = keyframes[keyframe_index][:-1]

        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
    
        # Decrement the remaining steps for this keyframe
        remaining_steps -= 1

        # If the keyframe duration is completed, move to the next keyframe
        if remaining_steps == 0:
            keyframe_index = (keyframe_index + 1) % K  # Cycle through keyframes
            remaining_steps = int(keyframes[keyframe_index][-1])

        # If the episode has ended (time limit reached)
        if terminated or truncated:
            print(f"Episode ended. Total reward: {episode_reward}")
            return episode_reward 
        
def binary_tournament(population,fitness):
    parents=[]
    for i in range(2):
        p_size=len(population)
        idx1,idx2=np.random.randint(0,p_size,2)
        if(fitness[idx1]>fitness[idx2]):
            parents.append(population[idx1])
        else:
            parents.append(population[idx2])
    return parents

