import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


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

def roulette_wheel(population,fitnesses):
    fitness_shift=fitnesses.copy()
    min_val=min(fitness_shift)
    if min_val<=0:
        fitness_shift=[i-min_val for i in fitness_shift]
    total_fitness=sum(fitness_shift)
    if total_fitness==0:
        probabilities=[1/len(population)]*len(population)
    else:
        probabilities=[i/total_fitness for i in fitness_shift]
    idx1,idx2=np.random.choice(len(population),2,p=probabilities)
    parents=[population[idx1],population[idx2]]
    return parents

def truncation(population,fitnesses):
    indices=np.argsort(fitnesses)[::-1]
    cutoff=len(population)//2
    top_indices=indices[:cutoff]
    idx1,idx2=np.random.choice(top_indices,2,replace=False)
    parents=[population[idx1],population[idx2]]
    return parents

def single_point_crossover(parent1,parent2):
    noOfKeyframes=len(parent1)
    index=np.random.randint(1,noOfKeyframes)
    p1=[kf.copy() for kf in parent1[:index]]
    p1=p1 + [kf.copy() for kf in parent2[index:]]
    p2=[kf.copy() for kf in parent2[:index]]
    p2=p2 + [kf.copy() for kf in parent1[index:]]
    return (p1,p2)

def ox_crossover(parent1,parent2):
    noOfKeyframes=len(parent1)
    i,j=sorted(np.random.choice(noOfKeyframes,2,replace=False))

    child1=[kf.copy() for kf in parent2]
    child1[i:j]=[kf.copy() for kf in parent1[i:j]]

    child2=[kf.copy() for kf in parent1]
    child2[i:j]=[kf.copy() for kf in parent2[i:j]]

    return child1,child2

def swap_mutation(chromosome):
    k=len(chromosome)
    idx1,idx2=np.random.choice(k,2,replace=False)
    mutated=[kf.copy() for kf in chromosome]
    mutated[idx1],mutated[idx2]=mutated[idx2],mutated[idx1]

    return mutated

def insert_mutation(chromosome):
    k=len(chromosome)
    idx1,idx2=np.random.choice(k,2,replace=False)
    mutated=[kf.copy() for kf in chromosome]
    val=mutated.pop(idx2)
    mutated.insert(idx1+1,val)
    return mutated

def gaussian_mutation(chromosome,mutation_rate=0.1):
    mutated=[kf.copy() for kf in chromosome]
    for kf in mutated:
        for i in range(6):
            if np.random.random() < mutation_rate:
                kf[i]+=np.random.normal(0,0.1)
                kf[i]=np.clip(kf[i],-1.0,1.0) #keep within bounds
        # mutate duration occasionally
        if np.random.random() < mutation_rate:
            kf[-1]=max(1,int(kf[-1] + np.random.randint(-5,5)))
    return mutated

def survival_selection(population,children,fitnesses,children_fitnesses):
    combined = population+children
    combined_fitnesses=fitnesses+children_fitnesses
    sorted_indices=np.argsort(combined_fitnesses)[::-1]
    new_populaton=[combined[i] for i in sorted_indices[:len(population)]]
    new_fitnesses=[combined_fitnesses[i] for i in sorted_indices[:len(population)]]
    return new_populaton,new_fitnesses

def run_ea(generations,p_size,K,selection,crossover,mutation,env):
    population=initialize_population(p_size,K)
    best_rewards=[]
    fitnesses=[evaluate_fitness(c,env) for c in population]

    for gen in range(generations):
        best_rewards.append(max(fitnesses))
        best_idx=np.argmax(fitnesses)
        best_chromosome=population[best_idx]
        children=[]
        for i in range(p_size//2):
            parents=selection(population,fitnesses)
            child1,child2=crossover(parents[0],parents[1])
            child1,child2=mutation(child1),mutation(child2)
            children.append(child1)
            children.append(child2)
        
        children_fitnesses=[evaluate_fitness(c,env) for c in children]
        population,fitnesses=survival_selection(population,children,fitnesses,children_fitnesses)
    return best_chromosome,best_rewards

def save_keyframes(chromosome, filename="best_keyframes.txt"):
    with open(filename, "w") as f:
        f.write(f"{len(chromosome)}\n")
        for kf in chromosome:
            line = ", ".join(f"{v:.2f}" for v in kf)
            f.write(line + "\n")
    print(f"Saved best keyframes to {filename}")

def plot_all_convergence(results):
    for title, rewards in results.items():
        plt.plot(rewards, label=title)
    plt.xlabel("Generation")
    plt.ylabel("Best Reward")
    plt.title("Comparison of Selection Methods")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    env=gym.make("HalfCheetah-v5")
    K=8
    p_size=15
    generations=25

    results={}
    best_overall = None
    best_overall_reward=float('-inf')

    
    # Combination 1: Binary Tournament + Single Point + Gaussian
    print("Running Combination 1...")
    best, rewards = run_ea(generations, p_size, K, binary_tournament, single_point_crossover, gaussian_mutation, env)
    results["Binary Tournament + Single Point"] = rewards
    if max(rewards) > best_overall_reward:
        best_overall_reward = max(rewards)
        best_overall = best

    # Combination 2: Roulette Wheel + OX + Gaussian
    print("Running Combination 2...")
    best, rewards = run_ea(generations, p_size, K, roulette_wheel, ox_crossover, gaussian_mutation, env)
    results["Roulette Wheel + OX"] = rewards
    if max(rewards) > best_overall_reward:
        best_overall_reward = max(rewards)
        best_overall = best

    # Combination 3: Truncation + Single Point + Gaussian
    print("Running Combination 3...")
    best, rewards = run_ea(generations, p_size, K, truncation, single_point_crossover, gaussian_mutation, env)
    results["Truncation + Single Point"] = rewards
    if max(rewards) > best_overall_reward:
        best_overall_reward = max(rewards)
        best_overall = best

    # Combination 4: Binary Tournament + OX + Insert Mutation
    print("Running Combination 4...")
    best, rewards = run_ea(generations, p_size, K, binary_tournament, ox_crossover, insert_mutation, env)
    results["Binary Tournament + OX + Insert"] = rewards
    if max(rewards) > best_overall_reward:
        best_overall_reward = max(rewards)
        best_overall = best

    # Plot all combinations
    plot_all_convergence(results)

    # Save best keyframes
    save_keyframes(best_overall, "best_keyframes.txt")
    print(f"Best overall reward: {best_overall_reward}")

    # Visualize best chromosome with rendering
    print("Visualizing best solution...")
    env_render = gym.make("HalfCheetah-v5", render_mode="human")
    final_reward = evaluate_fitness(best_overall, env_render)
    print(f"Final reward: {final_reward}")
    env.close()
env.close()