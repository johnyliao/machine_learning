import numpy as np
import matplotlib.pyplot as plt

def k_arm_bandit(steps=1000,epsilon=0,n=2000,k=10):
    qbase = np.random.normal(size=(n,k))
    qstar = qbase.argmax(axis=1)
    
    optimal = [0]
    reward_history = np.ndarray(shape=(steps,n,k))
    cumulative_rewards = [0]
    action_history =np.zeros(shape=(n,k),dtype=np.int32)
    for i in range(steps):
        rewards = qbase + np.random.normal(size=(n,k))
        reward_history[i] = rewards
        reward_history.shape
        if i == 0:
            selected_actions = rewards.argmax(axis=1)    
            action_history[np.arange(n),selected_actions] += 1
        else:
            temp = np.where(action_history==0,0.0001,action_history)
            qhat = np.sum(reward_history,axis=0)/temp
            greedy_actions = qhat.argmax(axis=1)
            probs = np.random.uniform(size=n)
            selected_actions = np.where(probs > epsilon, 
                                        greedy_actions, 
                                        np.random.randint(0,10,size=n))   
            action_history[np.arange(n),selected_actions] += 1
        cumulative_rewards.append(np.average(rewards[np.arange(n),selected_actions],axis=0))
        optimal.append(np.sum(selected_actions==qstar))
    return (optimal,cumulative_rewards)

greedy = k_arm_bandit(epsilon=0)
epsgreedy01 = k_arm_bandit(epsilon=0.1)
epsgreedy001 = k_arm_bandit(epsilon=0.01)
