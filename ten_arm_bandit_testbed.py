import numpy as np
import matplotlib.pyplot as plt

def k_arm_bandit(steps=1000,epsilon=0,n=2000,k=10):
    qbase = np.random.normal(size=(n,k))
    qstar = qbase.argmax(axis=1)
    
    optimal = [0]
    cumulative_rewards = [0]
    
    total_rewards = np.zeros(shape=(n,k))
    action_history =np.zeros(shape=(n,k),dtype=np.int32)
    
    for i in range(steps):
        
        # Sample-average method for estimating action values
        if np.any(action_history==0):
            temp = np.where(action_history==0,0.01,action_history)
            qhat = total_rewards/temp
        else:
            qhat = total_rewards/action_history
            
    
        # Take action
        greedy_actions = qhat.argmax(axis=1)    
        probs = np.random.uniform(size=n)
        selected_actions = np.where(probs > epsilon, 
                                    greedy_actions, 
                                    np.random.randint(0,10,size=n))
        np.sum(greedy_actions==selected_actions)
        np.sum(selected_actions == qstar)
        
        # Record number of optimal action taken
        optimal.append(np.sum(selected_actions==qstar))
    
        # Record action taken
        action_history[np.arange(n),selected_actions] += 1    
        
        # Get rewards
        rewards = qbase + np.random.randn(n,k)
        average_reward = np.average(rewards[np.arange(n),selected_actions],axis=0)
        cumulative_rewards.append(average_reward)
        total_rewards[np.arange(n),selected_actions] += rewards[np.arange(n),selected_actions]
    return (optimal,cumulative_rewards)

greedy = k_arm_bandit(epsilon=0)
epsgreedy01 = k_arm_bandit(epsilon=0.1)
epsgreedy001 = k_arm_bandit(epsilon=0.01)


n=200
# Plot of average rewards
for data in [greedy,epsgreedy001,epsgreedy01]:
    optimal,cumulative_rewards = data
    steps = len(cumulative_rewards)
    plt.plot(range(steps),cumulative_rewards)

# Plot % optimal action
for data in [greedy,epsgreedy001,epsgreedy01]:    
    optimal,_ = data
    steps = len(optimal)
    optimal_action = np.array(optimal)/n
    plt.plot(range(steps+1),optimal_action)
