# **Training an Agent to Play CartPole using Deep Q-Learning (DQN)**  

## 📌 **Introduction**  

This project demonstrates how to train a reinforcement learning agent to play **CartPole**, a classic problem in control theory, using **Deep Q-Learning (DQN)**.  

By the end of this project, you will understand:  
✔ **Reinforcement Learning (RL) fundamentals**  
✔ **How CartPole works and what makes it challenging**  
✔ **The Q-learning algorithm and its limitations**  
✔ **Deep Q-Learning (DQN) and how it improves Q-learning**  
✔ **How to train an AI agent to balance the pole using neural networks**  

---

## 🎯 **What is Reinforcement Learning?**  

**Reinforcement Learning (RL)** is a type of machine learning where an **agent** learns how to make decisions by interacting with an **environment** to maximize a **reward**.  

### **Key Components of RL**  

| Term         | Description |
|-------------|-------------|
| **Agent**   | The decision-making entity (our AI model) |
| **Environment** | The world where the agent operates (CartPole) |
| **State**   | A set of variables representing the environment (e.g., pole angle, cart velocity) |
| **Action**  | A decision the agent can make (e.g., move cart left or right) |
| **Reward**  | A signal that tells the agent how good its action was (+1 per timestep the pole is balanced) |
| **Policy**  | A strategy mapping states to actions |
| **Q-Function** | A function that estimates the future reward for taking an action in a given state |

### **The Goal of CartPole**  
In **CartPole**, the agent controls a cart that moves left or right to keep a pole balanced. The episode ends when:  
- The pole **falls past 15 degrees** from vertical.  
- The cart moves **out of bounds (±2.4 units)**.  
- The agent successfully balances the pole for **500 time steps** (winning condition).  

### **CartPole Environment Overview**  

```
         Pole (θ)
           |
           |
           |
  ---------|---------   <- Cart moves left/right
           |
```

---

## 🧠 **What is Q-Learning?**  

**Q-learning** is a reinforcement learning algorithm that helps an agent learn the best action to take in any given state using a **Q-table**.  

### **Q-Function Formula**  

\[
Q(s, a) = Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
\]

Where:  
- \( Q(s, a) \) = Current Q-value for taking action \( a \) in state \( s \).  
- \( \alpha \) = Learning rate (how much new information replaces old knowledge).  
- \( r \) = Immediate reward.  
- \( \gamma \) = Discount factor (importance of future rewards).  
- \( \max_{a'} Q(s', a') \) = Maximum expected future reward from the next state \( s' \).  

### **Why Not Just Use Q-Tables?**  

Q-learning stores values in a **Q-table**, which works well for **small environments**. However, **CartPole has continuous state values** (like velocity and angles), making it impossible to store all possible values in a table.  

To solve this, we replace the **Q-table with a deep neural network**! 🚀  

---

## 🤖 **What is Deep Q-Learning (DQN)?**  

**Deep Q-Learning (DQN)** is an extension of Q-learning that uses a **neural network** to approximate the **Q-values** instead of a table.  

### **How DQN Works**  

1. **Agent Observes** the current state \( s \).  
2. **Neural Network Predicts** Q-values for all possible actions.  
3. **Agent Chooses** the action with the highest Q-value (or explores randomly).  
4. **Environment Responds** with a new state \( s' \) and reward \( r \).  
5. **Replay Memory Stores** experiences for training later.  
6. **Neural Network Updates** using backpropagation to improve Q-value predictions.  

### **DQN Architecture**  

```
 State (4 values: cart position, velocity, pole angle, pole velocity)
       │
       ▼
 +-----------------+
 |   Input Layer   |  <- Takes state (4 inputs)
 +-----------------+
       │
       ▼
 +-----------------+
 | Hidden Layer 1  |  <- Learns patterns
 +-----------------+
       │
       ▼
 +-----------------+
 | Hidden Layer 2  |  <- More learning depth
 +-----------------+
       │
       ▼
 +-----------------+
 |  Output Layer   |  <- Predicts Q-values for actions (left, right)
 +-----------------+
       │
       ▼
 Decision: Move Left or Right
```

### **Improvements of DQN over Basic Q-Learning**  

✔ **Uses a neural network** to generalize over large state spaces  
✔ **Employs experience replay** (stores past experiences and samples them randomly for training)  
✔ **Uses a target network** (separate network for stable learning updates)  

---

## 📊 **Training Process**  

The training involves running **multiple episodes**, where the agent interacts with the environment and **gradually improves** by learning from past mistakes.  

### **Training Steps**  

1️⃣ Initialize the environment and **reset** the agent.  
2️⃣ The agent **observes the state** (cart position, velocity, pole angle, pole velocity).  
3️⃣ **Choose an action** using an **epsilon-greedy strategy** (random actions for exploration + Q-values for exploitation).  
4️⃣ Take the action and **store the experience** (state, action, reward, next state) in memory.  
5️⃣ **Train the neural network** using random samples from the memory (experience replay).  
6️⃣ Repeat for many episodes until the agent **balances the pole consistently**.  

---

## 🎓 **Conclusion**  

✔ **Reinforcement Learning** helps AI learn optimal decisions by trial and error.  
✔ **Q-Learning** is a simple but powerful technique for decision-making.  
✔ **DQN (Deep Q-Learning)** solves problems with large state spaces using neural networks.  
✔ **CartPole is a great beginner’s environment** to understand RL in action.  

🚀 **Next Steps**:  
- Try tuning **hyperparameters** (learning rate, discount factor, epsilon decay).  
- Modify the **neural network architecture** to improve learning speed.  
- Experiment with **more advanced algorithms** like PPO, A3C, or SAC.  

---

## 🏆 **Want to Learn More?**  

📘 **Reinforcement Learning Resources**  
- **"Reinforcement Learning: An Introduction"** by Richard Sutton & Andrew Barto  
- OpenAI Gym documentation: [https://gym.openai.com](https://gym.openai.com)  
- DeepMind's DQN Paper: [https://deepmind.com/research/highlighted-research/dqn](https://deepmind.com/research/highlighted-research/dqn)  

🚀 **Hands-on RL Platforms**  
- **Google Colab** (Run RL models in the cloud)  
- **Stable Baselines3** (Pre-built RL agents)  
- **Unity ML-Agents** (Train RL models in 3D worlds)  

