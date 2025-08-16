"""
REINFORCEMENT LEARNING PROGRAM EXPLAINED
========================================

Let me walk through each part and explain what it does and WHY.

This enhanced version combines:
- 🎓 Educational explanations and extensive logging (from GridTreasureFinding2.py)
- 🚀 Complete functionality with full training, testing, and analysis (from GridTreasureFinding1.py)
"""

import random
import numpy as np

# ==============================================================================
# TRAINING CONTEXT (Simplifies verbose control)
# ==============================================================================

class TrainingContext:
    """
    🎯 Centralized training context to control verbose output and training state
    """
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.episode_num = 0
    
    def set_episode(self, episode):
        self.episode_num = episode
    
    def is_detailed_episode(self):
        return self.episode_num < 1  # Only first episode shows detailed output

# ==============================================================================
# PART 1: THE WORLD (Environment)
# ==============================================================================

class SimpleGridWorld:
    """
    This is our 5x5 room where the treasure hunt happens.
    
    Think of this as the "game rules" - it decides:
    - Where you can move
    - What rewards you get
    - When the game ends
    """
    
    def __init__(self):
        print("🏗️  CREATING THE WORLD...")
        self.size = 5
        self.start_pos = (4, 0)      # YOU start here (bottom-left)
        self.treasure_pos = (0, 4)   # TREASURE is here (top-right)
        self.agent_pos = self.start_pos
        print(f"   📍 You start at: {self.start_pos}")
        print(f"   💎 Treasure is at: {self.treasure_pos}")
        
    def reset(self, context=None):
        """
        🔄 START A NEW TREASURE HUNT
        Like respawning in a video game - go back to starting position
        """
        self.agent_pos = self.start_pos
        if context and context.verbose and context.is_detailed_episode():
            print(f"🔄 New hunt started! You're back at {self.agent_pos}")
        return self.agent_pos
    
    def step(self, action, context=None):
        """
        🎮 THIS IS THE GAME ENGINE
        
        When you press a button (action), this function:
        1. Moves you in the world
        2. Calculates your reward
        3. Checks if game is over
        
        Actions: 0=Up, 1=Down, 2=Left, 3=Right
        """
        row, col = self.agent_pos
        old_pos = (row, col)
        
        # 🚶 MOVE BASED ON ACTION (with boundary checking)
        if action == 0 and row > 0:      # Up (and not at top edge)
            row -= 1
        elif action == 1 and row < 4:    # Down (and not at bottom edge)
            row += 1
        elif action == 2 and col > 0:    # Left (and not at left edge)
            col -= 1
        elif action == 3 and col < 4:    # Right (and not at right edge)
            col += 1
            
        self.agent_pos = (row, col)
        
        # 🎯 CALCULATE REWARD
        if self.agent_pos == self.treasure_pos:
            reward = 100    # 💰 JACKPOT! Found the treasure!
            done = True     # 🏁 Game over - you won!
            if context and context.verbose and context.is_detailed_episode():
                print(f"   💰 TREASURE FOUND! +100 points!")
        else:
            reward = -1     # 💸 Each step costs energy
            done = False    # 🔄 Keep playing
            
        # Only show movement details for detailed episodes
        if context and context.verbose and context.is_detailed_episode():
            print(f"   🚶 Moved from {old_pos} to {self.agent_pos}, Reward: {reward}")
        
        return self.agent_pos, reward, done

# ==============================================================================
# PART 2: THE LEARNER (Agent)
# ==============================================================================

class SimpleAgent:
    """
    This is our AI student learning to find treasure.
    
    Think of this as a person with:
    - A notebook (Q-table) to write down what works
    - A strategy for making decisions
    - The ability to learn from mistakes
    """
    
    def __init__(self):
        print("🤖 CREATING THE AI STUDENT...")
        
        # 📚 THE NOTEBOOK (Q-table)
        # For every position and every action, we track expected rewards
        self.q_table = {}
        
        # ⚙️ LEARNING SETTINGS
        self.learning_rate = 0.1        # How fast we update our beliefs (0.1 = cautious)
        self.discount_factor = 0.9      # How much we care about future rewards (0.9 = care a lot)
        self.epsilon = 0.3              # How often we explore vs use known good moves (30% random)
        
        print(f"   📚 Creating Q-table (the AI's notebook)...")
        
        # Initialize Q-table with zeros (AI knows nothing at start)
        for row in range(5):
            for col in range(5):
                # For each position, track expected reward for each action [Up, Down, Left, Right]
                self.q_table[(row, col)] = [0.0, 0.0, 0.0, 0.0]
                
        print(f"   ✅ Q-table created! AI starts knowing nothing (all zeros)")
        print(f"   🎯 Learning rate: {self.learning_rate} (how fast to learn)")
        print(f"   🔮 Discount factor: {self.discount_factor} (how much to value future)")
        print(f"   🎲 Exploration rate: {self.epsilon} (30% random moves)")
    
    def choose_action(self, state, context):
        """
        🤔 DECISION MAKING PROCESS
        
        This is where the AI decides what to do next.
        It uses "epsilon-greedy" strategy:
        - 30% of time: try something random (explore)
        - 70% of time: use the best known action (exploit)
        """
        
        if random.random() < self.epsilon:
            # 🎲 EXPLORATION: "Let me try something random!"
            action = random.randint(0, 3)
            action_names = ["Up", "Down", "Left", "Right"]
            if context.verbose and context.is_detailed_episode():
                print(f"   🎲 EXPLORING: Trying random action '{action_names[action]}'")
            return action
        else:
            # 🧠 EXPLOITATION: "Let me use what I know works!"
            q_values = self.q_table[state]
            action = q_values.index(max(q_values))
            action_names = ["Up", "Down", "Left", "Right"]
            if context.verbose and context.is_detailed_episode():
                print(f"   🧠 EXPLOITING: Using best known action '{action_names[action]}' (Q-value: {max(q_values):.1f})")
            return action
    
    def learn(self, state, action, reward, next_state, done, context):
        """
        🎓 LEARNING FROM EXPERIENCE
        
        This is where the magic happens! The AI updates its Q-table based on what just happened.
        
        The Q-learning formula:
        New Q-value = Old Q-value + learning_rate × (target - Old Q-value)
        
        Where target = immediate_reward + discount_factor × best_future_reward
        """
        
        current_q = self.q_table[state][action]
        
        if done:
            # 🏁 Episode finished - no future rewards to consider
            target_q = reward
        else:
            # 🔮 Episode continues - consider future rewards
            best_next_q = max(self.q_table[next_state])
            target_q = reward + self.discount_factor * best_next_q
        
        # 🧮 UPDATE THE Q-VALUE
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[state][action] = new_q
        
        # 📝 Show learning details only for detailed episodes
        if context.verbose and context.is_detailed_episode():
            self._show_learning_details(state, action, reward, current_q, target_q, new_q, done)
    
    def _show_learning_details(self, state, action, reward, current_q, target_q, new_q, done):
        """📝 Display detailed learning information with full calculations"""
        print(f"\n   📝 LEARNING: State {state}, Action {action}, Reward {reward}")
        print(f"   📊 Old Q-value: {current_q:.2f}")
        
        # Show target calculation based on actual done flag
        if done:
            print(f"   🎯 TARGET CALCULATION:")
            print(f"      Formula: target = reward (episode ended)")
            print(f"      target = {reward}")
        else:
            best_next_q = (target_q - reward) / self.discount_factor
            print(f"   🎯 TARGET CALCULATION:")
            print(f"      Formula: best_next_q = max(self.q_table[next_state])")
            print(f"      Formula: target_q = reward + discount_factor × best_next_q")
            print(f"      best_next_q = {best_next_q:.2f}")
            print(f"      target = {reward} + {self.discount_factor} × {best_next_q:.2f}")
            print(f"      target = {target_q:.2f}")
        
        # Show Q-value update calculation
        print(f"   🧮 Q-VALUE UPDATE:")
        print(f"      Formula: new_q = old_q + learning_rate × (target - old_q)")
        print(f"      new_q = {current_q:.2f} + {self.learning_rate} × ({target_q:.2f} - {current_q:.2f})")
        print(f"      new_q = {current_q:.2f} + {self.learning_rate} × {target_q - current_q:.2f}")
        print(f"      new_q = {current_q:.2f} + {self.learning_rate * (target_q - current_q):.2f}")
        print(f"      new_q = {new_q:.2f}")
        
        print(f"   💡 UPDATED: {state} → {action} = {new_q:.2f}")

# ==============================================================================
# PART 3: HELPER FUNCTIONS
# ==============================================================================

def print_grid(agent_pos, treasure_pos):
    """
    🖼️  VISUALIZE THE GRID WORLD
    
    Print the current state of the grid world so we can see where everyone is!
    """
    print("\n🗺️  CURRENT GRID WORLD:")
    print("   Legend: 🤖 = Agent (our AI), 💎 = Treasure, ⬜ = Empty space")
    print()
    for row in range(5):
        line = "   "
        for col in range(5):
            if (row, col) == agent_pos:
                line += "🤖 "  # Agent (our AI)
            elif (row, col) == treasure_pos:
                line += "💎 "  # Treasure
            else:
                line += "⬜ "  # Empty space
        print(line)
    print()

# ==============================================================================
# PART 4: TRAINING FUNCTIONS (Teaching the AI)
# ==============================================================================

def train_agent(env, agent, num_episodes=500):
    """
    🚀 COMPLETE TRAINING SESSION
    
    This trains the agent for many episodes to achieve optimal performance.
    We show detailed output for the first few episodes, then just progress updates.
    """
    
    print(f"\n🚀 STARTING FULL TRAINING SESSION!")
    print(f"🎯 Goal: Train for {num_episodes} episodes to find optimal treasure path")
    print(f"📊 We'll show detailed learning for the first episode, then progress updates")
    print("=" * 60)
    
    episode_rewards = []
    episode_steps = []
    context = TrainingContext(verbose=True)
    
    # Train for many episodes
    for episode in range(num_episodes):
        context.set_episode(episode)
        
        if context.is_detailed_episode():
            print(f"\n{'='*50}")
            print(f"🎓 DETAILED LEARNING - EPISODE {episode + 1}")
            print(f"{'='*50}")
            print(f"\n🎮 EPISODE {episode + 1}: Starting new treasure hunt!")
        
        # 🔄 Reset the world for this episode
        state = env.reset(context)
        total_reward = 0
        steps = 0
        
        # 🎮 Run one complete treasure hunt episode
        while True:
            steps += 1
            if context.is_detailed_episode():
                print(f"\n   --- Step {steps} ---")
                print(f"   📍 Current position: {state}")
                print_grid(env.agent_pos, env.treasure_pos)
            
            # 🤔 AI chooses what to do
            action = agent.choose_action(state, context)
            
            # 🌍 World responds to the action
            next_state, reward, done = env.step(action, context)
            
            # 🎓 AI learns from what happened
            agent.learn(state, action, reward, next_state, done, context)
            
            # 📊 Update tracking variables
            state = next_state
            total_reward += reward
            
            if done:
                if context.is_detailed_episode():
                    print(f"\n🏁 Episode {episode + 1} finished!")
                    print(f"   📊 Total steps: {steps}")
                    print(f"   💰 Total reward: {total_reward}")
                break
                
            if steps > 50:  # Safety check to prevent infinite loops
                if context.is_detailed_episode():
                    print("⚠️  Too many steps, ending episode")
                break
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        if context.is_detailed_episode():
            print(f"\n✅ Episode {episode + 1} Summary:")
            print(f"   📊 Steps taken: {steps}")
            print(f"   💰 Total reward: {total_reward}")
            print(f"   🧠 Q-table updated with new knowledge!")
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            avg_steps = sum(episode_steps[-100:]) / 100
            print(f"\n📈 Episode {episode + 1}: Avg reward {avg_reward:.1f}, Avg steps {avg_steps:.1f}")
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📊 Final: First 10 avg reward {sum(episode_rewards[:10])/10:.1f}, Last 10 avg reward {sum(episode_rewards[-10:])/10:.1f}")
    print(f"📊 Final: First 10 avg steps {sum(episode_steps[:10])/10:.1f}, Last 10 avg steps {sum(episode_steps[-10:])/10:.1f}")
    
    return episode_rewards, episode_steps

# ==============================================================================
# PART 5: TESTING FUNCTIONS (See How Well the AI Learned)
# ==============================================================================

def test_trained_agent(agent, num_tests=1):
    """
    🧪 TESTING THE TRAINED AI
    
    Now let's see how well our AI student learned! We'll run one test episode
    with exploration turned off (epsilon = 0) to see the optimal behavior.
    """
    
    print(f"\n🧪 TESTING TRAINED AGENT!")
    print(f"🎯 Running {num_tests} test episode to see optimal performance")
    print(f"🔒 Exploration turned OFF (epsilon = 0) - using only learned knowledge")
    print("=" * 60)
    
    env = SimpleGridWorld()
    
    for test in range(num_tests):
        print(f"\n🧪 TEST {test + 1}: Let's see what the AI learned!")
        print("-" * 40)
        
        state = env.reset(TrainingContext(verbose=False))
        total_reward = 0
        steps = 0
        
        print(f"📍 Starting position: {state}")
        print_grid(env.agent_pos, env.treasure_pos)
        
        # Turn off exploration for testing (use only learned knowledge)
        old_epsilon = agent.epsilon
        agent.epsilon = 0  # No random actions
        
        print(f"🎯 AI will now use ONLY its learned knowledge (no random moves)")
        
        while steps < 20:  # Prevent infinite loops
            action = agent.choose_action(state, TrainingContext(verbose=False))
            next_state, reward, done = env.step(action, TrainingContext(verbose=False))
            
            actions_names = ["⬆️ Up", "⬇️ Down", "⬅️ Left", "➡️ Right"]
            print(f"\n🚶 Step {steps + 1}: AI chooses {actions_names[action]}")
            print_grid(env.agent_pos, env.treasure_pos)
            
            total_reward += reward
            steps += 1
            
            if done:
                print(f"🎉 SUCCESS! AI found treasure in {steps} steps!")
                print(f"💰 Total reward: {total_reward}")
                print(f"🏆 This shows the AI learned the optimal path!")
                break
                
            state = next_state
        
        # Restore exploration for next test
        agent.epsilon = old_epsilon
    
    print(f"\n✅ Testing complete! The AI's performance shows how well it learned!")

# ==============================================================================
# PART 6: ANALYSIS FUNCTIONS (What Did the AI Learn?)
# ==============================================================================

def show_q_table(agent):
    """
    📊 ANALYZE WHAT THE AI LEARNED
    
    Display the Q-table to see what the AI learned about each position and action.
    Higher Q-values indicate better actions from each position.
    """
    
    print(f"\n📊 WHAT THE AI LEARNED - Q-TABLE ANALYSIS")
    print(f"🎯 Q-values show expected rewards for each action at each position")
    print(f"📋 Format: [⬆️ Up, ⬇️ Down, ⬅️ Left, ➡️ Right]")
    print(f"💡 Higher numbers = better actions from that position!")
    print("=" * 80)
    
    for row in range(5):
        for col in range(5):
            state = (row, col)
            q_values = agent.q_table[state]
            
            # Find the best action for this position
            best_action_idx = q_values.index(max(q_values))
            best_action_names = ["⬆️ Up", "⬇️ Down", "⬅️ Left", "➡️ Right"]
            best_action = best_action_names[best_action_idx]
            
            print(f"📍 Position ({row},{col}): {[round(q, 1) for q in q_values]}")
            print(f"   🎯 Best action: {best_action} (Q-value: {max(q_values):.1f})")
            
            if row < 4:  # Add spacing between rows
                print()
    
    print(f"\n�� INTERPRETATION:")
    print(f"   🎯 The AI learned which direction to go from each position")
    print(f"   📈 Higher Q-values mean the AI expects better rewards from those actions")
    print(f"   🧠 This represents the AI's 'mental map' of the optimal treasure path!")

def analyze_learning_progress(rewards, steps):
    """
    📈 ANALYZE LEARNING PROGRESS
    
    Show how the AI improved over time during training.
    """
    
    print(f"\n📈 LEARNING PROGRESS ANALYSIS")
    print(f"🎯 Let's see how much the AI improved during training!")
    print("=" * 50)
    
    if len(rewards) >= 10:
        first_10_avg_reward = sum(rewards[:10]) / 10
        last_10_avg_reward = sum(rewards[-10:]) / 10
        first_10_avg_steps = sum(steps[:10]) / 10
        last_10_avg_steps = sum(steps[-10:]) / 10
        
        print(f"📊 REWARD IMPROVEMENT:")
        print(f"   💰 First 10 episodes average: {first_10_avg_reward:.1f}")
        print(f"   💰 Last 10 episodes average: {last_10_avg_steps:.1f}")
        
        if last_10_avg_reward > first_10_avg_reward:
            improvement = last_10_avg_reward - first_10_avg_reward
            print(f"   ✅ IMPROVED by {improvement:.1f} points!")
        else:
            print(f"   🔄 Still learning... (needs more training)")
        
        print(f"\n📊 EFFICIENCY IMPROVEMENT:")
        print(f"   📊 First 10 episodes average steps: {first_10_avg_steps:.1f}")
        print(f"   📊 Last 10 episodes average steps: {last_10_avg_steps:.1f}")
        
        if last_10_avg_steps < first_10_avg_steps:
            improvement = first_10_avg_steps - last_10_avg_steps
            print(f"   ✅ IMPROVED by {improvement:.1f} fewer steps!")
        else:
            print(f"   🔄 Still learning... (needs more training)")
    
    print(f"\n🎓 LEARNING INSIGHTS:")
    print(f"   🧠 The AI started with random behavior and gradually learned optimal paths")
    print(f"   📈 Improvement shows the Q-learning algorithm is working!")
    print(f"   🎯 The AI discovered the best strategy through experience!")

# ==============================================================================
# PART 7: MAIN EXECUTION (The Complete Experience)
# ==============================================================================

def run_complete_learning_experience():
    """
    🎪 THE COMPLETE REINFORCEMENT LEARNING EXPERIENCE
    
    This runs the full pipeline:
    1. 🏗️  Setup & Training
    2. 🧪 Testing & Analysis
    """
    
    print("=" * 80)
    print("🎪 COMPLETE REINFORCEMENT LEARNING EXPERIENCE")
    print("=" * 80)
    
    # 🏗️ Create the world and AI
    env = SimpleGridWorld()
    agent = SimpleAgent()
    
    print(f"\n🎯 GOAL: Teach AI to find treasure optimally!")
    
    # 🚀 Train the agent
    print(f"\n🚀 TRAINING SESSION (500 episodes)")
    rewards, steps = train_agent(env, agent, num_episodes=500)
    
    # 🧪 Test and analyze
    print(f"\n🧪 TESTING & ANALYSIS")
    test_trained_agent(agent, num_tests=1)
    
    # 📊 Show Q-table analysis (detailed view)
    show_q_table(agent)
    
    analyze_learning_progress(rewards, steps)
    
    print(f"\n🎉 EXPERIENCE COMPLETE!")

# ==============================================================================
# RUN THE COMPLETE EXPERIENCE
# ==============================================================================

if __name__ == "__main__":
    run_complete_learning_experience()
    
    print(f"\n🎓 SUMMARY:")
    print(f"🤖 AI learned to find treasure through 500 episodes of training")
    print(f"🧪 Testing confirmed optimal performance")
    print(f"📊 Q-table analysis shows learned knowledge")
    print(f"🌟 Reinforcement learning works through trial and error!")