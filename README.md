Dynamic Pricing Optimization with Reinforcement Learning
========================================================

This project explores the use of **Reinforcement Learning (RL)** to optimize dynamic pricing strategies across multiple products. By simulating market dynamics and customer behavior, we train an RL agent to maximize total profit over a defined time period.

Table of Contents
-----------------

*   [Introduction](#introduction)
    
*   [Project Structure](#project-structure)
    
*   [Getting Started](#getting-started)
    
    *   [Prerequisites](#prerequisites)
        
    *   [Installation](#installation)
        
*   [Simulation Environment](#simulation-environment)
    
    *   [Products and Market Dynamics](#products-and-market-dynamics)
        
    *   [Customer Behavior Simulation](#customer-behavior-simulation)
        
*   [Reinforcement Learning Framework](#reinforcement-learning-framework)
    
    *   [State, Action, and Reward Definitions](#state-action-and-reward-definitions)
        
*   [Agent Development](#agent-development)
    
    *   [Deep Q-Network (DQN)](#deep-q-network-dqn)
        
    *   [Experience Replay](#experience-replay)
        
*   [Training the Agent](#training-the-agent)
    
    *   [Action Space](#action-space)
        
    *   [Training Loop](#training-loop)
        
*   [Evaluation](#evaluation)
    
    *   [Training Progress](#training-progress)
        
    *   [Pricing Strategies Analysis](#pricing-strategies-analysis)
        
    *   [Baseline Comparison](#baseline-comparison)
        
*   [Results](#results)
    
*   [Conclusion](#conclusion)
    
*   [License](#license)
    

Introduction
------------

Dynamic pricing is a strategy where businesses adjust the prices of their products or services in response to market demands, competition, and other external factors. This project implements a reinforcement learning approach to optimize pricing strategies for multiple products over time, aiming to maximize total profit.

Project Structure
-----------------

The project is organized into the following sections:

1.  **Import Libraries**: Importing necessary Python libraries.
    
2.  **Environment Setup**: Defining products, market dynamics, and simulating customer behavior.
    
3.  **Reinforcement Learning Framework**: Defining state, action, and reward structures.
    
4.  **Implement the Environment**: Creating the pricing environment for the agent to interact with.
    
5.  **Develop the RL Agent**: Implementing a Deep Q-Network (DQN) and experience replay.
    
6.  **Training the Agent**: Training the agent over multiple episodes.
    
7.  **Evaluation**: Analyzing the agent's performance and comparing it to a baseline strategy.
    

Getting Started
---------------

### Prerequisites

*   Python 3.7 or higher
    
*   [NumPy](https://numpy.org/)
    
*   Pandas
    
*   [Matplotlib](https://matplotlib.org/)
    
*   [PyTorch](https://pytorch.org/)
    

### Installation

1.  bashCopy codegit clone https://github.com/yourusername/dynamic-pricing-rl.gitcd dynamic-pricing-rl
    
2.  bashCopy codepython -m venv venvsource venv/bin/activate # On Windows use \`venv\\Scripts\\activate\`
    
3.  bashCopy codepip install -r requirements.txt
    
4.  bashCopy codejupyter notebookOpen Dynamic\_Pricing\_RL.ipynb in the Jupyter interface.
    

Simulation Environment
----------------------

### Products and Market Dynamics

We define a set of products, each with:

*   **Cost**: The production cost of the product.
    
*   **Base Price**: The standard selling price.
    
*   **Price Range**: Minimum and maximum allowable prices.
    

Example products:

*   **Product A**: Cost $30, Base Price $50
    
*   **Product B**: Cost $20, Base Price $40
    
*   **Product C**: Cost $50, Base Price $80
    
*   **Product D**: Cost $15, Base Price $25
    
*   **Product E**: Cost $100, Base Price $150
    

Market dynamics include:

*   **Max Demand**: The maximum potential demand.
    
*   **Price Sensitivity**: How demand changes with price.
    
*   **Sigma**: Standard deviation for demand fluctuation.
    

### Customer Behavior Simulation

Customer demand is simulated using an exponential decay function influenced by price and random fluctuations:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopy codedef calculate_demand(product, price):      # Exponential decay demand function with random noise      ...      return demand   `

Reinforcement Learning Framework
--------------------------------

### State, Action, and Reward Definitions

*   **State**: Current prices of all products.
    
*   **Action**: Price adjustments for each product.
    
*   **Reward**: Total profit for the current time step.
    

Agent Development
-----------------

### Deep Q-Network (DQN)

A neural network approximates the optimal action-value function:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopy codeclass DQNAgent(nn.Module):      def __init__(...):          super(DQNAgent, self).__init__()          ...   `

### Experience Replay

An experience replay buffer stores past experiences to break correlation between consecutive samples:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopy codeself.memory = deque(maxlen=buffer_size)   `

Training the Agent
------------------

### Action Space

Price adjustments are discretized into small steps:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopy codeprice_steps = [-2, -1, 0, 1, 2]  # Possible price changes   `

### Training Loop

The agent interacts with the environment, learns from experiences, and updates its policy:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopy codefor episode in range(num_episodes):      state = env.reset()      while not env.done:          action_indices = agent.select_action(state)          action = [price_steps[idx] for idx in action_indices]          next_state, reward, done, info = env.step(action)          agent.remember(state, action_indices, reward, next_state, done)          agent.replay()          state = next_state      agent.update_target_network()   `

Evaluation
----------

### Training Progress

Training metrics are plotted to visualize learning progress:

*   **Total Reward per Episode**
    
*   **Total Profit per Episode**
    

### Pricing Strategies Analysis

The agent's pricing strategies over time are analyzed and plotted for each product.

### Baseline Comparison

The RL agent's performance is compared against a baseline strategy with no price adjustments.

Results
-------

*   **Total Profit - RL Agent**: $2,919,129.40
    
*   **Total Profit - Baseline Strategy**: $2,177,019.50
    
*   **Profit Improvement**: 34.09%
    

**Average Daily Profit**:

*   **RL Agent**: $7,997.61
    
*   **Baseline Strategy**: $5,964.44
    

**Units Sold Comparison**:

ProductUnits Sold (RL Agent)Units Sold (Baseline)Product A6,04515,183Product B9,90722,291Product C1,7611,901Product D95,150138,877Product E752664

**Average Selling Price Comparison**:

ProductAvg. Price (RL Agent)Avg. Price (Baseline)Product A$74.18$50.00Product B$59.75$40.00Product C$116.73$80.00Product D$37.30$25.00Product E$221.25$150.00

Conclusion
----------

The reinforcement learning agent successfully learned dynamic pricing strategies that significantly improved total profit compared to the baseline. By adjusting prices in response to simulated market dynamics, the agent optimized the balance between price and demand.

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.
