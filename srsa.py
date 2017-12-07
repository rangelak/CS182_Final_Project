"""
---------------------------------------------
CS 182 Final Project:
SARSA Trading Agent
---------------------------------------------
Ben Barrett, Rahul Naidoo, Rangel Milushev
"""

import math
from quantopian.algorithm import calendars
import random
import datetime

# The Quantopian-specific initialize function to initialize all global variables & functions
def initialize(context):
    
    # Scheduling function to update the data daily
    schedule_function(
        func=daily_data, 
        date_rule=date_rules.every_day(), 
        time_rule=time_rules.market_open(hours=0, minutes=1), 
        calendar=calendars.US_EQUITIES, 
        half_days=True)
    
    # Scheduling function to run the main function daily
    schedule_function(
        func=mainFunction, 
        date_rule=date_rules.every_day(), 
        time_rule=time_rules.market_open(hours=0, minutes=1), 
        calendar=calendars.US_EQUITIES, 
        half_days=True)
    
    # Initializing global variables and parameters
    context.alpha = 0.2  # learning coeff
    context.discount = 0.9  # discount factor
    context.security = sid(39840)  # currently Tesla
    context.today_opening = 0
    context.yesterday_opening = 0
    context.change = 0
    context.agent = SARSA_Learner(context.discount, context.alpha)
    context.previous_state = (0, 0)
    context.previous_action = "buy" # first/default action is to buy
    context.current_state = None
    context.current_action = None
    context.time = 0
    context.epsilon_parameter = 0.01 # randomizing parameter

    # Testing use
    context.random_counter = 0 # counts number of random actions
    context.state_counter_dict = {} 
    context.CompletedDate = datetime.date(2017,12,1)


# Function that updates parameters daily
def daily_data(context, data):
    context.today_opening = data.current(context.security, 'open')
    context.yesterday_opening = data.history(context.security, 'open', 2, '1d')[0]
    context.change = percent_change(context.yesterday_opening, context.today_opening)
    
    # Below is used for testing
    """
    if (get_datetime().date() == context.CompletedDate):  
      for key in context.state_counter_dict.keys():
          print key
          print "--------------"
    """


# Returns a percentage context.change, rounded to the nearest integer
def percent_change(previous, current):
    difference = current - previous
    change_quotient = difference / previous
    result = round(change_quotient * 100)
    if result == -0.0:
        result = 0.0
    return result

# Buy function to be executed when we want to buy a stock
def buy(context):
    if not get_open_orders():
      cash = context.portfolio.cash

      # Check to see if we have money to buy
      if cash < 0:
        return 1
      else:
        order_target_percent(context.security, 1)
    return 0

# Sell function to be executed when we want to sell a stock
def sell(context):
    positions_value = context.portfolio.positions_value
    order_target_percent(context.security, 0)
    return 0

# Main function that executes daily
def mainFunction(context, data):
    
    # Get the change in percentage
    context.change = percent_change(context.yesterday_opening, context.today_opening)
    
    # Make sure we are getting a number
    while math.isnan(context.change):
        context.change = percent_change(context.yesterday_opening, context.today_opening)

    # Update the previous and current states
    if not context.current_state:
        context.current_state = context.previous_state
        context.current_action = context.previous_action
    else:
        context.previous_state = context.current_state
        context.current_state = get_next_state(context.previous_action, context.previous_state, context)
        context.previous_action = context.current_action
        context.current_action = context.agent.getPolicy(context.current_state)
        
        # Testing use
        if context.current_state in context.state_counter_dict:
            context.state_counter_dict[context.current_state] += 1
        else:
            context.state_counter_dict[context.current_state] = 0

    # Explore a random option with probability epsilon
    epsilon = math.exp((-1) * context.epsilon_parameter * context.time)
    acceptance_probability = random.random()
    buy_arr = ["sell", "hold"]
    sell_arr = ["buy", "hold"]
    hold_arr = ["buy", "sell"]
    
    # In the case of epsilon we want a different action than the SARSA one
    if acceptance_probability < epsilon:
        context.random_counter += 1
        if context.current_action == "buy":
            random.shuffle(buy_arr)
            context.current_action = buy_arr[0]
        elif context.current_action == "sell":
            random.shuffle(sell_arr)
            context.current_action = sell_arr[0]
        else:
            random.shuffle(hold_arr)
            context.current_action = hold_arr[0]

    # According to the action buy, sell or hold
    if context.current_action == "buy":
        if context.portfolio.cash > context.today_opening:
            buy(context)
    elif context.current_action == "sell":
        if context.portfolio.positions_value > 0:
            sell(context)

    # Get the next state and next action
    next_state_temp = get_next_state(context.current_action, context.current_state, context)
    next_action_temp = context.agent.getPolicy(next_state_temp)
    
    # Update the agent
    context.agent.update(
        context.current_state, 
        context.current_action, 
        next_state_temp, 
        next_action_temp,
        compute_reward(
            context.current_action, 
            context.change,
            context.current_state[0]))
    
    # Record the time
    context.time += 1

# Function to compute the reward
def compute_reward(action, percent_change, position):
    return percent_change * position

# Function to get the next state
def get_next_state(action, state, context):
    if not context.portfolio.positions:
        pos = 0
    else:
        pos = 1
    ret_state = (pos, context.change)
    return ret_state

# Class for the SARSA agent
class SARSA_Learner:

    # Initialize the agent with the discount and learning factors
    def __init__(self, discount, alpha):
        self.qValues = dict()
        self.discount = discount
        self.alpha = alpha
        self.actions = ["buy", "sell", "hold"]
    
    # Get the Q-value    
    def getQValue(self, state, action):
        if (state, action) in self.qValues:
          return self.qValues[(state, action)]
        else:
          self.qValues[(state, action)] = 0.0
          return 0.0

    # Compute the action from the Q Values
    def computeActionFromQValues(self, state):
        max_val = None
        max_action = None
        for action in self.actions:
          curr_val = self.getQValue(state, action)
          if max_val == None or max_val < curr_val:
            max_val = curr_val
            max_action = action
        return max_action

    # Update the values
    def update(self, state, action, nextState, nextAction, reward):
        if (state, action) not in self.qValues:
          self.qValues[(state, action)] = 0.0
        state_val = self.qValues[(state, action)]
        next_val = self.getQValue(nextState, nextAction)
        self.qValues[(state, action)] = state_val + self.alpha * (self.discount * next_val - state_val + reward)

    # Return the action for a state
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)