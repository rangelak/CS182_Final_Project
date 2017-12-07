"""
---------------------------------------------
CS 182 Final Project:
An Approximate Q-Learning Trading Agent
---------------------------------------------
Ben Barrett, Rahul Naidoo, Rangel Milushev
"""

import math
from quantopian.algorithm import calendars
import random

# Quantopian's initializing function for global variables and parameters
def initialize(context):

    # Schedule function to update data daily
    schedule_function(
        func=daily_data, 
        date_rule=date_rules.every_day(), 
        time_rule=time_rules.market_open(hours=0, minutes=1), 
        calendar=calendars.US_EQUITIES, 
        half_days=True)

    # Schedule function to execute main daily
    schedule_function(
        func=mainFunction, 
        date_rule=date_rules.every_day(), 
        time_rule=time_rules.market_open(hours=0, minutes=1), 
        calendar=calendars.US_EQUITIES, 
        half_days=True)

    # Initialize variables and parameters
    context.alpha = 0.2
    context.discount = 0.9
    context.security = sid(39840)  # currently Tesla
    context.today_opening = 0
    context.yesterday_opening = 0
    context.first_price = 0
    context.second_price = 0
    context.change = 0
    context.three_day_lookback = 3
    context.two_day_lookback = 2
    context.agent = ApproximateQLearner(context.discount, context.alpha)
    context.previous_state = (0, 0)
    context.previous_action = "buy"
    context.current_state = None
    context.current_action = None
    context.features = Features()
    context.time = 0
    context.epsilon_parameter = 0.2 
    context.lowest_price = 10000000
    context.highest_price = 0

    # Initializing the dictionary of features and the weights
    context.features.update_features_dict(
        context.previous_state, 
        context.previous_action, 
        context.today_opening,
        context.yesterday_opening, 
        context.lowest_price, 
        context.highest_price)
    context.agent.initialize_weights(context)

    # Used for Testing Purposes
    context.random_counter = 0 
    context.state_counter_dict = {}

# Function to update data daily
def daily_data(context, data):
    context.today_opening = data.current(context.security, 'open')
    context.yesterday_opening = data.history(context.security, 'open', 2, '1d')[0]
    context.first_price = data.history(context.security, 'open', context.three_day_lookback, '1d')[0]
    context.second_price = data.history(context.security, 'open', context.two_day_lookback, '1d')[0]
    context.change = percent_change(context.yesterday_opening, context.today_opening)
    # For testing purposes
    # agent.getWeights 


# Returns a percentage context.change, rounded to the nearest integer
def percent_change(previous, current):
    difference = current - previous
    change_quotient = difference / previous
    result = round(change_quotient * 100)
    if result == -0.0:
        result = 0.0
    return result

# Function to buy stock
def buy(context):
    if not get_open_orders():
      cash = context.portfolio.cash
      if cash < 0:
        return 1
      else:
        order_target_percent(context.security, 1)
    return 0

# Function to sell stock
def sell(context):
    positions_value = context.portfolio.positions_value
    order_target_percent(context.security, 0)
    return 0

# Main function to be executed daily
def mainFunction(context, data):

    # Update the highest and lowest price
    if context.today_opening > context.highest_price:
        context.highest_price = context.today_opening
    if context.today_opening < context.lowest_price:
        context.lowest_price = context.today_opening

    # Update the percentage change
    context.change = percent_change(context.yesterday_opening, context.today_opening)
    while math.isnan(context.change):
        context.change = percent_change(context.yesterday_opening, context.today_opening)

    # Update the current and previous states
    if not context.current_state:
        context.current_state = context.previous_state
        context.current_action = context.previous_action
    else:
        context.previous_state = context.current_state
        context.current_state = get_next_state(context.previous_action, context.previous_state, context)
        context.previous_action = context.current_action
        context.current_action = context.agent.getPolicy(context.current_state, context)
        if context.current_state in context.state_counter_dict:
            context.state_counter_dict[context.current_state] += 1
        else:
            context.state_counter_dict[context.current_state] = 0

    # Explore a random option with probability epsilon
    epsilon = math.exp((-1) * context.epsilon_parameter * context.time)

    # Get random float between 0 and 1
    acceptance_probability = random.random()


    # Randomly choose one of the suboptimal actions and execute one of them
    buy_arr = ["sell", "hold"]
    sell_arr = ["buy", "hold"]
    hold_arr = ["buy", "sell"]
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

    # Execute the action: buy, sell or hold
    if context.current_action == "buy":
        if context.portfolio.cash > context.today_opening:
            buy(context)
    elif context.current_action == "sell":
        if context.portfolio.positions_value > 0:
            sell(context)

    # Calculate the next state and the reward
    nextState = get_next_state(context.current_action, context.current_state, context)
    reward = compute_reward(context.current_action, context.change, context.current_state[0])
    
    # Update the weights
    context.agent.update(
        context.current_state, 
        context.current_action, 
        nextState,
        reward, 
        context)
                         
    # Update time that has passed since beginning of trading
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

# Class for the Approximate Q-learning Agent
class ApproximateQLearner:

    # Initialize the agent
    def __init__(self, discount, alpha):
        self.weights = {}
        self.actions = ["buy", "sell", "hold"]
        self.discount = discount
        self.alpha = alpha

    # Initialize the weights
    def initialize_weights(self, context):
        features = context.features.FeatureDict
        for feature, feature_value in features.iteritems():
            self.weights[feature] = 1

    # Get the QValue
    def getQValue(self, state, action, context):
        q_sum = 0
        features = context.features.FeatureDict
        
        # Update the dictionary of features based on new info
        context.features.update_features_dict(
            state, 
            action, 
            context.today_opening, 
            context.yesterday_opening,
            context.lowest_price, 
            context.highest_price)

        # Add to the sum feature * weight_of_feature
        for feature, feature_value in features.iteritems():
            q_sum += feature_value * self.weights[feature]
        return q_sum
    
    # Compute best action according to QValue
    def computeActionFromQValues(self, state, context):
        actions = self.actions
        if not actions:
            return None
        max_val = None
        max_action = None
        for action in actions:
            curr_val = self.getQValue(state, action, context)
            if not max_val or max_val < curr_val:
                max_val = curr_val
                max_action = action
        return max_action

    # Compute the Value of a state according to qVals
    def computeValueFromQValues(self, state, context):
        actions = self.actions
        if not actions:
            return 0.0
        max_val = max([self.getQValue(state, action, context) for action in actions])
        return max_val

    # Update function to update weights
    def update(self, state, action, nextState, reward, context):
        next_val = self.computeValueFromQValues(nextState, context)
        qVal = self.getQValue(state, action, context)
        features = context.features.FeatureDict        
        value_difference = reward + self.discount * next_val - qVal
        for feature, feature_value in features.items():
            self.weights[feature] = self.weights[feature] + self.alpha * value_difference * features[feature]

    # Gets the Policy
    def getPolicy(self, state, context):
        return self.computeActionFromQValues(state, context)

    # Testing purposes        
    def getWeights(self):
        for key in self.weights:
            print key
            print self.weights[key]
            print "------------------"
            return 0

# Class for the Features
class Features:
    def __init__(self):
        self.FeatureDict = dict()

    # Define a feature if we are at an all-time-high price
    def all_time_high(self, today_opening, highest_price, state, action):
        pos, pc = state
        if today_opening >= highest_price:
            if action == "sell":
                result = abs(2 * pc)
            elif action == "buy":
                result = - abs(pc)
            else:
                result = pc
        else:
            result = 0
        self.FeatureDict["all_time_high"] = result

    # Define a feature if we are at an all-time-low price
    def all_time_low(self, today_opening, lowest_price, state, action):
        pos, pc = state
        if today_opening <= lowest_price:
            if action == "sell":
                result = -abs(4 * pc)
            elif action == "buy":
                result = abs(2 * pc)
            else:
                result = abs(pc)
        else:
            result = 0
        self.FeatureDict["all_time_low"] = result

    # Feature that rewards holding if opening price is higher than yesterday's
    def opening_above_hold(self, today_opening, yesterday_opening, state, action):
        pos, pc = state
        if today_opening >= yesterday_opening:
            if action == "sell":
                result = abs(pc)
            elif action == "buy":
                result = abs(pc)
            else:
                result = abs(2 * pc)
        else:
            result = 0
        self.FeatureDict["opening_above_hold"] = result

    # Feature that rewards holding if opening price is lower than yesterday's
    def opening_below_hold(self, today_opening, yesterday_opening, state, action):
        pos, pc = state
        if today_opening <= yesterday_opening:
            if action == "sell":
                result = abs(pc)
            elif action == "buy":
                result = abs(pc)
            else:
                result = abs(2 * pc)
        else:
            result = 0
        self.FeatureDict["opening_below_hold"] = result

    # Feature that rewards buying if opening price is higher than yesterday's
    def opening_above_buy(self, today_opening, yesterday_opening, state, action):
        pos, pc = state
        if today_opening >= yesterday_opening:
            if action == "sell":
                result = abs(pc)
            elif action == "buy":
                result = abs(2 * pc)
            else:
                result = abs(pc)
        else:
            result = 0
        self.FeatureDict["opening_above_buy"] = result

    # Feature that rewards buying if opening price is lower than yesterday's
    def opening_below_buy(self, today_opening, yesterday_opening, state, action):
        pos, pc = state
        if today_opening <= yesterday_opening:
            if action == "sell":
                result = abs(pc)
            elif action == "buy":
                result = abs(2 * pc)
            else:
                result = abs(pc)
        else:
            result = 0
        self.FeatureDict["opening_below_buy"] = result

    # Feature that rewards selling if opening price is higher than yesterday's
    def opening_above_sell(self, today_opening, yesterday_opening, state, action):
        pos, pc = state
        if today_opening >= yesterday_opening:
            if action == "sell":
                result = abs(2 * pc)
            elif action == "buy":
                result = abs(pc)
            else:
                result = abs(pc)
        else:
            result = 0
        self.FeatureDict["opening_above_sell"] = result

    # Feature that rewards selling if opening price is lower than yesterday's
    def opening_below_sell(self, today_opening, yesterday_opening, state, action):
        pos, pc = state
        if today_opening >= yesterday_opening:
            if action == "sell":
                result = abs(2 * pc)
            elif action == "buy":
                result = abs(pc)
            else:
                result = abs(pc)
        else:
            result = 0
        self.FeatureDict["opening_below_sell"] = result

    # Updating the dictionary of features; calls all the above functions
    def update_features_dict(self, state, action, today_opening, yesterday_opening, lowest_price, highest_price):
      self.opening_below_sell(today_opening, yesterday_opening, state, action)
      self.opening_above_sell(today_opening, yesterday_opening, state, action)
      self.opening_below_buy(today_opening, yesterday_opening, state, action)
      self.opening_above_buy(today_opening, yesterday_opening, state, action)
      self.opening_below_hold(today_opening, yesterday_opening, state, action)
      self.opening_above_hold(today_opening, yesterday_opening, state, action)
      self.all_time_low(today_opening, lowest_price, state, action)
      self.all_time_high(today_opening, highest_price, state, action)
      return self.FeatureDict




