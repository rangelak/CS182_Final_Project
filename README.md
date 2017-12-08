# CS182 Final Project
The first step to clone our GitHub repository.

To be able to run our algorithms and use our system, please visit Quantopian's [homepage](https://www.quantopian.com/home) and create a free account. Then enter the platform's IDE, accessible [here](https://www.quantopian.com/algorithms/5a2ae310246a132c4d1d1996). To run our SARSA-based trading agent, copy-paste the code we have provided in `sarsa.py` into the IDE; to run our Tabular Q-learning trading agent, simply copy-paste the code we have provided in `tabularQ.py` into the IDE; to run our Approximate Q-learning trading agent, simply copy-paste the code provided in `approximateQ.py` into the IDE (in all three cases completely replace the default sample algorithm code found in the Quantopian IDE). 

Now, to see how either trading agent performs, run a full "backtest" through Quantopian by specifying the start and end dates of the period you wish to trade in, setting a reasonable amount of initial capital e.g. $10,000, and then clicking `Run Full Backtest` (note: our algorithms functions with different levels of initial capital, but we observed when trading Tesla shares that a fairly limited number are available, and the IDE couldn't satisfy all the orders, so, in order to avoid share orders not being filled we recommend $10,000). Since the agent is only trading every day, we recommend a more extensive period (i.e. 3-4 years). You can change the stock you are trading within code in the IDE (there should be a comment specifying which stock we are currently trading). To change it, just delete the current id, and start typing the stock symbol (the autocomplete on the IDE will give you the stock id). Once you have run the "full backtest" you should be able to see three graphs that reflect our performance on this iteration: cumulative performance, daily/weekly returns, and transactions.  

In the "Cumulative Performance" graph, Quantopian will report our trading algorithms' performances in historical simulation, and in particular against various benchmarks. We obtained the outputs presented in the **Experiments** section by repeatedly running full backtests, storing the resulting data, and subsequently charting our algorithms' performances based on different combinations of hyperparameters (i.e. learning rate *alpha*, discount factor *gamma*, etc.).