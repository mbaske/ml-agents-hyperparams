

## Hyperparameter Tuning for [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) v0.3

### How to use
* Implement your optimization algorithm in [/python/unitytrainers/hypertuner.py](https://github.com/mbaske/ml-agents-hyperparams/blob/master/v0_3/python/unitytrainers/hypertuner.py)
* Run /python/tune.py

Options are identical to running learn.py with the exception of --worker-id which is not supported. Instead, you might pass --workers=\<n> for setting the number of concurrent training sessions. Each session runs in its own process with the default number of processes matching the amount of CPU cores. 

Hyperparameters still get loaded from trainer_config.yaml. However, they will be complemented or overridden by the ones defined in your code.

Tested with ml-agents v0.3 on macOS (Python 3.6).

### Example
This is a simple grid search demo. We use the tennis environment from ml-agents examples. Training data is being created beforehand and then served incrementally to the training sessions.

	def grid_demo(self):
        stop = [StopCondition('episode_length', '> 40')]
        beta = [1e-4, 1e-3, 1e-2]
        gamma = [0.8, 0.9, 0.995]
        for b in beta:
            for g in gamma:
                hyper = {'beta': b, 'gamma': g}
                descr = '_beta_{0}_gamma_{1}'.format('%.0E' % Decimal(b), g)
                self.training_data.append(TrainingData(hyper, descr, stop))
                
Start your training sessions and compare their performance in TensorBoard.

	python3 tune.py tennis --run-id=tennis --save-freq=25000
	tensorboard --logdir=summaries

<img src="images/tensorboard.png" align="middle" width="1440"/>

Please note that I removed the logging of cumulative rewards during training. With multiple processes running at once, that was somewhat confusing. Use TensorBoard to track the training progess instead.