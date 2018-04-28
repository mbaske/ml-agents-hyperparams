# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning
# M.Baske, April 2018

import logging
from decimal import Decimal
from .training_data import TrainingData
from .training_data import StopCondition


class HyperTuner():
    def __init__(self):
        self.logger = logging.getLogger("hypertuner")
        self.counter = 0
        self.stack = []
        self._setup()

    def _setup(self):
        """
        Simple grid search demo. Create some training data and serve via get_next()
        """
        stop = [StopCondition('episode_length', '> 40')]
        beta = [1e-4, 1e-3, 1e-2]
        gamma = [0.8, 0.9, 0.995]
        for b in beta:
            for g in gamma:
                hyper = {'beta': b, 'gamma': g}
                descr = '_beta_{0}_gamma_{1}'.format('%.0E' % Decimal(b), g)
                self.stack.append(TrainingData(hyper, descr, stop))

    def result_handler(self, training_data):
        """
        Could be used for generating new training data based on results from previous sessions
        e.g. bayesian optimization
        """
        self.logger.info("Training session #{0} finished (exit {1})".format(training_data.uid, training_data.exit_status))
        r = len(training_data.result)
        if r > 0:
            s = 'Last stat summary:'
            for key, value in training_data.result[r - 1].items():
                s += '\n\t{0}: \t{1}'.format(key, value)
            self.logger.info(s)

    def get_training_data(self):
        """
        Called by tune.py - Every new training session fetches 1 TrainingData object,
        adds stat results to it during training and returns it to result_handler after completion
        """
        if len(self.stack) is self.counter:
            return None
        else:
            data = self.stack[self.counter]
            data.uid = self.counter
            self.counter += 1
            return data
