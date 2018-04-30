# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning
# M.Baske, April 2018

import logging
from decimal import Decimal
from .training_data import TrainingData, StopCondition


class HyperTuner():
    def __init__(self):
        self.logger = logging.getLogger("hypertuner")
        self.counter = 0
        self.training_data = []
        self._setup()

    def _setup(self):
        """
        Simple grid search demo.
        """
        stop = [StopCondition('episode_length', '> 40')]
        beta = [1e-4, 1e-3, 1e-2]
        gamma = [0.8, 0.9, 0.995]
        for b in beta:
            for g in gamma:
                hyper = {'beta': b, 'gamma': g}
                descr = '_beta_{0}_gamma_{1}'.format('%.0E' % Decimal(b), g)
                self.training_data.append(TrainingData(hyper, descr, stop))

    def result_handler(self, training_data):
        """
        Handles the training result
        Could be used for generating new training data based on results from previous sessions,
        e.g. bayesian optimization

        @type  training_data: TrainingData
        @param training_data: TrainingData
        """
        self.logger.info("Training session #{0} finished (exit {1})".format(training_data.uid, training_data.exit_status))
        r = len(training_data.result)
        if r > 0:
            s = 'Last stats summary:'
            for key, value in training_data.result[r - 1].items():
                s += '\n\t{0}: \t{1}'.format(key, value)
            self.logger.info(s)

    def get_training_data(self):
        """
        Called by tune.py - Every new training session fetches 1 TrainingData object,
        adds training stats to its result array during training and returns it to result_handler after completion.

        @rtype:   TrainingData
        @return:  TrainingData
        """
        if len(self.training_data) is self.counter:
            return None
        else:
            data = self.training_data[self.counter]
            data.uid = self.counter
            self.counter += 1
            return data
