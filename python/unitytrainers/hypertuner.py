# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning
# M.Baske, April 2018

import logging
from decimal import Decimal
from .training_data import TrainingData, StopCondition


class HyperTuner(object):
    def __init__(self, start_event):
        """
        Implements the logic for creating and serving training data.

        @type  start_event: StartEvent
        @param start_event: StartEvent
        """
        self.logger = logging.getLogger("hypertuner")
        self.start_event = start_event
        self.training_data = []
        self.counter = 0
        self.batch = 0

    @property
    def batch_process(self):
        """
        Set True for batch processing.
        Not to be confused with hyperparameter "batch_size"

        @rtype:   bool
        @return:  batch processing enabled
        """
        return True

    def start_trainig(self):
        """
        Must be invoked after data was generated.
        """
        self.start_event.dispatch()

    def setup(self):
        """
        Called by TrainerRunnner.
        """
        # self.grid_demo()
        self.batch_demo()

    def batch_demo(self):
        """
        Demonstrates batch processing. We just run the same grid search 3 times.
        """
        self.logger.info('Creating training data for batch #{0}'.format(self.batch))
        self.grid_demo()
        self.logger.info('Starting batch #{0}'.format(self.batch))
        self.start_trainig()

    def grid_demo(self):
        """
        Simple grid search demo.
        """
        stop = [StopCondition('episode_length', '> 40')]
        beta = [1e-4, 1e-3, 1e-2]
        gamma = [0.8, 0.9, 0.995]
        for b in beta:
            for g in gamma:
                hyper = {'beta': b, 'gamma': g}
                # descr = '_beta_{0}_gamma_{1}'.format('%.0E' % Decimal(b), g)
                descr = '#{0}_beta_{1}_gamma_{2}'.format(self.batch, '%.0E' % Decimal(b), g)
                self.training_data.append(TrainingData(hyper, descr, stop))

        # self.start_trainig()

    def result_handler(self, training_data):
        """
        Called by TrainerRunnner after a single training session has fininshed.

        @type  training_data: TrainingData
        @param training_data: TrainingData
        """
        self.logger.info('Training session #{0} finished (exit {1})'.format(training_data.uid, training_data.exit_status))
        r = len(training_data.result)
        if r > 0:
            s = 'Last stats summary:'
            for key, value in training_data.result[r - 1].items():
                s += '\n\t{0}: \t{1}'.format(key, value)
            self.logger.info(s)

    def batch_complete_handler(self):
        """
        Called by TrainerRunnner after a batch of training sessions has finished.
        """
        self.logger.info('Batch #{0} completed'.format(self.batch))

        # batch demo continued
        if self.batch < 2:
            self.batch += 1
            self.batch_demo()
        else:
            self.logger.info('All training sessions completed!')

    def get_training_data(self):
        """
        Called by TrainerRunnner.
        Every new training session fetches 1 TrainingData object, adds training stats to
        its result array during training and passes it to result_handler after completion.

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
