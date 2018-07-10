# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning.
# M.Baske, July 2018

import sys
import logging
import math
from decimal import Decimal
from unitytrainers.trainer_data import TrainerData, StopCondition
from unitytrainers.trainer_event import TrainerEvent


class HyperTuner(object):
    """
    Creates TrainerData and serves it to TrainerRunner in a batchwise fashion.
    Optionally handles training results for creating subsequent batches.
    """

    def __init__(self, runner):
        self.runner = runner
        self.trainer_done_event = TrainerEvent(self.trainer_done_handler)
        self.logger = logging.getLogger("unityagents")
        self.trainer_data = []
        self.num_trainer = 0
        self.num_batch = 0

    def initialize(self):
        """
        Called by tune.py at start up.
        """
        self.grid_demo()

    def grid_demo(self):
        """
        Simple grid search demo.
        """
        stop = [StopCondition('episode_length', '> 40')]

        beta = [1e-4, 1e-3, 1e-2]
        gamma = [0.8, 0.9, 0.995]
        hidden_units = int(math.pow(2, 6 + self.num_batch))

        for b in beta:
            for g in gamma:
                hyper = {'beta': b, 'gamma': g, 'hidden_units': hidden_units}
                descr = '_units_{0}_beta_{1}_gamma_{2}'.format(hidden_units, '%.0E' % Decimal(b), g)

                self.trainer_data.append(TrainerData(descr, hyper, stop))

        self.run_trainer_batch()

    def trainer_result_handler(self, trainer_data):
        """
        Called after a trainer has finished.
        Other trainers belonging to the same batch are either
        - already done
        - still running
        - launching as soon as TrainerRunner workers are available

        @type  trainer_data: TrainerData
        @param trainer_data: TrainerData
        """

        self.logger.info('Trainer #{0} finished (exit {1})'.format(trainer_data.num, trainer_data.exit_status))
        r = len(trainer_data.result)
        if r > 0:
            s = 'Last stats summary:'
            for key, value in trainer_data.result[r - 1].items():
                s += '\n\t{0}: \t{1}'.format(key, value)
            self.logger.info(s)

    def batch_complete_handler(self):
        """
        Called after a batch of trainers has finished.
        No more trainers are running at this point.

        @rtype:   bool
        @return:  Returns True if app should shut down, False otherwise
        """
        self.logger.info('Batch #{0} completed'.format(self.num_batch))
        self.num_batch += 1

        # batch demo, we just run the grid search 3 times
        if self.num_batch < 3:
            self.grid_demo()
            return False
        else:
            self.logger.info('All trainers finished')
            return True

    def run_trainer_batch(self):
        """
        Must be invoked after trainer data was generated.
        """
        self.wait_for_runner_idle = False
        for i in range(min(self.runner.available_workers, len(self.trainer_data))):
            self.run_next_trainer()
        self.runner.poll(self.trainer_done_event)

    def run_next_trainer(self):
        data = self.trainer_data[self.num_trainer]
        data.num = self.num_trainer
        self.num_trainer += 1
        self.logger.info('Starting {0}'.format(data))
        self.runner.run(data)

    def trainer_done_handler(self, args):
        file = [s for s in args if '--config' in s][0][9:]
        data = TrainerData.from_file(file)
        self.trainer_data[data.num] = data
        self.trainer_result_handler(data)

        if len(self.trainer_data) is self.num_trainer:
            self.wait_for_runner_idle = True
        else:
            self.run_next_trainer()

        if self.runner.is_idle:
            if self.wait_for_runner_idle:
                if self.batch_complete_handler():
                    self.logger.info('Shutting down')
                    sys.exit(0)
            else:
                self.logger.warn("No trainers running")  # shouldn't occur
        else:
            self.runner.poll(self.trainer_done_event)
