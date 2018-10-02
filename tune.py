# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning.
# M.Baske, October 2018

import sys
import logging
import math
from docopt import docopt
from decimal import Decimal
from trainers_mod.runner import Runner
from trainers_mod.training_data import TrainingData, StopCondition
from trainers_mod.training_event import TrainingEvent


class HyperTuner(object):
    """
    Creates TrainingData and serves it to Runner in a batchwise fashion.
    Optionally handles training results for creating subsequent batches.
    """

    def __init__(self, runner):
        self.runner = runner
        self.trainer_done_event = TrainingEvent(self.trainer_done_handler)
        self.logger = logging.getLogger('mlagents.trainers')
        self.training_data = []
        self.num_trainer = 0
        self.num_batch = 0

    def initialize(self):
        """
        Called at start up.
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

                self.training_data.append(TrainingData(descr, hyper, stop))

        self.run_trainer_batch()

    def trainer_result_handler(self, training_data):
        """
        Called after a training session has finished.
        Other sessions belonging to the same batch are either
        - already done
        - still running
        - launching as soon as Runner workers become available

        @type  training_data: TrainingData
        @param training_data: TrainingData
        """

        self.logger.info(self.format_msg('Completed {}'.format(training_data)))

    def batch_complete_handler(self):
        """
        Called after a batch of training sessions has finished.
        No more sessions are running at this point.

        @rtype:   bool
        @return:  Returns True if app should shut down, False otherwise
        """

        self.logger.info('Completed batch #{}'.format(self.num_batch))
        self.num_batch += 1

        # Batch demo, we just run the grid search 3 times.
        if self.num_batch < 3:
            self.grid_demo()
            return False
        else:
            self.logger.info('All training sessions completed.')
            return True

    def run_trainer_batch(self):
        """
        Must be invoked after trainer data was generated.
        """
        self.wait_for_runner_idle = False
        for i in range(min(self.runner.available_workers, len(self.training_data))):
            self.run_next_trainer()
        self.runner.poll(self.trainer_done_event)

    def run_next_trainer(self):
        data = self.training_data[self.num_trainer]
        data.num = self.num_trainer
        self.num_trainer += 1
        self.logger.info(self.format_msg('Starting {}'.format(data)))
        self.runner.run(data)

    def trainer_done_handler(self, args):
        file = [s for s in args if '--training-data-path' in s][0][21:]
        data = TrainingData.from_file(file)
        self.training_data[data.num] = data
        self.trainer_result_handler(data)

        if len(self.training_data) is self.num_trainer:
            self.wait_for_runner_idle = True
        else:
            self.run_next_trainer()

        if self.runner.is_idle:
            if self.wait_for_runner_idle:
                if self.batch_complete_handler():
                    self.logger.info('Shutting down.')
                    sys.exit(0)
            else:
                self.logger.warn('No training sessions running.')
        else:
            self.runner.poll(self.trainer_done_event)

    def format_msg(self, msg):
         s = '\n----------------------------------------------------------------'
         s += '\n' + msg
         s += '\n----------------------------------------------------------------'
         return s


if __name__ == '__main__':
    _USAGE = '''
    Usage:
      learn (<env>) [options]
      learn --help

    Options:
      --curriculum=<directory>      Curriculum json directory for environment [default: None].
      --keep-checkpoints=<n>        How many model checkpoints to keep [default: 5].
      --lesson=<n>                  Start learning from this lesson [default: 0].
      --load                        Whether to load the model or randomly initialize [default: False].
      --run-id=<path>               The directory name for model and summary statistics [default: ppo].
      --save-freq=<n>               Frequency at which to save model [default: 50000].
      --seed=<n>                    Random seed used for training [default: -1].
      --slow                        Whether to run the game at training speed [default: False].
      --train                       Whether to train model, or only run inference [default: False].
      --docker-target-name=<dt>     Docker volume to store training-specific files [default: None].
      --no-graphics                 Whether to run the environment in no-graphics mode [default: False].
      --workers=<n>                 Number of subprocesses / concurrent training sessions [default: 0].
      --trainer-config-path=<file>  Location of the trainer config yaml file [default: trainer_config.yaml].
    '''

    ht = HyperTuner(Runner(docopt(_USAGE)))
    ht.initialize()
