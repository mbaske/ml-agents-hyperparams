# # Unity ML Agents
#
# Modified to enable automated hyperparameter tuning
# M.Baske, April 2018

import signal
import logging
import os
from docopt import docopt
from concurrent import futures
from multiprocessing import cpu_count
from unitytrainers.trainer_controller_mod import TrainerController
from unitytrainers.hypertuner import HyperTuner


def start_process(training_data):
    if training_data:
        logger.info('Starting training session #{0}'.format(training_data.uid))
        logger.info(training_data)
        process = ex.submit(tc.start_learning, training_data)
        process.arg = training_data.uid
        process.add_done_callback(done_callback)
        return False
    else:
        logger.info('No more training data available, waiting for running sessions to finish')
        return True


def done_callback(process):
    global interrupted, out_of_data
    if process.cancelled():
        logger.warning('Process {0} was cancelled'.format(process.arg))
    elif process.done():
        error = process.exception()
        if error:
            logger.error('Process {0} - {1} '.format(process.arg, error))
        else:
            ht.result_handler(process.result())
            logger.debug('Process {0} done'.format(process.arg))
            if interrupted is False:
                if out_of_data is False:
                    out_of_data = start_process(ht.get_training_data())
                else:
                    logger.info('All training sessions completed!')
            else:
                logger.warning('Training was interrupted')


def on_interrupt(signum, frame):
    global interrupted
    interrupted = True


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, on_interrupt)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("hypertuner")

    _USAGE = '''
    Usage:
      tune (<env>) [options]
      tune --help

    Options:
      --curriculum=<file>        Curriculum json file for environment [default: None].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: False].
      --run-id=<path>            The sub-directory name for model and summary statistics [default: ppo].
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --seed=<n>                 Random seed used for training [default: -1].
      --slow                     Whether to run the game at training speed [default: False].
      --train                    Whether to train model, or only run inference [default: False].
      --workers=<n>              Number of concurrent training sessions [default: 0].
      --docker-target-name=<dt>       Docker Volume to store curriculum, executable and model files [default: Empty].
    '''
    # --worker-id is not supported. Port numbers are assigned internally and correspond to sub-processes.

    options = docopt(_USAGE)
    logger.info(options)
    # Docker Parameters
    if options['--docker-target-name'] == 'Empty':
        docker_target_name = ''
    else:
        docker_target_name = options['--docker-target-name']

    # General parameters
    run_id = options['--run-id']
    seed = int(options['--seed'])
    load_model = options['--load']
    train_model = options['--train']
    save_freq = int(options['--save-freq'])
    env_path = options['<env>']
    keep_checkpoints = int(options['--keep-checkpoints'])
    workers = int(options['--workers'])
    curriculum_file = str(options['--curriculum'])
    if curriculum_file == "None":
        curriculum_file = None
    lesson = int(options['--lesson'])
    fast_simulation = not bool(options['--slow'])

    # Constants
    # Assumption that this yaml is present in same dir as this file
    base_path = os.path.dirname(__file__)
    TRAINER_CONFIG_PATH = os.path.abspath(os.path.join(base_path, "trainer_config.yaml"))

    tc = TrainerController(env_path, run_id, save_freq, curriculum_file, fast_simulation, load_model, train_model,
                           keep_checkpoints, lesson, seed, docker_target_name, TRAINER_CONFIG_PATH)
    if workers is 0:
        try:
            workers = cpu_count()
        except NotImplementedError:
            workers = 1
    logger.info('Starting - {0} workers'.format(workers))
    ht = HyperTuner()
    ex = futures.ProcessPoolExecutor(max_workers=workers)
    out_of_data = False
    interrupted = False
    for i in range(workers):
        if out_of_data is False:
            out_of_data = start_process(ht.get_training_data())
