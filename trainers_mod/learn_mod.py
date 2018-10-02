# # Unity ML-Agents Toolkit
#
# Modified to enable automated hyperparameter tuning.
# M.Baske, October 2018

import logging
import numpy as np
from docopt import docopt

from trainer_controller_mod import TrainerControllerMod
from mlagents.trainers.exception import TrainerError


def run_training(run_seed, run_options):
    """
    Launches training session.
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    # Docker Parameters
    docker_target_name = (run_options['--docker-target-name']
        if run_options['--docker-target-name'] != 'None' else None)

    # General parameters
    env_path = run_options['<env>']
    run_id = run_options['--run-id']
    sub_id = options['--sub-id']
    load_model = run_options['--load']
    train_model = run_options['--train']
    save_freq = int(run_options['--save-freq'])
    keep_checkpoints = int(run_options['--keep-checkpoints'])
    worker_id = int(run_options['--worker-id'])
    curriculum_file = (run_options['--curriculum']
        if run_options['--curriculum'] != 'None' else None)
    lesson = int(run_options['--lesson'])
    fast_simulation = not bool(run_options['--slow'])
    no_graphics = run_options['--no-graphics']
    trainer_config_path = run_options['--trainer-config-path']
    training_data_path = (run_options['--training-data-path']
        if run_options['--training-data-path'] != 'None' else None)
    
    # Create controller and begin training.
    tc = TrainerControllerMod(env_path, run_id + sub_id,
                           save_freq, curriculum_file, fast_simulation,
                           load_model, train_model, worker_id,
                           keep_checkpoints, lesson, run_seed,
                           docker_target_name, trainer_config_path, 
                           no_graphics, training_data_path)
    tc.start_learning()


if __name__ == '__main__':
    try:
        print('''
    
                        ▄▄▄▓▓▓▓
                   ╓▓▓▓▓▓▓█▓▓▓▓▓
              ,▄▄▄m▀▀▀'  ,▓▓▓▀▓▓▄                           ▓▓▓  ▓▓▌
            ▄▓▓▓▀'      ▄▓▓▀  ▓▓▓      ▄▄     ▄▄ ,▄▄ ▄▄▄▄   ,▄▄ ▄▓▓▌▄ ▄▄▄    ,▄▄
          ▄▓▓▓▀        ▄▓▓▀   ▐▓▓▌     ▓▓▌   ▐▓▓ ▐▓▓▓▀▀▀▓▓▌ ▓▓▓ ▀▓▓▌▀ ^▓▓▌  ╒▓▓▌
        ▄▓▓▓▓▓▄▄▄▄▄▄▄▄▓▓▓      ▓▀      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌   ▐▓▓▄ ▓▓▌
        ▀▓▓▓▓▀▀▀▀▀▀▀▀▀▀▓▓▄     ▓▓      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌    ▐▓▓▐▓▓
          ^█▓▓▓        ▀▓▓▄   ▐▓▓▌     ▓▓▓▓▄▓▓▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▓▄    ▓▓▓▓`
            '▀▓▓▓▄      ^▓▓▓  ▓▓▓       └▀▀▀▀ ▀▀ ^▀▀    `▀▀ `▀▀   '▀▀    ▐▓▓▌
               ▀▀▀▀▓▄▄▄   ▓▓▓▓▓▓,                                      ▓▓▓▓▀
                   `▀█▓▓▓▓▓▓▓▓▓▌
                        ¬`▀▀▀█▓

        ''')
    except:
        print('\n\n\tUnity Technologies\n')

    logger = logging.getLogger('mlagents.trainers')
    _USAGE = '''
    Usage:
      learn <env> [options]
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
      --worker-id=<n>               Number to add to communication port (5005) [default: 0].
      --docker-target-name=<dt>     Docker volume to store training-specific files [default: None].
      --no-graphics                 Whether to run the environment in no-graphics mode [default: False].
      --sub-id=<path>               Unique id for training session [default: None].
      --training-data-path=<file>   Location of training data json file [default: None].
      --trainer-config-path=<file>  Location of the trainer config yaml file [default: trainer_config.yaml].
    '''

    options = docopt(_USAGE)
    logger.info(options)
    run_seed = int(options['--seed'])
    if run_seed == -1:
        run_seed = np.random.randint(0, 10000)
    run_training(run_seed, options)
