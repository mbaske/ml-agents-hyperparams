# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning.
# M.Baske, July 2018

from docopt import docopt
from trainer_runner import TrainerRunner
from hyper_tuner import HyperTuner


if __name__ == '__main__':
    _USAGE = '''
    Usage:
      learn (<env>) [options]
      learn --help

    Options:
      --curriculum=<file>        Curriculum json file for environment [default: None].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: False].
      --run-id=<path>            The sub-directory name for model and summary statistics [default: ppo].
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --seed=<n>                 Random seed used for training [default: -1].
      --slow                     Whether to run the game at training speed [default: False].
      --docker-target-name=<dt>  Docker Volume to store curriculum, executable and model files [default: Empty].
      --no-graphics              Whether to run the Unity simulator in no-graphics mode [default: False].
      --workers=<n>              Number of subprocesses / concurrent trainers [default: 0].
      --config=<path>            The sub-directory name for config json files [default: config].
    '''

    ht = HyperTuner(TrainerRunner(docopt(_USAGE)))
    ht.initialize()
