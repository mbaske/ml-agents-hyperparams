# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning.
# M.Baske, July 2018
import json


class TrainerData(object):
    # exit status flags
    DEFAULT = -1
    TRAINING_INTERRUPTED = -2

    @classmethod
    def from_file(cls, file):
        self = cls()
        if file:
            self.load(file)
        return self

    def __init__(self, descr='', hyperparams=None, stop_conditions=None, verbose_result=False):
        """
        Data for running a trainer.

        @type  descr: string
        @param descr: Will be added to the directory names of models and summaries (run_id + descr).
        @type  hyperparams: dict
        @param hyperparams: Hyperparameters to complement/override the values loaded from trainer_config.yaml
                            Parameter names must be identical.
        @type  stop_conditions: array
        @param stop_conditions: StopCondition objects, trainer stops if ANY of them evaluate True.
        @type  verbose_result: bool
        @param verbose_result: True: store all trainer stats summaries / False: store last summary only.
        """
        self._num = 0
        self._descr = descr
        self._hyperparams = hyperparams
        self._stop_conditions = stop_conditions
        self._result = []
        self._verbose = verbose_result
        self._exit_status = TrainerData.DEFAULT
        self._file = None

    @property
    def file(self):
        """
        Returns fully qualified path to location of the config file.

        @rtype:   string
        @return:  File path + name
        """
        return self._file

    @property
    def hyperparams(self):
        """
        Returns the hyperparameters for the trainer.

        @rtype:   dict
        @return:  Hyperparameters
        """
        return self._hyperparams

    @property
    def stop_conditions(self):
        """
        Returns the stop condition(s) for the trainer.

        @rtype:   array
        @return:  StopCondition objects
        """
        return self._stop_conditions

    @property
    def result(self):
        """
        Returns the trainer's stats summaries.

        @rtype:   array
        @return:  Dict objects containing trainer stats
        """
        return self._result

    @property
    def exit_status(self):
        """
        Returns the trainer's exit status.

        @rtype:   int
        @return:  Exit status (default, interrupted or stop condition index)
        """
        return self._exit_status

    @property
    def descr(self):
        """
        Returns the trainer's short description.

        @rtype:   string
        @return:  Suffix for model and summaries directory names
        """
        return self._descr

    @property
    def num(self):
        """
        Returns the unique consecutive trainer number.

        @rtype:   int
        @return:  Counter value set by HyperTuner
        """
        return self._num

    @num.setter
    def num(self, num):
        self._num = num

    def load(self, file=None):
        if file:
            self._file = file
        with open(self._file, 'r') as f:
            self.__dict__ = json.load(f)
        self._convert_stop_conditions(True)
        return self._file

    def save(self, file=None):
        if file:
            self._file = file
        if self._file:
            self._convert_stop_conditions(False)
            with open(self._file, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
        # else: ignore, learn.py was launched without --config
        return self._file

    def add_summary(self, summary):
        if self._verbose:
            self._result.append(summary)
        else:
            self._result = [summary]
        return self.stop_condition_met()

    def stop_condition_met(self):
        if self.stop_conditions:
            summary = self.result[len(self.result) - 1]
            for i, cond in enumerate(self.stop_conditions):
                if cond.evaluate(summary):
                    self._exit_status = i
                    return cond
        return None

    def flag_interrupted(self):
        self._exit_status = TrainerData.TRAINING_INTERRUPTED

    def _convert_stop_conditions(self, instantiate):
        if self.stop_conditions:
            d = []
            for sc in self._stop_conditions:
                if instantiate:
                    d.append(StopCondition(sc['prop'], sc['cond']))
                else:
                    d.append({'prop': sc.prop, 'cond': sc.cond})
            self._stop_conditions = d

    def _print_dict(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                self.print_dict(v)
            else:
                print("{0} : {1} {2}".format(k, v, type(v)))

    def __str__(self):
        s = 'Trainer #{0} (exit {1})'.format(self.num, self.exit_status)
        s += '\n\tHyperparameters:'
        if self.hyperparams:
            for key, value in self.hyperparams.items():
                s += '\n\t{0}: \t{1}'.format(key, value)
        if self.stop_conditions:
            for sc in self.stop_conditions:
                s += '\n\tStop if {0}'.format(sc)
        r = len(self.result)
        if r > 0:
            s += '\n\tResult:'
            for key, value in self.result[r - 1].items():
                s += '\n\t{0}: \t{1}'.format(key, value)
        return s


class StopCondition():
    def __init__(self, prop, cond):
        self.prop = prop
        self.cond = cond

    def evaluate(self, summary):
        if self.prop in summary:
            return eval("summary['" + self.prop + "'] " + self.cond)
        return False

    def __str__(self):
        return self.prop + ' ' + self.cond
