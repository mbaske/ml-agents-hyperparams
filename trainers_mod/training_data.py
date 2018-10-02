# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning.
# M.Baske, October 2018

import json


class TrainingData(object):
    # Exit status flags.
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
        Data for running a training session.

        @type  descr: string
        @param descr: Will be added to the directory names of models and summaries (run_id + descr).
        @type  hyperparams: dict
        @param hyperparams: Hyperparameters to complement/override values loaded from trainer_config.yaml
                            Parameter names must be identical.
        @type  stop_conditions: array
        @param stop_conditions: StopCondition objects, training stops if ANY of them evaluate True.
        @type  verbose_result: bool
        @param verbose_result: True: store all training stats summaries / False: store last summary only.
        """
        self._num = 0
        self._seed = -1
        self._descr = descr
        self._hyperparams = hyperparams
        self._stop_conditions = stop_conditions
        self._result = {}
        self._verbose = verbose_result
        self._exit_status = TrainingData.DEFAULT
        self._file = None
        self._stop_condition_met = None

    @property
    def file(self):
        """
        Returns fully qualified path to location of the json config file.

        @rtype:   string
        @return:  File path + name
        """
        return self._file

    @property
    def hyperparams(self):
        """
        Returns the hyperparameters for the training session.

        @rtype:   dict
        @return:  Hyperparameters
        """
        return self._hyperparams

    @property
    def hyperparams_str(self):
        """
        Returns a string representation of the above hyperparameters.

        @rtype:   string
        @return:  Hyperparameters
        """
        s = 'Hyperparameters'
        if self.hyperparams:
            for k, v in self.hyperparams.items():
                s += '\n- {0}: \t{1}'.format(k, v)
        else:
            s += ' - N/A'
        return s

    @property
    def stop_conditions(self):
        """
        Returns the stop conditions for the training session.

        @rtype:   array
        @return:  StopCondition objects
        """
        return self._stop_conditions

    @property
    def stop_conditions_str(self):
        """
        Returns a string representation of the above stop conditions.

        @rtype:   string
        @return:  Stop Conditions
        """
        s = 'Stop Conditions'
        if self.stop_conditions:
            for sc in self.stop_conditions:
                s += '\n- {0}'.format(sc)
        else:
            s += ' - N/A'
        return s

    @property
    def result(self):
        """
        Returns the training session's stats summaries.

        @rtype:   object
        @return:  Dict objects containing training stats
        """
        return self._result

    @property
    def result_str(self):
        """
        Returns a string representation of the above stats.

        @rtype:   string
        @return:  Result/Stats
        """
        s = 'Result/Stats'
        if self.result:
            for brain, result in self.result.items():
                s += '\n- ' + brain
                for summary in result:
                    for k, v in summary.items():
                        s += '\n\t- {0}: \t{1}'.format(k, v)
        else:
            s += ' - N/A'
        return s

    @property
    def exit_status(self):
        """
        Returns the training session's exit status.

        @rtype:   int
        @return:  Exit status (default, interrupted or stop condition index)
        """
        return self._exit_status

    @property
    def descr(self):
        """
        Returns the training session's short description.

        @rtype:   string
        @return:  Suffix for model and summaries directory names
        """
        return self._descr

    @property
    def num(self):
        """
        Returns the unique consecutive training session number.

        @rtype:   int
        @return:  Counter value set by HyperTuner
        """
        return self._num

    @num.setter
    def num(self, num):
        self._num = num

    @property
    def seed(self):
        """
        Returns the random seed used for training.

        @rtype:   int
        @return:  Seed value
        """
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed

    def flag_interrupted(self):
        self._exit_status = TrainingData.TRAINING_INTERRUPTED

    def load(self, file=None):
        self._file = file or self._file
        if self._file :
            with open(self._file, 'r') as f:
                self.__dict__ = json.load(f)
            self._convert_stop_conditions(True)

    def save(self, file=None):
        self._file = file or self._file
        if self._file:
            self._convert_stop_conditions(False)
            with open(self._file, 'w') as f:
                json.dump(self.__dict__, f, indent=4)

    def add_summary(self, summary):
        brain = summary['brain_name']
        if self._verbose:
            if brain not in self._result:
                self._result[brain] = [summary]
            else:
                self._result[brain].append(summary)
        else:
            self._result[brain] = [summary]
        self._stop_condition_met = self.eval_stop_conditions(summary)
        return self._stop_condition_met

    def eval_stop_conditions(self, summary):
        if self.stop_conditions:
            for i, cond in enumerate(self.stop_conditions):
                if cond.evaluate(summary):
                    self._exit_status = i
                    return summary['brain_name'] + ' ' + str(cond)
        return None

    def _convert_stop_conditions(self, instantiate):
        if self.stop_conditions:
            d = []
            for sc in self._stop_conditions:
                if instantiate:
                    d.append(StopCondition(sc['prop'], sc['cond']))
                else:
                    d.append({'prop': sc.prop, 'cond': sc.cond})
            self._stop_conditions = d


    def __str__(self):
        s = 'Training Session #{0} (exit {1})'.format(self.num, self.exit_status)
        if self._file:
            s += '\n' + self._file
        if self._stop_condition_met:
            s += '\n' + self._stop_condition_met  
        s += '\n\n' + self.hyperparams_str
        s += '\n\n' + self.stop_conditions_str
        s += '\n\n' + self.result_str
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
