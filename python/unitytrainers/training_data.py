# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning
# M.Baske, April 2018


class TrainingData():
    # exit status flags
    DEFAULT = -1
    TRAINING_INTERRUPTED = -2

    def __init__(self, hyperparams, descr, stop_conditions=None, verbose_result=False):
        """
        :param hyperparams: dict - Will complement/override values loaded from trainer_config.yaml (Param names must be identical)
        :param descr: string - Will be added to the directory names of models and summaries (run_id + descr)
        :param stop_conditions: array of StopCondition objects, training stops if ANY of them evaluate True
        :param verbose_result: bool - True: store all training stats summaries / False: store last summary only
        """
        self._hyperparams = hyperparams
        self._descr = descr
        self._stop_conditions = stop_conditions
        self._verbose = verbose_result
        self._result = []
        self._exit_status = -1
        self._uid = ''

    @property
    def hyperparams(self):
        """
        :return: dict - Hyperparameters
        """
        return self._hyperparams

    @property
    def stop_conditions(self):
        """
        :return: array of StopCondition objects
        """
        return self._stop_conditions

    @property
    def result(self):
        """
        :return: array of dicts containing training stats
        """
        return self._result

    @property
    def exit_status(self):
        """
        :return: int - Exit status (default, interrupted or stop condition index)
        """
        return self._exit_status

    @property
    def descr(self):
        """
        :return: string - Suffix for model and summaries directory names (run_id + descr)
        """
        return self._descr

    @property
    def uid(self):
        """
        :return: int - Counter value set by HyperTuner
        """
        return self._uid

    @uid.setter
    def uid(self, uid):
        self._uid = uid

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
        self._exit_status = TrainingData.TRAINING_INTERRUPTED

    def __str__(self):
        s = 'Training data #{0} (exit {1})'.format(self.uid, self.exit_status)
        s += '\n\tHyperparameters:'
        for key, value in self.hyperparams.items():
            s += '\n\t{0}: \t{1}'.format(key, value)
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
