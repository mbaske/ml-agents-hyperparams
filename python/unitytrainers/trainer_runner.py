# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning
# M.Baske, April 2018

import logging
import signal
from concurrent import futures
from multiprocessing import cpu_count
from .hypertuner import HyperTuner


class TrainerRunner(object):
    def __init__(self, trainer_controller, workers):
        global interrupted
        interrupted = False
        signal.signal(signal.SIGUSR1, self._on_interrupt)
        self.logger = logging.getLogger("hypertuner")
        self.tuner = HyperTuner(StartEvent(self._start))
        self.tc = trainer_controller
        if workers is 0:
            try:
                self.workers = cpu_count()
            except NotImplementedError:
                self.workers = 1
        self.logger.info('Initializing Process Pool - {0} workers'.format(self.workers))
        self.pool = futures.ProcessPoolExecutor(max_workers=self.workers)
        self.tuner.initialize()

    def _start(self):
        self.out_of_data = False
        for i in range(self.workers):
            if self.out_of_data is False:
                self.out_of_data = self._start_process(self.tuner.get_training_data())

    def _start_process(self, training_data):
        if training_data:
            self.logger.info('Starting training session #{0}'.format(training_data.uid))
            self.logger.info(training_data)
            process = self.pool.submit(self.tc.start_learning, training_data)
            process.arg = training_data.uid
            process.add_done_callback(self._done_callback)
            return False
        else:
            if self.tuner.batch_process:
                self.logger.info('No more training data available, waiting for batch to finish')
            else:
                self.logger.info('No more training data available, waiting for running sessions to finish')
            return True

    def _done_callback(self, process):
        if process.cancelled():
            self.logger.warning('Process {0} was cancelled'.format(process.arg))
        elif process.done():
            error = process.exception()
            if error:
                self.logger.error('Process {0} - {1} '.format(process.arg, error))
            else:
                self.tuner.result_handler(process.result())
                self.logger.debug('Process {0} done'.format(process.arg))
                if interrupted is False:
                    if self.out_of_data is False:
                        self.out_of_data = self._start_process(self.tuner.get_training_data())
                    else:
                        if self.tuner.batch_process:
                            self.tuner.batch_complete_handler()
                        else:
                            self.logger.info('All training sessions completed!')
                else:
                    self.logger.warning('Training was interrupted')

    def _on_interrupt(self, signum, frame):
        global interrupted
        interrupted = True


class StartEvent(object):
    def __init__(self, default_handler):
        self.handlers = [default_handler]

    def __exit__(self):
        self.handlers = []

    def add(self, handler):
        self.handlers.append(handler)
        return self

    def remove(self, handler):
        self.handlers.remove(handler)
        return self

    def dispatch(self):
        for handler in self.handlers:
            handler()
