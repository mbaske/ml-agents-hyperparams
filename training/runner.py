# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning.
# M.Baske, October 2018

import sys
import subprocess
import threading
import queue
import os
import logging
from multiprocessing import cpu_count


class Runner(object):
    def __init__(self, options):
        self.procs = []
        self.options = options
        self._set_workers()
        self.dir = os.path.dirname(os.path.realpath(__file__))
        self.config_path = self._create_config_path()

    @property
    def is_idle(self):
        return len(self.procs) is 0

    @property
    def available_workers(self):
        return self._workers - len(self.procs)

    def run(self, training_data):
        options = self.options.copy()
        training_data.save(self.config_path + options['--run-id'] + training_data.descr + '.json')
        if training_data.seed != -1:
            options['--seed'] = training_data.seed
        options['--train'] = True
        options['--sub-id'] = training_data.descr
        options['--worker-id'] = training_data.num
        options['--training-data-path'] = training_data.file
        args = [sys.executable, self.dir + '/learn.py', options['<env>']]
        del(options['<env>'])
        for k, v in options.items():
            if type(v) is bool:
                if v:
                    args.append(k)
            else:
                args.append(k + '=' + str(v))
        # Launch learn.py instance.
        self.procs.append(subprocess.Popen(args))

    def poll(self, event):
        q = queue.Queue()
        try:
            t = threading.Thread(target=self._poll, args=[q])
            t.start()
            t.join()
            # learn.py instance complete.
            index = q.get()
            args = self.procs[index].args
            del(self.procs[index])
            # Notify HyperTuner.
            event.dispatch(args)
        except KeyboardInterrupt:
            q.put(True)
            sys.exit(0)

    def _poll(self, q):
        index = -1
        while index is -1:
            try:
                q.get(timeout=1)
                break
            except queue.Empty:
                for i, p in enumerate(self.procs):
                    if p.poll() is not None:
                        index = i
                        q.put(index)
                        break
        return False

    def _create_config_path(self):
        try:
            if not os.path.exists('./config/'):
                os.makedirs('./config/')
        except Exception:
            raise Exception('The folder "config" could not be created.')
        return 'config/'

    def _set_workers(self):
        workers = int(self.options['--workers'])
        if workers is 0:
            try:
                self._workers = cpu_count()
            except NotImplementedError:
                self._workers = 1
        else:
            self._workers = workers
        del(self.options['--workers'])
