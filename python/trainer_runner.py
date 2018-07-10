# # Unity ML Agents
#
# Added to enable automated hyperparameter tuning.
# M.Baske, July 2018

import sys
import subprocess
import threading
import queue
import os
from multiprocessing import cpu_count
from unityagents import UnityException


class TrainerRunner(object):
    def __init__(self, options):
        self.procs = []
        self.options = options
        self._set_workers()
        self.config_path = self._create_config_path()

    @property
    def is_idle(self):
        return len(self.procs) is 0

    @property
    def available_workers(self):
        return self._workers - len(self.procs)

    def run(self, trainer_data):
        options = self.options.copy()
        # create file
        trainer_data.save(self.config_path + options['--run-id'] + trainer_data.descr + '.json')
        # set launch parameters for learn.py
        options['--train'] = True
        options['--run-id'] += trainer_data.descr
        options['--worker-id'] = trainer_data.num
        options['--config'] = trainer_data.file
        args = [sys.executable, 'learn.py', options['<env>']]
        del(options['<env>'])
        for k, v in options.items():
            if type(v) is bool:
                if v:
                    args.append(k)
            else:
                args.append(k + '=' + str(v))
        # launch learn.py instance
        self.procs.append(subprocess.Popen(args))

    def poll(self, trainer_done_event):
        q = queue.Queue()
        try:
            t = threading.Thread(target=self._poll, args=[q])
            t.start()
            t.join()
            # learn.py instance complete
            index = q.get()
            args = self.procs[index].args
            del(self.procs[index])
            # notify HyperTuner
            trainer_done_event.dispatch(args)
        except KeyboardInterrupt:
            q.put(True)
            self.logger.info('Shutting down')
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
        path = './' + self.options['--config'] + '/'
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception:
            raise UnityException('The folder {} could not be created.'.format(path))
        return path

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
