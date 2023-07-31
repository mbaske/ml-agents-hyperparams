from datetime import datetime
import json
import os
import platform
import requests
import subprocess
import sys
import time
from typing import Any, Dict, List, Union
import urllib.parse
import yaml

"""
Handles command line arguments.

:raises ValueError: if no config yaml path was specified
"""


class ArgParser():
    def __init__(self):
        args: List[str] = sys.argv
        del args[0]

        value: Any = self.get_value(args, '.yaml')
        if value:
            self.config_path: str = str(value)
        else:
            raise ValueError('No config yaml path specified.')

        value = self.get_value(args, 'run-id')
        self.run_id: str = str(value) if value else 'run'

        value = self.get_value(args, 'num-envs')
        self.num_envs: int = int(value) if value else 1

        value = self.get_value(args, 'base-port')
        self.base_port: int = int(value) if value else 5005

        self.env_args: List[str] = args

    """
    Returns argument value for a specified search string.

    :param List[str] args: arguments list
    :param str search: search string
    :return: argument value
    :rtype: Any
    """

    def get_value(self, args: List[str], search: str) -> Any:
        value: Any = None
        l: List[int] = [n for n, s in enumerate(args) if search in s]
        if l:
            arg: str = args[l[0]]
            value = arg.split('=')[1] if '=' in arg else arg
            del args[l[0]]
        return value

    """
    Returns a run ID with # suffix

    :param int n: run count
    :return: run ID
    :rtype: List[str]
    """

    def get_run_id(self, n: int) -> str:
        return f'{self.run_id}-{n}'

    """
    Returns the arguments for an mlagents-learn subprocess.

    :param int n: run count
    :param int i: slot index
    :param str config_path: config path for training run
    :return: arguments list
    :rtype: List[str]
    """

    def get_process_args(self, n: int, i: int, config_path: str) -> List[str]:
        args: List[str] = ['mlagents-learn', config_path, f'--run-id={self.get_run_id(n)}',
                           f'--base-port={self.base_port + i}']
        args.extend(self.env_args)
        return args

    def __str__(self) -> str:
        return f'config_path: {self.config_path}, num_envs: {str(self.num_envs)}, env_args: {", ".join(self.env_args)}'


"""
Util for transforming config keys.
"""


class KeyUtil():
    incr = 0
    join = '___'

    """
    Makes the key unique by appending a numerical value.

    :param str key: simple key
    :return: unique key
    :rtype: str
    """

    def unique(key: str) -> str:
        KeyUtil.incr += 1
        return key + KeyUtil.join + str(KeyUtil.incr)

    """
    Simplifies the key by removing the numerical value.

    :param str key: unique key
    :return: simple key
    :rtype: str
    """

    def simple(key: str) -> str:
        return key.split(KeyUtil.join)[0]


"""
Stores value options for config param.
"""


class ValueOption():
    """
    :param str key: name of config param
    :param List[Any] values: list of possible values
    """

    def __init__(self, key: str, values: List[Any]):
        self.key: str = key
        self.values: List[Any] = values
        log(f'Found config param option - {self}')

    def __str__(self) -> str:
        return f'{KeyUtil.simple(self.key)}: {", ".join(map(str, self.values))}'


"""
Stop condition for training runs.
A run can stop prematurely if the latest scalar value for a 
specified metric (tag) is outside of specified min/max limits.
"""


class StopCondition():
    tb_api = 'http://localhost:6006/data/plugin/scalars/scalars?'

    """
    :param Dict[str, Any] params: tag/step/min/max params
    """

    def __init__(self, params: Dict[str, Any]):
        assert 'tag' in params, 'No tag found in stop condition.'
        self.tag: str = params['tag']
        self.min: float = float(params['min'] if 'min' in params else -999999999)
        self.max: float = float(params['max'] if 'max' in params else 999999999)
        self.max = max(self.max, self.min)
        self.step: int = int(params['step'] if 'step' in params else 0)
        log(f'Found stop condition - {self}')

    """
    Calls TensorBoard HTTP API for specified run and checks whether 
    the latest scalar value for {tag} is outside of min/max limits.

    :param str run_id: run id
    :return: true if value is out of min/max limits
    :return: message if value is out of min/max limits 
    :rtype: bool
    """

    def evaluate(self, run_id: str) -> Union[bool, str]:
        args: Dict[str, str] = {'run': run_id, 'tag': self.tag}
        url: str = StopCondition.tb_api + urllib.parse.urlencode(args)
        try:
            r: Response = requests.get(url=url, verify=False, timeout=5)
            if r.status_code == requests.codes.ok:
                data: List[List[float]] = json.loads(r.text)
                step: int = data[-1][1]  # Latest step
                value: float = data[-1][2]  # Latest scalar
                if step >= self.step:
                    if value < self.min:
                        return True, f'{self.tag}: {value} < {self.min} [step: {step}]'
                    elif value > self.max:
                        return True, f'{self.tag}: {value} > {self.max} [step: {step}]'
            # else:
            # log(r.text)
            # No scalar data yet.
        except:
            log('Could not connect to TensorBoard.')

        return False, None

    def __eq__(self, other):
        return self.tag == other.tag

    def __str__(self) -> str:
        return f'tag: {self.tag}, step: {self.step}, min: {str(self.min)}, max: {str(self.max)}'


"""
Stores behavior specific info.
"""


class Behavior():
    """
    :param str name: behavior name
    :param str run_id: run id
    :param Dict[str, Any] config: behavior config settings
    :param Dict[str, Any] defaults: default settings if available
    """

    def __init__(self, name: str, run_id: str, config: Dict[str, Any], defaults: Dict[str, Any]):
        self.name: str = name
        # Objects generated from opt_values and opt_stop fields
        self.value_options: List[ValueOption] = []
        self.stop_conditions: List[StopCondition] = []

        # Run IDs for each value combination,
        # verbose run IDs contain behavior name: RunID-#/BehaviorName
        # that's how they are listed in TensorBoard
        self.verbose_run_ids: List[str] = []
        # Info strings for each value combination
        self.value_infos: List[List[str]] = []
        # Config settings for each value combination
        self.mod_configs: List[Dict[str, Any]] = []

        if defaults is not None:
            self.copy_defaults(config, defaults)
        # Make keys unique
        parsed: Dict[str, Any] = self.parse_config(self.unique_keys(config))

        if self.value_options:
            # List of param names we have optional values for
            param_names: List[str] = [x.key for x in self.value_options]
            # Create all possible combinations of optional values
            value_combos: List[List[Any]] = self.get_value_combinations()
            # Create specific config settings for each value combination
            for i, value_combo in enumerate(value_combos):
                mod_config: Dict[str, Any] = self.insert_values(parsed, param_names, value_combo)
                # Revert unique keys back to simple ones
                self.mod_configs.append(self.simple_keys(mod_config))

                self.verbose_run_ids.append(f'{run_id}-{str(i)}\{name}')
                # Info for value options:
                # RunID-#
                # - Behavior name
                #   - param1 name: param1 value
                #   - param2 name: param2 value
                # ...
                value_info: List[str] = [f'\n{run_id}-{str(i)}\n- {name}\n']
                for j, param_name in enumerate(param_names):
                    # param_names still has unique keys
                    key: str = KeyUtil.simple(param_name)
                    value_info.append(f'  - {key}: {str(value_combo[j])}\n')
                self.value_infos.append(value_info)
        else:
            # No value options, keep behavior as is
            self.mod_configs.append(self.simple_keys(parsed))
            self.verbose_run_ids.append(f'{run_id}-0\{name}')
            self.value_infos.append([f'- {name}\n', '  - no value options\n'])

    def __str__(self) -> str:
        return f'run_ids: {", ".join(self.run_ids)}'

    """
    Copies default settings into behavior config settings.
    The resulting run config won't contain a default_settings section.
    If there are opt_values set in the original defaults, then they will be 
    applied to each individual behavior. Note that this can blow up the total
    number of runs, because every behavior permutation is combined with every
    other behavior permutation.

    :param Dict[str, Any] config: config settings
    :param Dict[str, Any] defaults: default settings
    :rtype: None
    """

    def copy_defaults(self, config: Dict[str, Any], defaults: Dict[str, Any]) -> None:
        for k, v in defaults.items():
            if k not in config:
                config[k] = v
            if isinstance(v, dict):
                self.copy_defaults(config[k], v)

    """
    Parses the 'opt_' params in config yaml.
    Returns a config settings copy without those params.
    Generates ValueOption and StopCondition objects.

    :param Dict[str, Any] config: behavior config settings
    :param str key: config param name
    :return: updated config settings copy
    :rtype: Dict[str, Any]
    """

    def parse_config(self, config: Dict[str, Any], key: str = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in config.items():
            if 'opt_values' in k:
                self.value_options.append(ValueOption(key, v))
            elif 'opt_stop' in k:
                self.stop_conditions.append(StopCondition(v))
            if isinstance(v, dict):
                v = self.parse_config(v, k)
            if 'opt_' not in k:
                result[k] = v
        return result

    """
    Generates option value combinations.

    :param List[List[Any]] result: value combination list
    :param List[Any] tmp: temp. value list
    :param int i: value option index
    :return: list of all value combinations
    :rtype: List[List[Any]]
    """

    def get_value_combinations(self, result: List[List[Any]] = None, tmp: List[Any] = None, i: int = 0) -> List[
        List[Any]]:
        n: int = len(self.value_options)
        result: List[List[Any]] = [] if result is None else result
        tmp: List[Any] = [None] * n if tmp is None else tmp
        for v in self.value_options[i].values:
            tmp[i] = v  # Current value by option index
            if i is n - 1:
                result.append(tmp.copy())
            else:
                self.get_value_combinations(result, tmp, i + 1)
        return result

    """
    Inserts value combination in config settings.
    Returns a config settings copy with modified params.

    :param Dict[str, Any] config: config settings
    :param List[str] param_names: list of param names
    :param List[Any] values: list of possible values
    :return: updated config settings copy
    :rtype: Dict[str, Any]
    """

    def insert_values(self, config: Dict[str, Any], param_names: List[str], values: List[Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in config.items():
            if isinstance(v, dict):
                v = self.insert_values(v, param_names, values)
            result[k] = values[param_names.index(k)] if k in param_names else v
        return result

    """
    Returns a config settings copy with unique key names.

    :param Dict[str, Any] config: behavior config settings
    :param str key: config param name
    :param bool ignore: whether to ignore the current key
    :return: updated config settings copy
    :rtype: Dict[str, Any]
    """

    def unique_keys(self, config: Dict[str, Any], key: str = None, ignore: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in config.items():
            if isinstance(v, dict):
                v = self.unique_keys(v, k, ignore or 'opt_' in k)
            result[k if ignore or 'opt_' in k else KeyUtil.unique(k)] = v
        return result

    """
    Returns a config settings copy with simple key names.

    :param Dict[str, Any] config: behavior config settings
    :param str key: config param name
    :return: updated config settings copy
    :rtype: Dict[str, Any]
    """

    def simple_keys(self, config: Dict[str, Any], key: str = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in config.items():
            if isinstance(v, dict):
                v = self.simple_keys(v, k)
            result[KeyUtil.simple(k)] = v
        return result


"""
Loads, parses and saves config settings.
"""


class Config():
    """
    :param ArgParser args: ArgParser instance
    """

    def __init__(self, args: ArgParser):
        file_path: str = args.config_path
        name: str = os.path.basename(file_path).split('.')[0]
        dir: str = os.path.dirname(file_path)

        config: Dict[str, Any] = self.load_config(file_path)
        defaults: Dict[str, Any] = config['default_settings'] if 'default_settings' in config else None

        log(f'Parsing {name}...')
        self.behaviors: List[Behavior] = []
        self.stop_conditions: List[StopCondition] = []

        for k, v in config['behaviors'].items():
            b: Behavior = Behavior(k, args.run_id, v, defaults)
            self.behaviors.append(b)
            # Stop conditions are global in the sense that if they are set for
            # only one of multiple behaviors, the corresponding training run will
            # stop when they are met - regardless of the other behaviors' metrics.
            for cond in b.stop_conditions:
                if cond not in self.stop_conditions:
                    self.stop_conditions.append(cond)

        # Create all possible run ID combinations
        # Outer list: by combo, inner list: verbose run ids
        self.verbose_run_id_combos: List[List[str]] = self.get_run_id_combinations()
        self.num_runs: int = len(self.verbose_run_id_combos)

        # Gather config settings for each run id combination
        # Outer list: by combo, inner list: by run id, dict: config settings
        run_configs: List[List[Dict[str, Any]]] = []

        config_names: List[str] = []
        config_info: List[str] = []

        for i, run_id_combo in enumerate(self.verbose_run_id_combos):
            config_name: str = name + '-' + str(i)
            config_names.append(config_name)
            # Configs by run id
            configs: List[Dict[str, Any]] = []
            for run_id in run_id_combo:
                # Find matching behavior
                for behavior in self.behaviors:
                    if run_id in behavior.verbose_run_ids:
                        b: int = behavior.verbose_run_ids.index(run_id)
                        configs.append(behavior.mod_configs[b])
                        config_info.extend(behavior.value_infos[b])
                        break
            run_configs.append(configs)

        n: int = len(run_configs)
        assert n is self.num_runs, f'Wrong number of configs {n}, should be {self.num_runs}'
        self.save_info(config_info, dir)

        # Build and save combined config settings
        self.config_paths: List[str] = []
        for i in range(self.num_runs):
            save_config: Dict[str, Any] = {'behaviors': {}}
            for j, b in enumerate(self.behaviors):
                save_config['behaviors'][b.name] = run_configs[i][j]
            file_path: str = self.save_config(save_config, dir, config_names[i])
            self.config_paths.append(file_path)

        log(f'{self.num_runs} training runs queued. See config_info.txt for details.')

    """
    Generates run id combinations.

    :param List[List[str]] result: run id combination list
    :param List[str] tmp: temp. run id list
    :param int i: behavior index
    :return: list of all run id combinations
    :rytpe: List[List[str]]
    """

    def get_run_id_combinations(self, result: List[List[str]] = None, tmp: List[str] = None, i: int = 0) -> List[
        List[str]]:
        n: int = len(self.behaviors)
        result: List[List[str]] = [] if result is None else result
        tmp: List[str] = [None] * n if tmp is None else tmp
        for id in self.behaviors[i].verbose_run_ids:
            tmp[i] = id  # Current run id by behavior index
            if i is n - 1:
                result.append(tmp.copy())
            else:
                self.get_run_id_combinations(result, tmp, i + 1)
        return result

    """
    Loads config settings from yaml file.

    :param str path: path to config yaml file
    :return: config settings
    :rytpe: Dict[str, Any]
    """

    def load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            return config
        except FileNotFoundError:
            log(f'Could not load configuration from {path}.')

    """
    Saves config settings to yaml file.

    :param Dict[str, Any] config: config settings
    :param str dir: file directory
    :param str name: file name
    :return: path to config file
    :rytpe: str
    """

    def save_config(self, config: Dict[str, Any], dir: str, name: str) -> str:
        path: str = os.path.join(dir, name + ".yaml")
        try:
            with open(path, 'w') as f:
                try:
                    yaml.dump(config, f, sort_keys=False)
                except TypeError:  # Older versions of pyyaml don't support sort_keys
                    yaml.dump(dir, f)
            return path
        except FileNotFoundError:
            log(f'Could not save configuration to {path}.')

    """
    Saves config info to text file.

    :param List[str] info: lines of output text
    :param str dir: file directory
    :rytpe: None
    """

    def save_info(self, info: List[str], dir: str) -> None:
        path: str = os.path.join(dir, "config_info.txt")
        try:
            with open(path, "w") as f:
                for line in info:
                    f.write(line)
        except FileNotFoundError:
            log(f'Could not save configuration info to {path}.')

    def __str__(self):
        return self.name


"""
Handles training runs.
"""


class Runner():
    """
    :param ArgParser args: ArgParser instance
    """

    def __init__(self, args: ArgParser):
        self.args: ArgParser = args
        self.config: Config = Config(args)
        # Each slot can store a subprocess
        num_slots: int = args.num_envs
        self.slots: List[subprocess.Popen] = [None] * num_slots
        # Short run IDs don't contain behavior names, these
        # are the IDs passed as arguments to the subprocesses
        self.short_run_ids: List[str] = [None] * num_slots
        # Need to store verbose run IDs for the subprocesses too,
        # in order to evaluate stop conditions for each behavior
        self.verbose_run_ids: List[List[str]] = [None] * num_slots
        self.run_count: int = 0
        self.run_controller()

    """
    Starts training runs and checks for stop conditions.

    :param int interval: state check interval
    :rytpe: None
    """

    def run_controller(self, interval: int = 60) -> None:
        interrupt: bool = False

        while self.has_active_runs() or self.has_pending_runs():
            while True:
                i: int = self.get_free_slot()
                if i > -1 and self.has_pending_runs():
                    self.start_process(i)
                else:
                    break
            try:
                while True:
                    time.sleep(interval)
                    exit: bool = False
                    done: List[int] = []
                    for i, slot in enumerate(self.slots):
                        if slot:
                            id: str = self.short_run_ids[i]
                            log(f'Checking {id} progress...')
                            if slot.poll() is not None:
                                done.append(i)
                            elif self.must_stop(i):
                                self.stop_process(i)
                                exit = True

                    for i in done:
                        id: str = self.short_run_ids[i]
                        code: int = self.slots[i].returncode
                        if code is 0:
                            log(f'{id} complete.')
                        else:
                            log(f'An error occurred in {id}: {code}.')
                        self.slots[i].kill()  # TODO do we need this?
                        self.slots[i] = None
                        exit = True

                    if exit:
                        break

            except KeyboardInterrupt:
                interrupt = True
                for i, slot in enumerate(self.slots):
                    if slot:
                        self.stop_process(i)
                break

        if interrupt:
            log('Training was interrupted.')
        else:
            log('All training runs complete.')

    """
    Starts a training run / launches a subprocess.

    :param int i: slot index where the process will be stored
    :rytpe: None
    """

    def start_process(self, i: int) -> None:
        n: int = self.run_count
        self.run_count += 1

        args: List[str] = self.args.get_process_args(n, i, self.config.config_paths[n])
        if platform.system() == 'Windows':
            self.slots[i] = subprocess.Popen(args, creationflags=subprocess.CREATE_NEW_CONSOLE)
        elif platform.system() == 'Linux':
            self.slots[i] = subprocess.Popen('gnome-terminal -x ' + ' '.join(args), shell=True)

        run_id: str = self.args.get_run_id(n)
        self.short_run_ids[i] = run_id
        self.verbose_run_ids[i] = self.config.verbose_run_id_combos[n]
        log(f'{run_id} started.')

    """
    Stops specified process.

    :param int i: process slot index
    """

    def stop_process(self, i: int):
        # Windows only
        self.slots[i].terminate()
        self.slots[i] = None

    """
    Whether any stop condition was met for a specified process.

    :param int i: process slot index
    :return: true if process must stop.
    :rytpe: bool
    """

    def must_stop(self, i: int) -> bool:
        for cond in self.config.stop_conditions:
            for id in self.verbose_run_ids[i]:
                stop, reason = cond.evaluate(id)
                if stop:
                    log(f'Stopping {self.short_run_ids[i]} because {reason}')
                    return True
        return False

    """
    Whether there are any pending runs.

    :return: true if there are any pending runs
    :rytpe: bool
    """

    def has_pending_runs(self) -> bool:
        return self.run_count < self.config.num_runs

    """
    Whether there are any active runs.

    :return: true if there are any active runs
    :rytpe: bool
    """

    def has_active_runs(self) -> bool:
        for slot in self.slots:
            if slot is not None:
                return True
        return False

    """
    Returns next available process slot index or -1.

    :return: next available process slot index or -1
    :rytpe: int
    """

    def get_free_slot(self) -> int:
        for i, slot in enumerate(self.slots):
            if slot is None:
                return i
        return -1


def log(msg):
    now: datetime = datetime.now()
    current_time: str = now.strftime("%H:%M:%S")
    print(f'[{current_time}] {msg}')


def main():
    assert platform.system() == 'Windows' or platform.system() == 'Linux', 'Unsupported platform.'
    runner = Runner(ArgParser())


if __name__ == "__main__":
    main()