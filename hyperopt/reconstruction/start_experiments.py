from typing import Dict, List, TypedDict, Any
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import omegaconf
from argparse import ArgumentParser
import os
import pickle
from utils import load_nemo_jsons
import hyperopt
from objective_function import train_loss_objective
from functools import partial
import hyperopt

def _handle_hyperparams(hyperparam_config: omegaconf.dictconfig.DictConfig) -> Dict:
    """The hyperparams are written in the experiment yaml-file
    as tuples of arguments. e.g:

        hyperparams_search_spaces:
        'a' :
            func_name: hyperopt.hp.uniform
            args:
              - a
              - 0
              - 1
          'b' :
            func_name: hyperopt.hp.uniform
            args:
              - b
              - 0
              - 1

    This function parses the hyperopt parameters. For instance:
        a = hyperparam_config(a, 0, 1)
        ...

    Parameters
    ----------
    hyperparam_config : omegaconf.dictconfig.DictConfig
        hyperparam_config

    Returns
    -------
    Dict

    """
    for key in hyperparam_config.keys():
        assert not OmegaConf.is_missing(hyperparam_config, key), f'Missing value for hyperparam {key} in yaml'


    hyperparam_dict = {}

    for variable_name, variable_dict in hyperparam_config.items():

        function_name: str = variable_dict['func_name']
        func_args = variable_dict.get('args', [])
        func_args = list(func_args)

        func_kwargs = variable_dict.get('kwargs', {})
        func_kwargs = dict(func_kwargs)

        # The function is translated as string. We import the
        # library and search for each submodule
        module_path_elements = function_name.split('.')
        callable_function = __import__(module_path_elements.pop(0))
        for ele in module_path_elements:
            callable_function = getattr(callable_function, ele)

        hyperparam_dict[variable_name] = callable_function(*func_args, **func_kwargs)

    return hyperparam_dict

def _handle_models(model_dict: omegaconf.dictconfig.DictConfig) -> Dict[str, str]:
    """The models are written in a list in the experiment_yaml file.
    Here all models in the list are tested if they exist.

    Parameters
    ----------
    model_dict : omegaconf.dictconfig.DictConfig
        model_dict

    Returns
    -------
    Dict[str, str]

    """
    for key in model_dict.keys():
        assert not OmegaConf.is_missing(model_dict, key), f'Missing value for model {key} in yaml'


    new_dict = {}
    for model_name, model_path in model_dict.items():
        model_path = os.path.abspath(os.path.expanduser(model_path))
        assert os.path.isfile(model_path), f'{model_path} is not a valid model path'
        new_dict[model_name] = model_path

    return new_dict

def _handle_int(number: int, variable_name: str) -> int:
    assert isinstance(number, int) or number < 1,\
            f'The provided value of "{variable_name}" ({number}) is not valid integer'
    return number

def _handle_exp_dir(path: str) -> str:
    """The paths of the experiment_dirs are converted into absolute paths and
    the directories are created if they do no exist.

    Parameters
    ----------
    path : str
    path

    Returns
    -------
    str

    """
    path =  os.path.abspath(os.path.expanduser(path))

    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def _handle_json_file_paths(json_file_paths: List[str]) -> List[str]:
    """The json files are converted into absolute paths, checked for its
    existence and returned.

    Parameters
    ----------
    json_file_paths : List[str]
        json_file_paths

    Returns
    -------
    List[str]

    """
    new_list = []

    for json_file_path in json_file_paths:
        json_file_path = os.path.abspath(os.path.expanduser(json_file_path))
        assert os.path.isfile(json_file_path), f'{json_file_path} is not a valid path'
        new_list.append(json_file_path)
    return new_list

class ExperimentDict(TypedDict):
    """ExperimentDict.
    Attributes
        exp_dir: str
            A directory that shall contain all conducted experiments. In this
            case, for each model, there shall be subfolder that holds the
            conducted trials as logit. This logit will simply be used
            if we start the experiment again.
        models: Dict[str, str]
            The models item is a dictionary that holds string as values
            and keys in general the key represents a name chosen by
            conducter of the experiment, while the value has to be
            the path to a yaml file
        hyperparams_search_spaces: Dict[str, Any]
            A dictionary that contains the different hyperparameter-search
            spaces as values and the name of the hyperparameter as key.
            Those dictionaries are needed for running the
            `hyperopt.fmin` function
        amount_search_iterations: int
            provides information how many iterations the hyperopt parameter
            shall be searched.
        amount_train_epochs: int
            provides the number of epochs that each model shall train on
    """

    exp_dir: str
    # The models item is a dictionary that holds string as values
    # and keys in general the key represents a name chosen by
    # conducter of the experiment, while the value has to be
    # the path to a yaml file
    models: Dict[str, str]
    hyperparams_search_spaces: Dict
    amount_search_iterations: int
    amount_train_epochs: int

def load_experiment_dict_from_yaml(path: str) -> ExperimentDict:
    """Loads the experiment_yaml and handles the expected values.

    Parameters
    ----------
    path : str
        path

    Returns
    -------
    Dict

    """
    conf: omegaconf.dictconfig.DictConfig = OmegaConf.load(path)


    hyperparams = conf['hyperparams_search_spaces']
    hyperparams = _handle_hyperparams(hyperparams)

    models = conf['models']
    models = _handle_models(models)

    exp_dir = conf['exp_dir']
    exp_dir = _handle_exp_dir(exp_dir)

    amount_search_iterations = conf['amount_search_iterations']
    amount_search_iterations = _handle_int(amount_search_iterations, 'amount_search_iterations')

    amount_train_epochs = conf['amount_train_epochs']
    amount_train_epochs = _handle_int(amount_train_epochs, 'amount_train_epochs')

    return {
        'exp_dir': exp_dir,
        'models': models,
        'hyperparams_search_spaces': hyperparams,
        'amount_search_iterations' : amount_search_iterations,
        'amount_train_epochs' : amount_train_epochs
    }

def _execute_hyperopt_with_minibatches(number_of_evals,
                                       model_experiment_dir:str,
                                       model_name: str,
                                       model_config_path: str,
                                       amount_train_epochs: int,
                                       hyperparam_dict: Dict,
                                       minibatch_size=1,
                                      ):
    """_A wrapper function around the hyperopt.fmin function that
    runs on minibatches. After each minibatch the trials will be
    saved in a pickle and the progress of previous runs is saved.
    If we restart the function and there already is a 'trials.pickle'
    in the model_experiment_dir the trials will be loaded and
    the experiment continues from the last step.


    Parameters
    ----------
    number_of_evals :
        The number of evaluations of searching the hyperparameters
    model_experiment_dir : str
        The path to the directory of the respective model.
    json_files : List[str]
        The json-files containing information about the audio_filepath
        and the labels.
    model_config_path : str
        The path to the config-yaml
    hyperparam_dict : Dict
        The hyperparam_dict dict as explained in the ExperimentDict class.
    minibatch_size :
        For each model the hyperopt.trials will be saved as trials.pickle
    """

    models_path = os.path.join(model_experiment_dir, model_name)
    trials_path = os.path.join(model_experiment_dir, 'trials.p')


    if os.path.isfile(trials_path):
        trials = pickle.load(open(trials_path, 'rb'))
    else:
        trials = hyperopt.Trials()

    amount_of_exectuted_trials = len(trials.trials)

    # We have to do that because the hyperopt.fmin function
    # expects its functions to only use one single parameter
    objective = partial(train_loss_objective, model_config_path, amount_train_epochs, )

    while amount_of_exectuted_trials < number_of_evals:
        amount_of_exectuted_trials += minibatch_size
        best = hyperopt.fmin(objective,
                             hyperparam_dict,
                             algo=hyperopt.tpe.suggest,
                             max_evals=amount_of_exectuted_trials,
                             trials=trials)

        pickle.dump(trials, open(trials_path, 'wb'))
        print(best)

def run_experimtens(experiment_dict: ExperimentDict):
    experiment_dir = experiment_dict['exp_dir']
    iterations = experiment_dict['amount_search_iterations']
    amount_train_epochs = experiment_dict['amount_train_epochs']

    for model_name, model_path in experiment_dict['models'].items():
        model_dir_path = os.path.join(experiment_dir, model_name)
        if not os.path.isdir(model_dir_path):
            os.makedirs(model_dir_path)

        _execute_hyperopt_with_minibatches(
            number_of_evals=iterations,
            model_experiment_dir=model_dir_path,
            model_name=model_name,
            model_config_path=model_path,
            amount_train_epochs=amount_train_epochs,
            hyperparam_dict=experiment_dict['hyperparams_search_spaces']
        )

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('yaml_file_path', type=str, help='A file that is built like "./experiments_example.yaml"')

    args = argparser.parse_args()
    yaml_file_path = os.path.abspath(os.path.expanduser(args.yaml_file_path))
    experiment_dict = load_experiment_dict_from_yaml(yaml_file_path)
    run_experimtens(experiment_dict)
