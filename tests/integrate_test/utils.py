import json
import sys
import pathlib
from pathlib import Path

def load_test_ref_result(test_func_name):
    """
    Read reference results from json file.
    """
    json_path = Path(__file__).parent / "data" / "ref_results.json"

    try:
        with open(json_path, "r", encoding='UTF-8') as f:
            all_ref_results = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Reference data file {json_path} does not exist")
    except json.JSONDecodeError as e:
        raise ValueError(f"Format of json file {json_path}is wrong, error: {e}")
    
    if test_func_name not in all_ref_results.keys():
        raise KeyError(f"Reference result for {test_func_name} not found")
    else:
        ref_results = all_ref_results[test_func_name]['result']

    return ref_results

def initilize_test_env():
    from abacusagent.env import set_envs
    set_envs()

def get_path_type():
    if sys.platform.startswith('linux'):
        return pathlib.PosixPath
    elif sys.platform.startswith('win'):
        return pathlib.WindowsPath
    else:
        raise ValueError(f"Platform {sys.platform} not supported!")
