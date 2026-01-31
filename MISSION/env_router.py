import_path_ref = {
    "dca": ("MISSION.dca.collective_assault_parallel_run", 'ScenarioConfig'),
}

env_init_function_ref = {
    "dca": ("MISSION.dca.collective_assault_parallel_run", 'make_collective_assault_env'),
}

##################################################################################################################################
##################################################################################################################################
from config import GlobalConfig
import importlib, os
from UTIL.colorful import print亮蓝



def load_ScenarioConfig():
    if GlobalConfig.env_name not in import_path_ref:
        assert False, ('need to find path of ScenarioConfig')
    import_path, ScenarioConfig = import_path_ref[GlobalConfig.env_name]
    GlobalConfig.ScenarioConfig = getattr(importlib.import_module(import_path), ScenarioConfig)


def make_env_function(env_name, rank):
    load_ScenarioConfig()
    ref_env_name = env_name

    if 'native_gym' in env_name:
        assert '->' in env_name
        ref_env_name, env_name = env_name.split('->')
    elif 'sr_tasks' in env_name:
        assert '->' in env_name
        ref_env_name, env_name = env_name.split('->')

    import_path, func_name = env_init_function_ref[ref_env_name]
    env_init_function = getattr(importlib.import_module(import_path), func_name)
    return lambda: env_init_function(env_name, rank)



def make_parallel_envs(process_pool, marker=''):
    from UTIL.shm_env import SuperpoolEnv
    from config import GlobalConfig
    from MISSION.env_router import load_ScenarioConfig
    load_ScenarioConfig()

    env_args_dict_list = [({
        'env_name':GlobalConfig.env_name,
        'proc_index':i if 'test' not in marker else -(i+1),
        'marker':marker
    },) for i in range(GlobalConfig.num_threads)]

    if GlobalConfig.env_name == 'dca':
        # This particular env has a cython file that needs to be compiled in main process
        # that must be loaded in main process
        from MISSION.dca.cython_func import laser_hit_improve3

    if GlobalConfig.num_threads > 1:
        envs = SuperpoolEnv(process_pool, env_args_dict_list)
    else:
        envs = SuperpoolEnv(process_pool, env_args_dict_list)

    return envs
