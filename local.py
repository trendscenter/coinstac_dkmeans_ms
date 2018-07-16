import os
import sys
import numpy as np

DEFAULT_work_dir = './.coinstac_tmp/'
DEFAULT_config_file = 'config.cfg'
DEFAULT_data_file = 'data.txt'
DEFAULT_k = 5


def local_init_env(work_dir=DEFAULT_work_dir, config_file=DEFAULT_config_file,
                   k=DEFAULT_k, optimization=DEFAULT_optimization, shuffle=DEFAULT_shuffle,
                   data_file=DEFAULT_data_file):
    """

    """
    logging.info('LOCAL: Initializing remote environment')
    # initialize environment
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    # create parameter / config file if it doesn't exist
    config_path = os.path.join(work_dir, config_file)
    if not os.path.exists(config_path):
        config = configparser.ConfigParser()
        config['LOCAL'] = dict(k=k, optimization=optimization, shuffle=shuffle, data_file=data_file)
        with open(config_path, 'w') as file:
            config.write(file)
    # output
    computation_output = dict(
        output=dict(
            work_dir=work_dir,
            config_file=config_file,
            computation_phase="local_init_env"
            ),
        success=True
    )
    return json.dumps(computation_output)


def local_init_centroids(work_dir=DEFAULT_work_dir, config_file=DEFAULT_config_file):
    logging.info('LOCAL: Initializing centroids')
    config_path = os.path.join(work_dir, config_file)
    config = configparser.ConfigParser()
    config.read(config_path)
    data = np.loadtxt(config['LOCAL']['data_file'])
    centroids = local.initialize_own_centroids(data, config['LOCAL']['k'])
    # output
    computation_output = dict(
        output=dict(
            work_dir=work_dir,
            config_file=config_file,
            centroids=centroids,
            computation_phase="local_init_env"
            ),
        success=True
    )
    return json.dumps(computation_output)


def local_compute_clustering(remote_centroids=None):
    # Returned after remote centroid initialization
    computation_output = dict(
        output=dict(
            computation_phase="local_compute_clustering"
            ),
        success=True
    )
    # Returned after remote optimization step
    computation_output = dict(
        output=dict(
            computation_phase="local_compute_clustering_2"
            ),
        success=True
    )
    return json.dumps(computation_output)


def local_compute_optimizer():
    computation_output = dict(
        output=dict(
            computation_phase="remote_aggregate_output"
            ),
        success=True
    )
    return json.dumps(computation_output)



if __name__ == '__main__':

    parsed_args = json.loads(sys.stdin.read())
    phase_key = list(list_recursive(parsed_args, 'computation_phase'))

    if "remote_init_env" in phase_key:
        computation_output = local_init_env(**parsed_args['input'])
        sys.stdout.write(computation_output)
    elif "local_init_env" in phase_key:
        computation_output = local_init_centroids(**parsed_args['input'])
        sys.stdout.write(computation_output)
    elif "remote_init_centroids" in phase_key:
        computation_output = local_compute_clustering(**parsed_args['input'])
        sys.stdout.write(computation_output)
    elif "local_compute_clustering" in phase_key:
        computation_output = local_compute_optimizer(**parsed_args['input'])
        sys.stdout.write(computation_output)
    elif "remote_optimization_step" in phase_key:
        computation_output = local_compute_clustering(**parsed_args['input'])
        sys.stdout.write(computation_output)
    elif 'remote_converged_false' in phase_key:
        computation_output = local_compute_optimizer(**parsed_args['input'])
        sys.stdout.write(computation_output)
    else:
        raise ValueError('Phase error occurred')
