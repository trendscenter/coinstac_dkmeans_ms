import os
import sys
import numpy as np

CONFIG_FILE = 'config.cfg'
DEFAULT_data_file = 'data.txt'
DEFAULT_k = 5
DEFAULT_shuffle = True
DEFAULT_learning_rate = 0.001
DEFAULT_optimization = 'lloyd'


def local_init_env(config_file=CONFIG_FILE, k=DEFAULT_k, optimization=DEFAULT_optimization, shuffle=DEFAULT_shuffle,
                   data_file=DEFAULT_data_file, learning_rate=DEFAULT_learning_rate, **kwargs):
    """

    """
    logging.info('LOCAL: Initializing remote environment')
    if not os.path.exists(config_file):
        config = configparser.ConfigParser()
        config['LOCAL'] = dict(k=k, optimization=optimization, shuffle=shuffle, data_file=data_file,
                               learning_rate=learning_rate)
        with open(config_path, 'w') as file:
            config.write(file)
    # output
    computation_output = dict(
        output=dict(
            config_file=config_file,
            computation_phase="local_init_env"
            ),
        success=True
    )
    return json.dumps(computation_output)


def local_init_centroids(config_file=CONFIG_FILE, **kwargs):
    logging.info('LOCAL: Initializing centroids')
    config = configparser.ConfigParser()
    config.read(config_file)
    data = np.loadtxt(config['LOCAL']['data_file'])
    centroids = local.initialize_own_centroids(data, config['LOCAL']['k'])
    # output
    computation_output = dict(
        output=dict(
            config_file=config_file,
            centroids=centroids,
            computation_phase="local_init_env"
            ),
        success=True
    )
    return json.dumps(computation_output)


def local_compute_clustering(config_file=CONFIG_FILE, remote_centroids=None, computation_phase="", **kwargs):
    logging.info('LOCAL: Initializing centroids')
    config = configparser.ConfigParser()
    config.read(config_file)
    data = np.loadtxt(config['LOCAL']['data_file'])

    cluster_labels = local.compute_clustering(data, remote_centroids)

    new_comp_phase = "local_compute_clustering"
    if computation_phase == "remote_optimization_step":
        new_comp_phase = "local_compute_clustering_2"
    computation_output = dict(
        output=dict(
            computation_phase=new_comp_phase,
            cluster_labels=cluster_labels,
            remote_centroids=remote_centroids,
            ),
        success=True
    )
    return json.dumps(computation_output)


def local_compute_optimizer(config_file=CONFIG_FILE, remote_centroids=None, cluster_labels=None, **kwargs):
    logging.info('LOCAL: Initializing centroids')
    config = configparser.ConfigParser()
    config.read(config_file)
    data = np.loadtxt(config['LOCAL']['data_file'])
    k = config['LOCAL']['k']
    learning_rate = config['LOCAL']['learning_rate']
    optimization = config['LOCAL']['optimization']
    if optimization == 'lloyd':
        local_optimizer = local.compute_mean(data, cluster_labels, k)
    elif optimization == 'gradient':
        # Gradient descent has sites compute gradients locally
        local_optimizer = \
            local.compute_gradient(node, cluster_labels[i],
                                   remote_centroids, learning_rate)
    computation_output = dict(
        output=dict(
            local_optimizer=local_optimizer,
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
