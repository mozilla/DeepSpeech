from tensorflow.python.client import device_lib


def get_available_gpus(config):
    r"""
    Returns the number of GPUs available on this system.
    """
    local_device_protos = device_lib.list_local_devices(session_config=config)
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
