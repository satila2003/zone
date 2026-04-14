import sys
import platform


def get_python_env_info():
    env_info = {
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'compiler': platform.python_compiler(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
    }

    return env_info



info = get_python_env_info()

for key, value in info.items():
    print(f"{key}: {value}")

import torch
import sys
import numpy
print(numpy.__version__)
print(sys.version)
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
import sys
print(sys.executable)



