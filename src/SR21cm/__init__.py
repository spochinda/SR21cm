from importlib.metadata import version
import importlib
import os

try: 
    __version__ = version("SR21cm")
except:
    pass

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# List all Python files in the directory
modules = [f[:-3] for f in os.listdir(current_dir) if f.endswith('.py') and f != '__init__.py']
#exclude old GAN modules
modules = [module for module in modules if 'wgan' not in module and 'utils_GAN' not in module]

for module in modules:
    #print(f'Importing module: {module}')
    importlib.import_module(f'.{module}', package=__name__)

__all__ = modules