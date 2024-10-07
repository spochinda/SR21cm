from setuptools import setup, find_packages

setup(
    name='SR21cm',  # Change this to your package's name
    version='0.1.0',
    description='A package for 3D super-resolution of the 21cm differential brightness temperature',
    author='spochinda',
    author_email='sp2053@cam.ac.uk',
    url='https://github.com/spochinda/21cmGen',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires = ">=3.8",
    install_requires=open('requirements.txt').read().splitlines(),
)

