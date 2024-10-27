import os 
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


cwd = os.path.dirname(os.path.abspath(__file__))

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

#class PostInstallCommand(install):
#    """Post-installation for installation mode."""
#    def run(self):
#        install.run(self)
#        os.system('python -m unidic download')

#class PostDevelopCommand(develop):
#    """Post-installation for development mode."""
#    def run(self):
#        develop.run(self)
#        os.system('python -m unidic download')

setup(
    name='melotts_onnx',
    version='0.0.2',
    author="Season Studio",
    author_email="season-studio@outlook.com",
    description="An implementation of melo tts by onnxruntime",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=reqs,
    package_data={
        '': ['*.txt', 'cmudict_*'],
    },
    exclude_package_data={
        '': ['models/*', 'venv/*']
    }
)