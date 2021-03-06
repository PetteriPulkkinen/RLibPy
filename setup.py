from setuptools import setup
from setuptools import find_namespace_packages

setup(name='RLibPy',
      version='0.1',
      description='Reinforcement learning algorithms for OpenAI gym environments.',
      url='https://github.com/PetteriPulkkinen/RLibPy.git',
      author='Petteri Pulkkinen',
      author_email='petteri.pulkkinen@aalto.fi',
      licence='MIT',
      packages=find_namespace_packages(),
      install_requires=[
            'numpy', 'gym'
      ],
      zip_safe=False)
