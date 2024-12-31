from setuptools import setup, find_packages

setup(
    name='flowenv',
    version='0.0.1',
    packages=find_packages(include=["const", "flow_test", "flow_train"]),
)