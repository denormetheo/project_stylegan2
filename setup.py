from setuptools.config import read_configuration
from setuptools import setup, find_packages

setup(
    name="lcc",
    version="1.0.0",
    author="Leila Berrada, Theo DENORME",
    author_email="denorme.theox@gmail.com, leila.berrada@student-cs.fr",
    packages=find_packages(),
    install_requires=['python_version >= "3.7.15"'],
)