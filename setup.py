from setuptools import setup, find_packages
import versioneer


with open('requirements.txt') as f:
    requirements = f.read().strip().splitlines()


setup(
    packages=find_packages(),
    install_requires=requirements,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)
