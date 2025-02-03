from setuptools import find_packages,setup
from typing import List


Hyphen_e_dot = '-e .'
def get_requirements(filename:str)->List[str]:
    '''will return list of requirements
    '''
    reuirements = []
    with open(filename) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if Hyphen_e_dot in reuirements:
            requirements.remove(Hyphen_e_dot)
    return reuirements
setup(
    name='ml_project',
    version = '3.12.7',
    author = 'Aditya',
    author_email='ag82620790@gmail.com' ,
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)