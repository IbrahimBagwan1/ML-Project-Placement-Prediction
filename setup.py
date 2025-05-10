from setuptools import find_packages,setup
from typing import List

Leave = "-e ."

def get_requirements(file_path:str)->List[str]:
  '''
  This function returns the list of Requirements.
  '''
  requirements = []
  with open(file_path) as file_obj:
    requirements = file_obj.readlines()
    requirements= [req.replace("\n","") for req in requirements]
    if Leave in requirements:
      requirements.remove(Leave)
  return requirements

setup(
  name = 'MlProject',
  version = '0.0.1',
  author = 'Ibrahim RasulAhmed Bagwan',
  author_email = 'irbagwan12@gmail.com',
  packages=find_packages(),
  install_requires = get_requirements("requirements.txt")

)