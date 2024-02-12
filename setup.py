from setuptools import setup, find_packages

# Function to read requirements from requirements.txt
def read_requirements(file_path):
    with open(file_path, 'r') as f:
        requirements = f.readlines()
    # Remove whitespace and comments
    requirements = [req.strip() for req in requirements if not req.startswith('#')]
    return requirements

requirements = read_requirements( 'requirements.txt')
setup(
    name='newspapers',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    author='Pietro Piccini',
    author_email='pietro.piccini@hotmail.com',
    description='newspaper project',
    url='',
)
