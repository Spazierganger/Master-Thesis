from setuptools import find_packages, setup

install_requires = [
    'numpy',
    'seml',
    'scipy',
    'tqdm',
    'pyscf<2.0',
    'h5py==3.1.0',
    'optax',
    'sympy',
    'matplotlib',
    'jaxboard',
    'uncertainties'
]


setup(name='pesnet',
      version='0.1.0',
      description='Wave Function Neural Network',
      packages=find_packages('.'),
      install_requires=install_requires,
      zip_safe=False)
