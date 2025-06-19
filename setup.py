from setuptools import setup

setup(name='oneshot',
      version='0.0.1',
      author='Axel BREUER',
      description='A one-shot inpainting algorithm based on the topological asymptotic analysis',
      packages=['oneshot'],
      python_requires='>=3.9',
      setup_requires=["setuptools>=17.1"],
      install_requires=[
        "fipy"
      ],
      )
