from setuptools import setup

setup(
    name='pytorch-acdc',
    version='0.0.1',
    packages=['pytorch_acdc'],
    platforms='any',
    install_requires=['torch>=0.4.1'],
    url='https://github.com/gngdb/pytorch-acdc',
    license='MIT',
    author='Gavin Gray',
    author_email='g.d.b.gray@ed.ac.uk',
    description='ACDC linear and convolutional layers.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
