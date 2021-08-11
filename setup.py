from distutils.core import setup

setup(
    name='RunML',
    version='0.0.1',
    author='Odyssée',
    packages=['runml'],
    url='https://github.com/houseofai/runml',
    license='LICENSE',
    description='Useful towel-related stuff.',
    long_description=open('README.md').read(),
    install_requires=[
        'tensorflow-hub==0.12.0',
        'tensorflow==2.5.0',
    ],
)
