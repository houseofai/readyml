import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'readyml'
DESCRIPTION = 'ReadyML - Easy and Ready Machine Learning.'
URL = 'https://github.com/houseofai/readyml'
AUTHOR = 'Odyssée'
REQUIRES_PYTHON = '>=3.7'
VERSION = '0.0.3'

# What packages are required for this module to be executed?
REQUIRED = [
    'tensorflow-hub==0.12.0',
    'tensorflow==2.5.0',
    'pillow',
    'scipy',
    'imageio',
    'matplotlib',
    'opencv-python',
    'requests',
    # Models
    # Image Restoration
    'mirnet @ git+https://git@github.com/houseofai/MIRNet#egg=mirnet',
    # Face Detection
    'fastface',
    # Transformer (NMT): https://pytorch.org/hub/pytorch_fairseq_translation/
    'fastBPE', 'regex', 'sacremoses', 'subword_nmt', 'hydra-core', 'omegaconf'
]

################################################################################
here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        #os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))
        os.system('{0} setup.py sdist'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        #self.status('Pushing git tags…')
        #os.system('git tag v{0}'.format(about['__version__']))
        #os.system('git push --tags')

        sys.exit()

setup(
    name=NAME,
    version=about['__version__'],
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    license='LICENSE',
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=REQUIRED,

    include_package_data=True,
    packages=['readyml', 'readyml.utils', 'readyml.labels'],
    #packages=find_packages(NAME),

    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
