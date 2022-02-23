from setuptools import setup

setup(
    name='meshlearn',
    version='0.0.1',
    description='Predict vertex-wise mesh descriptors. E.g., predict the local gyrification index for a mesh vertex. The local gyrification index is a brain morphometry descriptor used in computational neuroimaging. It describes the folding of the human cortex at a specific point, based on a mesh reconstruction of the cortical surface from a magnetic resonance image.',
    url='https://github.com/dfsp-spirit/meshlearn',
    author='Tim Schaefer',
    author_email='ts+code@rcmd.org',
    license='MIT',
    packages=['meshlearn'],
    install_requires=['tensorflow>=2.0',
                      'tensorflow_datasets',
                      'numpy',
                      'matplotlib',
                      'nibabel',
                      'trimesh',
                      'brainload',
                      'sklearn'
                      ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pytest-console-scripts', 'pytest-runner', 'tox'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
    package_dir = {'': 'src'},
    entry_points = {
        'console_scripts': [
            'meshlearn_lgi = meshlearn.clients.meshlearn_lgi:main',
        ]
    }
)
