from setuptools import setup

setup(
    name='lgilearn',
    version='0.0.1',
    description='Predict the local gyrification index for a mesh vertex. The local gyrification index is a brain morphometry descriptor used in computational neuroimaging. It describes the folding of the human cortex at a specific point, based on a mesh reconstruction of the cortical surface from a magnetic resonance image.',
    url='https://github.com/dfsp-spirit/lgilearn',
    author='Tim Schaefer',
    author_email='ts+code@rcmd.org',
    license='MIT',
    packages=['lgilearn'],
    install_requires=['tensorflow>=2.0',
                      'numpy',
                      ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
