from setuptools import setup

setup(
    name='lgilearn',
    version='0.0.1',
    description='Predict local gyrification index for mesh vertex',
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
