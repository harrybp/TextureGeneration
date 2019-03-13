'''
Setup defines the application, requirements and entry points into the CLI
'''
from setuptools import setup

setup(
    name='texture_generation',
    version='0.1.0',
    packages=['gan', 'gatys'],
    test_suite="tests",
    entry_points={
        'console_scripts': [
            'texture_gan = gan.__main__:main',
            'texture_gatys = gatys.__main__:main',
            'demo_gan = gan.demo'
        ]
    },
    install_requires=[
        'numpy',
        'Pillow',
        'six'
    ])
