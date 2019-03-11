from setuptools import setup
setup(
    name = 'texture_generation',
    version = '0.1.0',
    packages = ['gan', 'gatys'],
    entry_points = {
        'console_scripts': [
            'texture_gan = gan.__main__:main',
            'texture_gatys = gatys.__main__:main'
        ]
    })