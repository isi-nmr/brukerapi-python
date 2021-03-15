from setuptools import setup

setup(name='brukerapi',
      version='0.1.4.0',
      description='Bruker API',
      author='Tomas Psorn',
      author_email='tomaspsorn@isibrno.cz',
      url='https://github.com/isi-nmr/brukerapi-python',
      download_url='https://github.com/isi-nmr/brukerapi-python/releases/latest',
      packages=['brukerapi',],
      install_requires=['numpy','pyyaml'],
      entry_points={
            "console_scripts": [
                  "bruker=brukerapi.cli:main"
            ],
      },
      include_package_data=True,
      license='MIT',
      zip_safe=False
     )

