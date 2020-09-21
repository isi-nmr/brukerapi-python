from setuptools import setup

setup(name='brukerapi',
      version='0.1.1.0',
      description='Bruker API',
      author='Tomas Psorn',
      author_email='tomaspsorn@isibrno.cz',
      url='https://github.com/isi-nmr/brukerapi-python',
      download_url='https://github.com/isi-nmr/brukerapi-python/archive/v0.1.0.tar.gz',
      packages=['brukerapi',],
      install_requires=['numpy','pyyaml'],
      include_package_data=True,
      license='MIT',
      zip_safe=False
     )

