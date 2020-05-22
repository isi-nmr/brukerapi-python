[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3831320.svg)](https://doi.org/10.5281/zenodo.3831320)

# brukerapi-python
A Python package providing I/O interface for Bruker data sets

**Features**
- read/write any fid
- read/write 2dseq
- read/write ser
- read/write rawdata
- Split
- Filter

**Examples**
- [Read 2dseq file](examples/read_2dseq.ipynb)
- [Read fid file](examples/read_fid.ipynb)
- [Split slice packages](examples/split_sp_demo.ipynb)
- [Split FG_ECHO](examples/split_fg_echo_demo.ipynb)
- [Split FG_ISA](examples/split_fg_isa_demo.ipynb)

**Resources:**

- [documentation](https://bruker-api.readthedocs.io/en/latest/)


Install using pip
--------------------------------------------

    pip install brukerapi
   
  
Install from source
--------------------------------------------

    git clone https://github.com/isi-nmr/brukerapi-python.git
    cd brukerapi-python
    python setup.py build
    python setup.py install
    
Getting started
--------------------------------------------

Work with **2dseq** data set:

~~~~{.python}
from brukerapi.dataset import Dataset
d = Dataset('path/2dseq')                       #create data set
d.data                                          #get data
d.get_value('VisuCoreSize')                     #get parameter
~~~~

Load **fid** data set:
~~~~{.python}
from brukerapi.dataset import Dataset
dataset = Dataset('path/fid')                   #create data set
dataset.data                                    #get data
dataset.get_value('ACQ_size')                   #get parameter
~~~~

Load **study**:
~~~~{.python}
from brukerapi.study import Study
study = Study('path')                           #create study
dataset = study.get_dataset(scan_id='1')        #get data set
dataset.load()                                  #load data
dataset.get_value('ACQ_size')                   #get parameter
~~~~

Load single **jcampdx** file:
~~~~{.python}
from brukerapi.jcampdx import JCAMPDX
params = JCAMPDX('path/method')
params.get_value('PVM_NAverages')               #get value

~~~~

