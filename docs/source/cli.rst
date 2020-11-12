Command line interface (CLI)
============================

**Together with the API a** `bruker` **command line tool is installed. It can be used with the following sub-commands:**

* `report`_
* `split`_
* `filter`_

It is also possible to create command line `pipelines`_ using our api

**Data used in examples on this page are freely available at Zenodo:**

- https://doi.org/10.5281/zenodo.4048253

It is possible to run examples in this section by setting an environment variable `DATA_PATH` to contain a path to the downloaded dataset::

    export DATA_PATH={path_to_20200612_094625_lego_phantom_3_1_2}

report
------

For each data set contained in a folder specified by the `-i` argument, the report sub-command saves properties individual data sets into a JSON, or a YAML file located in the dataset folder, or a folder, or file specified using the `-o` argument. It is also possible to only export properties defined by the `-p` argument.

Save properties of all data sets contained within the `20200612_094625_lego_phantom_3_1_2 <https://doi.org/10.5281/zenodo.4048253>`_ folder::

    bruker report -i 20200612_094625_lego_phantom_3_1_2/3/pdata/1/2dseq

The following is the content of the resulting `report.json` file located in the same folder as the `2dseq` file:

::

    {
        "TE": 4,
        "dim_type": [
            "spatial",
            "spatial",
            "<FG_SLICE>"
        ],
        "encoded_dim": 2,
        "shape_final": [
            256,
            256,
            3
        ],
        "is_single_slice": false,
        "shape_fg": [
            3
        ],
        "numpy_dtype": "int16",
        "shape_block": [
            256,
            256
        ],
        "pv_version": "6.0.1",
        "date": "<class 'datetime.datetime'>",
        "shape_frames": [
            3
        ],
        "offset": [
            0,
            0,
            0
        ],
        "TR": 100,
        "slope": [
            0.00176479209428005,
            0.00176479209428005,
            0.00176479209428005
        ],
        "shape_storage": [
            256,
            256,
            3
        ],
        "num_slice_packages": 3
    }

**The reporting sub-command can be used in one of five possible cases:**

- **folder in-place**
- **folder to folder**
- **dataset in-place**
- **dataset to file**
- **dataset to folder**

Each of these cases is described in the examples section.

**usage**::

    bruker report [-h] -i INPUT [-f {json,yml}] [-p PROPS [PROPS ...]]

**arguments:**
  * **-h, --help** show help message and exit
  * **-i --input** path to a Bruker data set, or a folder containing Bruker data sets
  * **-o --output** path to a folder, or a file to report to
  * **-f, --format** format of report files, one of {json,yml}, default value is **json**
  * **-p, --props** list of properties to export, if undefined, all properties are exported

**examples:**

The following list of scenarios is avaliable for the report subcommand:

- **Folder in-place** – for every dataset within the folder (recursively) save its report file to the same folder where the dataset is located. It is possible to choose format using the `-f` argument.::

    bruker report -i ${DATA_PATH}/20200612_094625_lego_phantom_3_1_2/ -f yml

- **Folder to folder** – for every dataset within the folder (recursively) save all report files to a folder specified by the `-o` argument. It is possible to choose format using the `-f` argument.::

    bruker report -i ${DATA_PATH}/20200612_094625_lego_phantom_3_1_2/ -o ${DATA_PATH}/tmp/ -f yml

- **Dataset in-place** – for a dataset within the folder (recursively) save all report files to a folder specified by the `-o` argument. It is possible to choose format using the `-f` argument.::

    bruker report -i ${DATA_PATH}/20200612_094625_lego_phantom_3_1_2/3/pdata/1/2dseq  -f yml

- **Dataset to file** – for every dataset within the folder (recursively) save all report files to a folder specified by the `-o` argument. It is possible to choose format using the `-f` argument.::

    bruker report -i 20200612_094625_lego_phantom_3_1_2/3/pdata/1/2dseq -o ${DATA_PATH}/tmp/report.json

- **Dataset to folder** – for every dataset within the folder (recursively) save all report files to a folder specified by the `-o` argument. It is possible to choose format using the `-f` argument.::

    bruker report -i 20200612_094625_lego_phantom_3_1_2/ -o ${DATA_PATH}/tmp  -f yml

- Say we are only interested in `TE` and `TR` properties and we want to specify name of the report file, to achieve this, we can use the `-p` argument.::

    bruker report -i 20200612_094625_lego_phantom_3_1_2/3/pdata/1/2dseq -p TE TR

split
-----

Usage::

    bruker report [-h] -i INPUT [-f {json,yml}] [-p PROPS [PROPS ...]]

**arguments:**
  * **-h, --help** show help message and exit
  * **-i --input** path to a Bruker data set, or a folder containing Bruker data sets
  * **-f, --format** format of report files, one of {json,yml}
  * **-p, --props** list of properties to export, if undefined, all properties are exported

Split by **slice package**::

    bruker split -i 20200612_094625_lego_phantom_3_1_2/43/pdata/2/2dseq -s


Split by **`FG_ISA`**::

    bruker split -i 20200612_094625_lego_phantom_3_1_2/43/pdata/2/2dseq -f FG_ISA

Split by **`FG_ECHO`**::

    bruker split -i 20200612_094625_lego_phantom_3_1_2/43/pdata/2/2dseq -f FG_ECHO

filter
------

The `filter` sub-command provides an option to make various queries on folders containing Bruker data. It is possible to list all data sets measured with the same pulse sequence, data sets measured during the last month,etc.


**List all data sets measured using the EPI pulse sequence**::

    bruker filter -i ${DATA_PATH}/20200612_094625_lego_phantom_3_1_2 -q "#PULPROG=='<EPI.ppg>'"


pipelines
---------

It is possible to assemble pipelines using the Bruker API and xargs. Let us see some examples:

Using filter and report subcommands to only report datasets measured by the MGE sequence::

    bruker filter -i /home/tomas/data/20200612_094625_lego_phantom_3_1_2/ -q "#PULPROG=='<MGE.ppg>'" | xargs -I {} bruker report -i {} -o /home/tomas/data/reports
