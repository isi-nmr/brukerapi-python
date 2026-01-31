import json
import os
from pathlib import Path

import numpy as np
import pkg_resources

from brukerapi.folders import Folder

API_VERSION = pkg_resources.get_distribution("brukerapi").version
SUITES = ["test_parameters", "test_properties", "test_data", "test_mmap"]


def test_generator(path_folder, path_config, suites):
    if suites:
        if isinstance(suites, str):
            suites = [suites]
    else:
        suites = SUITES

    if "test_properties" in suites:
        properties = {}

    folder = Folder(path_folder)

    for dataset in folder.get_dataset_list_rec():
        with dataset(parameter_files=["subject"]) as d:
            print(f"Generating tests for {d.id}")
            if "test_parameters" in suites:
                generate_parameters_test(d)

            if "test_properties" in suites:
                properties[d.id] = generate_properties_test(d, path_folder)

            if "test_data" in suites:
                generate_data_test(d)

    if "test_properties" in suites:
        properties = dict(sorted(properties.items()))

        with open(path_config / ("properties_" + folder.path.name + ".json"), "w") as json_file:
            json.dump(properties, json_file, indent=4, sort_keys=True)


def generate_parameters_test(dataset):
    """
    Save each JCAMP-DX parameter file loaded within the dataset as JSON file to the same directory. These files are then used for testing consistency of the JCAMP-DX functionality.

    :param dataset: Instance of a Dataset class
    """
    for jcampdx in dataset._parameters.values():
        jcampdx.to_json(path=str(jcampdx.path) + ".json")


def generate_properties_test(dataset, abs_path):
    """

    :param dataset:
    :param abs_path:
    :return:
    """
    return dataset.to_dict()


def generate_data_test(dataset):
    """Generate configuration entry for a input/output functionality of an interface

    :param dataset:
    :return:
    """
    np.savez(dataset.path, data=dataset.data)


if __name__ == "__main__":
    # test_generator(Path(os.environ['PATH_DATA']) / '20201208_100554_lego_rod_1_2', Path(__file__).parent / 'config/',
    #                suites=['test_properties'])
    # test_generator(Path(os.environ['PATH_DATA']) / '20201208_105201_lego_rod_1_3', Path(__file__).parent / 'config/',
    #                suites=['test_properties'])
    # test_generator(Path(os.environ['PATH_DATA']) / '20201208_105201_lego_rod_1_4', Path(__file__).parent / 'config/',
    #                suites=['test_properties'])
    # test_generator(Path(os.environ['PATH_DATA']) / '20201208_105201_lego_rod_1_5', Path(__file__).parent / 'config/',
    #                suites=['test_properties'])
    # test_generator(Path(os.environ['PATH_DATA']) / '20201208_105201_lego_rod_1_6', Path(__file__).parent / 'config/',
    #                suites=['test_properties'])
    # test_generator(Path(os.environ['PATH_DATA']) / '20201208_105201_lego_rod_1_7', Path(__file__).parent / 'config/',
    #                suites=['test_properties'])
    test_generator(Path(os.environ["PATH_DATA"]) / "0.2H2", Path(__file__).parent / "config/")
    # test_generator(Path(os.environ['PATH_DATA']) / '20200612_094625_lego_phantom_3_1_2', Path(__file__).parent / 'config/')
    # test_generator(Path(os.environ['PATH_DATA']) / '20210128_122257_LEGO_PHANTOM_API_TEST_1_1',
    #                Path(__file__).parent / 'config/', suites=['test_parameters', 'test_data'])
    # test_generator(Path(os.environ['PATH_DATA']) / 'bruker2nifti_qa/raw', Path(__file__).parent / 'config/auto_test_qa.json')
