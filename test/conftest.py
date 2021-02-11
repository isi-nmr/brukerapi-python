import pytest
from pathlib import Path
import json
from brukerapi.folders import Folder

def pytest_addoption(parser):
    parser.addoption("--test_data", action="store", default="")
    parser.addoption("--test_suites", action="store", default="")
    parser.addoption("--properties_reference", action="store", default="")
    # parser.addoption("--properties_reference", action="store", default="")


def pytest_generate_tests(metafunc):

    ids, suites, data = get_test_data(metafunc)
    if 'test_parameters' in metafunc.fixturenames and 'test_parameters' in suites:
        metafunc.parametrize('test_parameters', data, indirect=True, ids=ids)
    elif 'test_properties' in metafunc.fixturenames and 'test_properties' in suites:
        metafunc.parametrize('test_properties', data, indirect=True, ids=ids)
    elif 'test_data' in metafunc.fixturenames and 'test_data' in suites:
        metafunc.parametrize('test_data', data, indirect=True, ids=ids)

def get_test_data(metafunc):
    suites = metafunc.config.option.test_suites.split(" ")
    study_id = Path(metafunc.config.option.test_data).name
    ids = []
    data = []

    # if properties test configuration exists
    if metafunc.config.option.properties_reference and Path(metafunc.config.option.properties_reference).exists():
        with Path(metafunc.config.option.properties_reference).open() as file:
            ref_state = json.load(file)
    else:
        ref_state = {}

    for dataset in Folder(Path(metafunc.config.option.test_data)).get_dataset_list_rec():
        ids.append(str(dataset.path))
        with dataset(parameter_files=['subject']) as d:
            data.append((dataset.path, ref_state[d.id])) if ref_state else data.append((dataset.path, {}))

    return ids, suites, data

@pytest.fixture(autouse=True)
def WRITE_TOLERANCE():
    return 1.e6

@pytest.fixture()
def test_parameters(request):
    return request.param

@pytest.fixture()
def test_properties(request):
    try:
        return request.param
    except AttributeError:
        return None

@pytest.fixture()
def test_data(request):
    return request.param

@pytest.fixture()
def test_jcampdx_data(request):
    return None

@pytest.fixture()
def test_split_data(request):
    return None
