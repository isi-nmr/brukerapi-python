import pytest
from pathlib import Path
import json
from brukerapi.folders import Folder

def pytest_addoption(parser):
    parser.addoption("--test_data", action="store", default="")
    parser.addoption("--test_suites", action="store", default="")


def pytest_generate_tests(metafunc):
    ids, suites, data = get_test_data(metafunc)
    if 'test_parameters' in metafunc.fixturenames and 'test_parameters' in suites:
        metafunc.parametrize('test_parameters', data, indirect=True, ids=ids)
    elif 'test_properties' in metafunc.fixturenames and 'test_properties' in suites:
        metafunc.parametrize('test_properties', data, indirect=True, ids=ids)
    elif 'test_data' in metafunc.fixturenames and 'test_data' in suites:
        metafunc.parametrize('test_data', data, indirect=True, ids=ids)

def get_test_data(metafunc):
    print('in')
    suites = metafunc.config.option.test_suites.split(" ")
    study_id = Path(metafunc.config.option.test_data).name
    ids = []
    data = []

    with (Path('config') / (study_id + '_properties.json')).open() as file:
        ref_state = json.load(file)

    for dataset in Folder(metafunc.config.option.test_data).get_dataset_list_rec():
        ids.append(str(dataset.path))
        with dataset(parameter_files=['subject']) as d:
            data.append((dataset.path, ref_state[d.id]))

    return ids, suites, data

@pytest.fixture(autouse=True)
def WRITE_TOLERANCE():
    return 1.e6

@pytest.fixture()
def test_dataset(request):
    return request.param

@pytest.fixture()
def test_parameters(request):
    return request.param

@pytest.fixture()
def test_properties(request):
    return request.param

@pytest.fixture()
def test_data(request):
    return request.param
