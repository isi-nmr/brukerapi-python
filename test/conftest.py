import pytest
from pathlib import Path
import json


def pytest_addoption(parser):
    parser.addoption("--test_data", action="store", default="")


@pytest.fixture()
def test_io_data(request):
    return request.param


@pytest.fixture()
def test_jcampdx_data(request):
    return request.param


@pytest.fixture()
def test_ra_data(request):
        return request.param


@pytest.fixture()
def test_split_data(request):
        return request.param


def pytest_generate_tests(metafunc):
    config_path = Path(__file__).parent / metafunc.config.option.test_data
    if "test_io_data" in metafunc.fixturenames:
        try:
            ids, testdata = get_test_data(config_path, 'test_io')
            metafunc.parametrize('test_io_data', testdata, indirect=True, ids=ids)
        except:
            metafunc.parametrize('test_io_data', [], indirect=True, ids=[])
    elif "test_jcampdx_data" in metafunc.fixturenames:
        try:
            ids, testdata = get_test_data(config_path, 'test_jcampdx')
            metafunc.parametrize('test_jcampdx_data', testdata, indirect=True, ids=ids)
        except:
            metafunc.parametrize('test_jcampdx_data', [], indirect=True, ids=[])
    elif "test_ra_data" in metafunc.fixturenames:
        try:
            ids, testdata = get_test_data(config_path, 'test_ra')
            metafunc.parametrize('test_ra_data', testdata, indirect=True, ids=ids)
        except:
            metafunc.parametrize('test_ra_data', [], indirect=True, ids=[])
    elif "test_split_data" in metafunc.fixturenames:
        try:
            ids, testdata = get_test_data(config_path, 'test_split')
            metafunc.parametrize('test_split_data', testdata, indirect=True, ids=ids)
        except:
            metafunc.parametrize('test_split_data', [], indirect=True, ids=[])


def get_test_data(config_path, test_suite):
    with config_path.open() as file:
        data = json.load(file)[test_suite]
    return list(data.keys()), list(data.values())


@pytest.fixture(autouse=True)
def data_path():
    return Path('bruker2nifti_qa/raw')


@pytest.fixture(autouse=True)
def WRITE_TOLERANCE():
    return 1.e6