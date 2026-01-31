import os
import pytest
from pathlib import Path
import json
from brukerapi.folders import Folder
import subprocess
import zipfile

# -------------------------------
# Pytest options
# -------------------------------
def pytest_addoption(parser):
    parser.addoption("--test_data", action="store", default="")
    parser.addoption("--test_suites", action="store", default="")
    parser.addoption("--properties_reference", action="store", default="")

# -------------------------------
# Zenodo configuration
# -------------------------------
ZENODO_DOI = "10.5281/zenodo.4522220"

ZENODO_FILES = {
    "PV5.1": "0.2H2.zip",
    "PV601": "20200612_094625_lego_phantom_3_1_2.zip",
    "PV700": "20210128_122257_LEGO_PHANTOM_API_TEST_1_1.zip",
}

TEST_DIR = Path(__file__).parent
ZENODO_ZIP_DIR = TEST_DIR / "zenodo_zips"
TEST_DATA_ROOT = TEST_DIR / "test_data"



# -------------------------------
# Helpers
# -------------------------------
def _resolve_requested_datasets(opt: str | None):
    if not opt or opt.lower() == "all":
        return list(ZENODO_FILES.keys())
    return [opt]

def _download_zenodo():
    ZENODO_ZIP_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["python", "-m", "zenodo_get", ZENODO_DOI, "-o", str(ZENODO_ZIP_DIR)],
        check=True
    )
    
def _find_jcampdx_files(dataset_name: str):
    """
    Returns a list of tuples (dataset_folder, JCAMPDX_file_path)
    """
    files = []
    dataset_root = TEST_DATA_ROOT / dataset_name
    if not dataset_root.exists():
        return files

    # Iterate over all dataset subfolders
    for subfolder in dataset_root.iterdir():
        if subfolder.is_dir():
            for f in subfolder.rglob("method"):  # find all method files recursively
                files.append((subfolder, f))
    return files    

def _find_2dseq_datasets(dataset_name: str):
    dataset_root = TEST_DATA_ROOT / dataset_name
    if not dataset_root.exists():
        return []

    datasets = []
    for subfolder in dataset_root.iterdir():
        if subfolder.is_dir():
            folder_obj = Folder(subfolder)
            for ds in folder_obj.get_dataset_list_rec():
                # Only include if a 2dseq file exists
                if ds.type=="2dseq":
                    datasets.append(ds)
    return datasets

def _ensure_test_data(dataset_name: str):
    dataset_dir = TEST_DATA_ROOT / dataset_name
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        return

    if dataset_name not in ZENODO_FILES:
        raise pytest.UsageError(
            f"Unknown test dataset '{dataset_name}'. Available: {', '.join(ZENODO_FILES)}"
        )

    zip_path = ZENODO_ZIP_DIR / ZENODO_FILES[dataset_name]
    if not zip_path.exists():
        _download_zenodo()

    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Extract and flatten top-level folder
    with zipfile.ZipFile(zip_path) as zf:
        top_level_dirs = set()
        for member in zf.namelist():
            parts = member.split("/")
            if parts[0]:
                top_level_dirs.add(parts[0])

        if len(top_level_dirs) == 1:
            top_folder = list(top_level_dirs)[0]
            for member in zf.namelist():
                flattened_member = "/".join(member.split("/")[1:])
                if flattened_member:
                    target_path = dataset_dir / flattened_member
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())
        else:
            zf.extractall(dataset_dir)
# -------------------------------
# Parametrization: one test per dataset
# -------------------------------
def pytest_generate_tests(metafunc):

    requested = _resolve_requested_datasets(metafunc.config.option.test_data or "all")
    ref_state = {}
    if metafunc.config.option.properties_reference and Path(metafunc.config.option.properties_reference).exists():
        with open(metafunc.config.option.properties_reference) as f:
            ref_state = json.load(f)

    # -------------------------------
    # JCAMPDX tests
    # -------------------------------
    if 'test_jcampdx_data' in metafunc.fixturenames:
        jcamp_ids = []
        jcamp_data = []
        for dataset_name in requested:
            _ensure_test_data(dataset_name)
            for folder, file_path in _find_jcampdx_files(dataset_name):
                jcamp_ids.append(f"{dataset_name}/{folder.name}/{file_path.name}")
                jcamp_data.append(
                    ({'parameters': {}, 'path': file_path.relative_to(folder)}, folder)
                )
        metafunc.parametrize("test_jcampdx_data", jcamp_data, ids=jcamp_ids)

    # -------------------------------
    # Regular dataset tests
    # -------------------------------
    if 'test_data' in metafunc.fixturenames:
        data_ids = []
        data_items = []
        for dataset_name in requested:
            _ensure_test_data(dataset_name)
            dataset_root = TEST_DATA_ROOT / dataset_name
            for subfolder in dataset_root.iterdir():
                if subfolder.is_dir():
                    folder_obj = Folder(subfolder)
                    for dataset in folder_obj.get_dataset_list_rec():
                        data_ids.append(f"{dataset_name}/{dataset.id}")
                        data_items.append((dataset.path, ref_state.get(dataset.id, {})))
        metafunc.parametrize('test_data', data_items, indirect=True, ids=data_ids)

    # -------------------------------
    # Random access tests
    # -------------------------------
    if 'test_ra_data' in metafunc.fixturenames:
        ra_ids = []
        ra_items = []
        for dataset_name in requested:
            _ensure_test_data(dataset_name)
            dataset_root = TEST_DATA_ROOT / dataset_name
            for subfolder in dataset_root.iterdir():
                if subfolder.is_dir():
                    folder_obj = Folder(subfolder)
                    for dataset in _find_2dseq_datasets(dataset_name):
                        ra_ids.append(f"{dataset_name}/{dataset.id}")
                        ra_items.append((dataset.path, ref_state.get(dataset.id, {})))
        metafunc.parametrize('test_ra_data', ra_items, indirect=True, ids=ra_ids)

    # -------------------------------
    # Split tests (only 2dseq datasets)
    # -------------------------------
    if 'test_split_data' in metafunc.fixturenames:
        split_ids = []
        split_items = []
        for dataset_name in requested:
            _ensure_test_data(dataset_name)
            for ds in _find_2dseq_datasets(dataset_name):
                split_ids.append(f"{dataset_name}/{ds.id}")
                split_items.append((ds.path, ref_state.get(ds.id, {})))
        metafunc.parametrize('test_split_data', split_items, indirect=True, ids=split_ids)

# -------------------------------
# Fixtures
# -------------------------------
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
    return request.param

@pytest.fixture()
def test_split_data(request):
    return request.param

@pytest.fixture()
def test_ra_data(request):
    return request.param