import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pytest

from brukerapi.folders import Folder


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
    "PV51": "0.2H2.zip",
    "PV601": "20200612_094625_lego_phantom_3_1_2.zip",
    "PV700": "20210128_122257_LEGO_PHANTOM_API_TEST_1_1.zip",
}

LOCAL_DATASETS = ["PV360-V37"]

GITHUB_DATASETS = {
    "PV360_StdData": {
        "repository": "https://github.com/cecilyen/PV360_StdData.git",
        "revision": "6f1b67e5dbc3d7b3646a6315959ccf6d4bd02237",
        "media": "https://media.githubusercontent.com/media/cecilyen/PV360_StdData/6f1b67e5dbc3d7b3646a6315959ccf6d4bd02237",
    },
}

TEST_DIR = Path(__file__).parent
ZENODO_ZIP_DIR = TEST_DIR / "zenodo_zips"
TEST_DATA_ROOT = TEST_DIR / "test_data"


def pytest_sessionstart(session):
    for dataset in ZENODO_FILES:
        _ensure_test_data(dataset)
    for dataset in GITHUB_DATASETS:
        _ensure_github_test_data(dataset)


# -------------------------------
# Helpers
# -------------------------------
def _resolve_requested_datasets(opt: str | None):
    if not opt or opt.lower() == "all":
        available_local_datasets = [name for name in LOCAL_DATASETS if (TEST_DATA_ROOT / name).is_dir()]
        return [*ZENODO_FILES, *GITHUB_DATASETS, *available_local_datasets]
    return [opt]


def _is_required_github_data_file(path: Path):
    return path.name in {"2dseq", "traj"} or path.name.startswith("rawdata.job")


def _is_git_lfs_pointer(path: Path):
    try:
        with path.open("rb") as file:
            return file.read(42) == b"version https://git-lfs.github.com/spec/v1"
    except OSError:
        return False


def _download_github_lfs_file(dataset_name: str, path: Path):
    relative_path = path.relative_to(TEST_DATA_ROOT / dataset_name)
    media_root = GITHUB_DATASETS[dataset_name]["media"]
    url = f"{media_root}/{urllib.parse.quote(relative_path.as_posix())}"
    temporary_path = path.with_name(f"{path.name}.download")

    try:
        with urllib.request.urlopen(url) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary_path.replace(path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        pytest.exit(f"GitHub test-data download failed for {relative_path}: {error}", returncode=1)


def _ensure_github_test_data(dataset_name: str):
    dataset_dir = TEST_DATA_ROOT / dataset_name
    config = GITHUB_DATASETS[dataset_name]
    git_environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}

    if not dataset_dir.exists():
        process = subprocess.run(
            ["git", "clone", "--depth", "1", config["repository"], str(dataset_dir)],
            check=False,
            capture_output=True,
            text=True,
            env=git_environment,
        )
        if process.returncode != 0:
            pytest.exit(f"GitHub test-data clone failed: {process.stderr.strip()}", returncode=1)

    revision = subprocess.run(
        ["git", "-C", str(dataset_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0 or revision.stdout.strip() != config["revision"]:
        fetch = subprocess.run(
            ["git", "-C", str(dataset_dir), "fetch", "--depth", "1", "origin", config["revision"]],
            check=False,
            capture_output=True,
            text=True,
            env=git_environment,
        )
        checkout = subprocess.run(
            ["git", "-C", str(dataset_dir), "checkout", "--detach", config["revision"]],
            check=False,
            capture_output=True,
            text=True,
            env=git_environment,
        )
        if fetch.returncode != 0 or checkout.returncode != 0:
            pytest.exit(f"GitHub test-data checkout failed: {fetch.stderr}{checkout.stderr}".strip(), returncode=1)

    for path in dataset_dir.rglob("*"):
        if path.is_file() and _is_required_github_data_file(path) and _is_git_lfs_pointer(path):
            _download_github_lfs_file(dataset_name, path)


def _download_zenodo():
    ZENODO_ZIP_DIR.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "zenodo_get",
            ZENODO_DOI,
            "-o",
            str(ZENODO_ZIP_DIR),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout
        text=True,
        bufsize=1,  # line-buffered
    )

    for line in process.stdout:
        print(line, end="")  # stream live

    returncode = process.wait()
    if returncode != 0:
        pytest.exit(
            f"Zenodo download failed with exit code {returncode}",
            returncode=1,
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
            folder_obj = Folder(subfolder, dataset_state={"parameter_files": [], "property_files": [], "load": 2})
            for ds in folder_obj.get_dataset_list_rec():
                # Only include if a 2dseq file exists
                if ds.type == "2dseq":
                    datasets.append(ds)
    return datasets


def _ensure_test_data(dataset_name: str):
    dataset_dir = TEST_DATA_ROOT / dataset_name
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        return

    if dataset_name not in ZENODO_FILES:
        raise pytest.UsageError(f"Unknown test dataset '{dataset_name}'. Available: {', '.join(ZENODO_FILES)}")

    zip_path = ZENODO_ZIP_DIR / ZENODO_FILES[dataset_name]

    # Download if missing OR corrupted
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad_file = zf.testzip()
            if bad_file is not None:
                zip_path.unlink()  # corrupted → delete
                raise zipfile.BadZipFile
        except zipfile.BadZipFile:
            _download_zenodo()
    else:
        _download_zenodo()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)


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
    if "test_jcampdx_data" in metafunc.fixturenames:
        jcamp_ids = []
        jcamp_data = []
        for dataset_name in requested:
            for folder, file_path in _find_jcampdx_files(dataset_name):
                jcamp_ids.append(f"{dataset_name}/{folder.name}/{file_path.name}")
                jcamp_data.append(({"parameters": {}, "path": file_path.relative_to(folder)}, folder))
        metafunc.parametrize("test_jcampdx_data", jcamp_data, ids=jcamp_ids)

    # -------------------------------
    # Regular dataset tests
    # -------------------------------
    if "test_data" in metafunc.fixturenames:
        data_ids = []
        data_items = []
        for dataset_name in requested:
            dataset_root = TEST_DATA_ROOT / dataset_name
            folder_obj = Folder(dataset_root, dataset_state={"parameter_files": [], "property_files": [], "load": 2})
            for dataset in folder_obj.get_dataset_list_rec():
                data_ids.append(f"{dataset_name}/{dataset.id}")
                data_items.append((dataset.path, ref_state.get(dataset.id, {})))

        metafunc.parametrize("test_data", data_items, indirect=True, ids=data_ids)

    # -------------------------------
    # Random access tests
    # -------------------------------
    if "test_ra_data" in metafunc.fixturenames:
        ra_ids = []
        ra_items = []
        for dataset_name in requested:
            dataset_root = TEST_DATA_ROOT / dataset_name

            folder_obj = Folder(dataset_root, dataset_state={"parameter_files": [], "property_files": [], "load": 2})
            for dataset in _find_2dseq_datasets(dataset_name):
                ra_ids.append(f"{dataset_name}/{dataset.id}")
                ra_items.append((dataset.path, ref_state.get(dataset.id, {})))

        metafunc.parametrize("test_ra_data", ra_items, indirect=True, ids=ra_ids)

    # -------------------------------
    # Split tests (only 2dseq datasets)
    # -------------------------------
    if "test_split_data" in metafunc.fixturenames:
        split_ids = []
        split_items = []
        for dataset_name in requested:
            for ds in _find_2dseq_datasets(dataset_name):
                split_ids.append(f"{dataset_name}/{ds.id}")
                split_items.append((ds.path, ref_state.get(ds.id, {})))
        metafunc.parametrize("test_split_data", split_items, indirect=True, ids=split_ids)


# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture(autouse=True)
def WRITE_TOLERANCE():
    return 1.0e6


@pytest.fixture
def test_parameters(request):
    return request.param


@pytest.fixture
def test_properties(request):
    try:
        return request.param
    except AttributeError:
        return None


@pytest.fixture
def test_data(request):
    return request.param


@pytest.fixture
def test_jcampdx_data(request):
    return request.param


@pytest.fixture
def test_split_data(request):
    return request.param


@pytest.fixture
def test_ra_data(request):
    return request.param
