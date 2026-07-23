from pathlib import Path

from brukerapi.dataset import Dataset
from brukerapi.folders import Folder, Processing, Study


def test_folder_traversal_skips_processed_spectra(tmp_path):
    experiment_path = tmp_path / "1"
    processing_path = experiment_path / "pdata" / "1"
    processing_path.mkdir(parents=True)

    for path in [
        experiment_path / "acqp",
        experiment_path / "method",
        experiment_path / "fid",
        processing_path / "visu_pars",
        processing_path / "reco",
        processing_path / "2dseq",
        processing_path / "1r",
        processing_path / "1i",
    ]:
        path.write_text("")

    folder = Folder(
        tmp_path,
        dataset_state={"parameter_files": [], "property_files": [], "load": 0},
    )

    experiment = folder["1"]
    experiment_datasets = {child.path.name for child in experiment.children if isinstance(child, Dataset)}
    assert experiment_datasets == {"fid"}

    processing = next(child for child in experiment.get_processing_list() if isinstance(child, Processing))
    processing_datasets = {child.path.name for child in processing.children if isinstance(child, Dataset)}
    assert processing_datasets == {"2dseq"}


def test_study_get_dataset_returns_fid_and_2dseq():
    study = Study(
        Path("test/test_data/PV51/0.2H2"),
        dataset_state={"parameter_files": [], "property_files": [], "load": 0},
    )

    fid = study.get_dataset(exp_id="10")
    reconstructed = study.get_dataset(exp_id="10", proc_id="1")

    assert fid.path.name == "fid"
    assert reconstructed.path.name == "2dseq"

    with fid, reconstructed:
        assert fid.data.size > 0
        assert reconstructed.data.size > 0
