from brukerapi.dataset import Dataset
from brukerapi.folders import Experiment, Processing


def test_folder_traversal_skips_ser_and_processed_spectra(tmp_path):
    experiment_path = tmp_path / "1"
    processing_path = experiment_path / "pdata" / "1"
    processing_path.mkdir(parents=True)

    for path in [
        experiment_path / "acqp",
        experiment_path / "method",
        experiment_path / "fid",
        experiment_path / "ser",
        processing_path / "visu_pars",
        processing_path / "reco",
        processing_path / "2dseq",
        processing_path / "1r",
        processing_path / "1i",
    ]:
        path.write_text("")

    experiment = Experiment(
        experiment_path,
        dataset_state={"parameter_files": [], "property_files": [], "load": 0},
    )

    experiment_datasets = {child.path.name for child in experiment.children if isinstance(child, Dataset)}
    assert experiment_datasets == {"fid"}

    processing = next(child for child in experiment.children if isinstance(child, Processing))
    processing_datasets = {child.path.name for child in processing.children if isinstance(child, Dataset)}
    assert processing_datasets == {"2dseq"}
