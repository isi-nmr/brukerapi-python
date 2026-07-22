import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "brukerapi" / "config"


def _load_config(name):
    with (CONFIG_DIR / name).open() as f:
        return json.load(f)


def _contains_sw_version_gate(conditions):
    for condition in conditions:
        if isinstance(condition, str) and "ACQ_sw_version" in condition:
            return True
        if isinstance(condition, list) and condition and condition[0] == "#ACQ_sw_version":
            return True
    return False


def test_fid_dtype_and_block_layout_are_not_version_gated():
    config = _load_config("properties_fid_core.json")

    standard_dtype_branches = config["numpy_dtype"][:6]
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in standard_dtype_branches)

    standard_block_size_branches = config["block_size"][:2]
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in standard_block_size_branches)

    standard_acq_length_branches = config["acq_lenght"][1:4]
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in standard_acq_length_branches)


def test_fid_common_scheme_detection_is_not_version_gated():
    config = _load_config("properties_fid_core.json")

    for branch in config["scheme_id"]:
        if branch["cmd"] == "'SPIRAL'":
            continue
        assert not _contains_sw_version_gate(branch["conditions"])


def test_rawdata_standard_dtype_is_not_version_gated_but_pv360_v1_branch_remains():
    config = _load_config("properties_rawdata_core.json")

    standard_dtype_branches = config["numpy_dtype"][:6]
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in standard_dtype_branches)
    assert _contains_sw_version_gate(config["numpy_dtype"][6]["conditions"])


def test_traj_scheme_detection_is_not_version_gated():
    config = _load_config("properties_traj_core.json")
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in config["scheme_id"])
