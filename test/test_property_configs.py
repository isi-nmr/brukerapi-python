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


def test_rawdata_pv360_v3_uses_prefix_matching():
    config = _load_config("properties_rawdata_core.json")

    for branch in config["numpy_dtype"][6:10]:
        assert "#ACQ_sw_version=='<PV-360.1.1>' or #ACQ_sw_version.value.startswith('<PV-360.3.')" in branch["conditions"]

    assert config["job_desc"][1]["conditions"] == ["#ACQ_sw_version.value.startswith('<PV-360.3.')"]
    assert config["shape_storage"][0] == {
        "cmd": "(@job_desc[0],) + (#PVM_EncNReceivers,) + (@job_desc[6],)",
        "conditions": ["#ACQ_sw_version.value.startswith('<PV-360.3.')"],
    }


def test_traj_scheme_detection_is_not_version_gated():
    config = _load_config("properties_traj_core.json")
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in config["scheme_id"])


def test_fid_scheme_config_keeps_exact_matches_before_code_fallback():
    config = _load_config("properties_fid_core.json")
    branches = config["scheme_id"]

    assert branches[0]["cmd"] == "'CART_2D'"
    assert branches[-1]["cmd"] == "'ZTE'"


def test_traj_scheme_config_keeps_exact_matches_before_code_fallback():
    config = _load_config("properties_traj_core.json")

    assert [branch["cmd"] for branch in config["scheme_id"]] == ["'RADIAL'", "'SPIRAL'", "'ZTE'"]


def test_epi_layout_uses_actual_digitized_sample_count():
    config = _load_config("properties_fid_core.json")
    encoding = next(branch for branch in config["encoding_space"] if branch["conditions"] == ["@scheme_id=='EPI'"])
    k_space = next(branch for branch in config["k_space"] if branch["conditions"] == ["@scheme_id=='EPI'"])

    assert encoding["cmd"][0] == "#PVM_DigNp"
    assert k_space["cmd"][0] == "#PVM_DigNp // (#PVM_EncMatrix[1] // #NSegments)"


def test_zte_scheme_is_not_shadowed_by_radial():
    fid = _load_config("properties_fid_core.json")
    traj = _load_config("properties_traj_core.json")

    fid_radial_programs = fid["scheme_id"][3]["conditions"][0][1]
    traj_radial_programs = traj["scheme_id"][0]["conditions"][0][1]

    assert "ZTE.ppg" not in fid_radial_programs
    assert "ZTE.ppg" not in traj_radial_programs
    assert any(branch["cmd"] == "'ZTE'" for branch in fid["scheme_id"])
    assert any(branch["cmd"] == "'ZTE'" for branch in traj["scheme_id"])
    zte_encoding = next(branch for branch in fid["encoding_space"] if branch["conditions"] == ["@scheme_id=='ZTE'"])
    assert zte_encoding["cmd"][4] == "#NPro // #ACQ_phase_factor"
