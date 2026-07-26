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

    standard_acq_length_branches = config["acq_length"][1:3]
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in standard_acq_length_branches)


def test_fid_scheme_detection_is_not_version_gated():
    config = _load_config("properties_fid_core.json")

    for branch in config["scheme_id"]:
        assert not _contains_sw_version_gate(branch["conditions"])
        assert "pv_version" not in " ".join(str(condition) for condition in branch["conditions"])


def test_rawdata_standard_dtype_is_not_version_gated_but_pv360_branch_remains():
    config = _load_config("properties_rawdata_core.json")

    standard_dtype_branches = config["numpy_dtype"][:6]
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in standard_dtype_branches)
    assert "@pv_version.split('.')[0]=='360'" in config["numpy_dtype"][6]["conditions"]


def test_rawdata_pv360_branches_match_on_the_parsed_major_version():
    """Spec 7.1/13: an exact ACQ_sw_version string misses unlisted point releases.

    Matching `<PV-360.1.1>` or a `<PV-360.3.` prefix leaves PV-360.2.x and
    PV-360.4.x falling through every branch.
    """
    config = _load_config("properties_rawdata_core.json")

    for branch in config["numpy_dtype"][6:10]:
        assert "@pv_version.split('.')[0]=='360'" in branch["conditions"]

    assert all("ACQ_sw_version" not in " ".join(branch["conditions"]) for branch in config["job_desc"])
    assert config["pv_version"][0]["cmd"].startswith("#ACQ_sw_version[1:-1]")


def test_rawdata_job_layout_is_selected_by_record_arity():
    config = _load_config("properties_rawdata_core.json")

    assert config["job_desc"] == [
        {
            "cmd": "[v for v in #ACQ_jobs.nested if v[-1] == '<{}>'.format(@subtype)][0]",
            "conditions": ["len(#ACQ_jobs.nested[0])==9"],
        },
        {
            "cmd": "#ACQ_jobs.nested[int(@subtype[3:])]",
            "conditions": ["len(#ACQ_jobs.nested[0])==8"],
        },
    ]
    assert config["shape_storage"] == [
        {
            "cmd": "(@job_desc[0],) + (#PVM_EncNReceivers,) + (@job_desc[6] if len(@job_desc)==9 else @job_desc[3],)",
            "conditions": [],
        }
    ]


def test_fid_pv360_word_size_branches_match_rawdata():
    fid = _load_config("properties_fid_core.json")
    rawdata = _load_config("properties_rawdata_core.json")

    assert fid["numpy_dtype"][6:10] == rawdata["numpy_dtype"][6:10]


def test_fid_dtype_configuration_has_no_topspin_dtypa_fallbacks():
    fid = _load_config("properties_fid_core.json")

    assert all("DTYPA" not in " ".join(branch["conditions"]) for branch in fid["numpy_dtype"])


def test_traj_scheme_detection_is_not_version_gated():
    config = _load_config("properties_traj_core.json")
    assert all(not _contains_sw_version_gate(branch["conditions"]) for branch in config["scheme_id"])


def test_traj_custom_properties_define_a_stable_id():
    traj = _load_config("properties_traj_custom.json")

    assert [*traj] == ["subj_id", "study_id", "exp_id", "id"]
    assert traj["id"][0]["cmd"] == "'Traj_{}*{}*{}'.format(@exp_id, @subj_id, @study_id)"


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


def test_2dseq_scaling_prefers_visu_and_falls_back_to_reco():
    config = _load_config("properties_2dseq_core.json")

    assert [branch["cmd"] for branch in config["slope"]] == [
        "#VisuCoreDataSlope.array",
        "#RECO_map_slope.array",
    ]
    assert [branch["cmd"] for branch in config["offset"]] == [
        "#VisuCoreDataOffs.array",
        "#RECO_map_offset.array",
    ]


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


def test_fid_3d_radial_layout_includes_the_partition_axis():
    fid = _load_config("properties_fid_core.json")
    conditions = [
        "@scheme_id=='RADIAL'",
        "#ACQ_dim==3",
        "np.atleast_1d(#PVM_EncMatrix).size>2",
        "#ACQ_size[1]==#NPro*#PVM_EncMatrix[2]",
    ]

    block_count = next(branch for branch in fid["block_count"] if branch["conditions"] == conditions)
    encoding = next(branch for branch in fid["encoding_space"] if branch["conditions"] == conditions)
    permute = next(branch for branch in fid["permute"] if branch["conditions"] == conditions)
    k_space = next(branch for branch in fid["k_space"] if branch["conditions"] == conditions)
    dim_type = next(branch for branch in fid["dim_type"] if branch["conditions"] == conditions)

    assert block_count["cmd"] == "#NPro*#PVM_EncMatrix[2]*#NI*#NR"
    assert encoding["cmd"][-2] == "#PVM_EncMatrix[2]"
    assert permute["cmd"] == [0, 2, 4, 3, 5, 6, 1]
    assert k_space["cmd"][2] == "#PVM_EncMatrix[2]"
    # one label per stored axis: read, projection, partition, object (NI),
    # repetition (NR), channel
    assert len(dim_type["cmd"]) == 6
    assert dim_type["cmd"][3] == "'object'"
