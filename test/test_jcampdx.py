import numpy as np

from brukerapi.jcampdx import JCAMPDX, GenericParameter


# @pytest.mark.skip(reason="in progress")
def test_jcampdx(test_jcampdx_data):
    dataset_info, dataset_folder = test_jcampdx_data
    jcamp_file_path = dataset_folder / dataset_info["path"]

    j = JCAMPDX(jcamp_file_path)

    for key, ref in test_jcampdx_data[0]["parameters"].items():
        parameter_test = j.get_parameter(key)
        size_test = parameter_test.size
        value_test = parameter_test.value
        type_test = value_test.__class__

        value_ref = ref["value"]
        size_ref = ref["size"]
        type_ref = ref["type"]

        # test SIZE
        if size_ref == "None":
            size_ref = None
        if isinstance(size_ref, list):
            size_ref = tuple(size_ref)
        elif isinstance(size_ref, int):
            size_ref = (size_ref,)
        assert size_ref == size_test

        # test TYPE
        assert type_ref == type_test.__name__

        # test VALUE
        if isinstance(value_test, np.ndarray):
            value_ref = np.array(value_ref)
            assert np.array_equal(value_ref, value_test)
        elif isinstance(value_test, list):
            assert value_test == value_ref
        else:
            assert value_ref == value_test


def test_parse_value_preserves_delimiters_inside_angle_brackets():
    enum = "(operation, <[1H] TX Volume, RX Surface Array>)"
    struct = "(7, <label, with comma) and parenthesis>, 9)"

    assert GenericParameter.parse_value(enum) == ["operation", "<[1H] TX Volume, RX Surface Array>"]
    assert GenericParameter.parse_value(struct) == [7, "<label, with comma) and parenthesis>", 9]


def test_parallel_lists_preserve_delimiters_inside_angle_brackets():
    value = "(first, <Display, One>) (second, <Display) Two, value>)"

    parts = GenericParameter.split_parallel_lists(value)

    assert [GenericParameter.parse_value(part) for part in parts] == [
        ["first", "<Display, One>"],
        ["second", "<Display) Two, value>"],
    ]


def test_jcampdx_get_value_preserves_enum_display_name(tmp_path):
    path = tmp_path / "configscan"
    path.write_text(
        "##TITLE=Parameter List\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "##$CONFIG_SCAN_operation_mode=(operation, <[1H] TX Volume, RX Surface Array>)\n"
        "##END=\n"
    )

    assert JCAMPDX(path).get_value("CONFIG_SCAN_operation_mode") == [
        "operation",
        "<[1H] TX Volume, RX Surface Array>",
    ]
