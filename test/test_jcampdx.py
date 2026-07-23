import numpy as np

from brukerapi.jcampdx import (
    JCAMPDX,
    DataParameter,
    GenericParameter,
    GeometryParameter,
)


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


def test_run_length_expansion_handles_mid_array_and_exponents():
    parameter = GenericParameter("##$VALUES", "", "1 @2*(-2.5e-3) 4", "4.24")

    assert np.array_equal(parameter.value, np.array([1, -2.5e-3, -2.5e-3, 4]))


def test_run_length_expansion_handles_angle_bracket_strings():
    parameter = GenericParameter("##$VALUES", "", "start @2*(<Name, display value>) end", "4.24")

    assert np.array_equal(
        parameter.value,
        np.array(["start", "<Name, display value>", "<Name, display value>", "end"]),
    )


def test_run_length_expansion_handles_multiple_and_nested_runs():
    parameter = GenericParameter("##$VALUES", "", "@2*(1) @2*(@2*(<enum>))", "4.24")

    assert np.array_equal(parameter.value, np.array(["1", "1", "<enum>", "<enum>", "<enum>", "<enum>"]))


def test_jcampdx_data_parameter_parses_multiline_xy_pairs(tmp_path):
    path = tmp_path / "data"
    path.write_text(
        "##TITLE=XY Data\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "##$POINTS=(XY..XY)\n"
        "1.0, 2.0\n"
        "3.0, 4.0\n"
        "##END=\n"
    )

    assert np.array_equal(
        JCAMPDX(path).get_value("POINTS"),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_jcampdx_float_and_list_serialization_round_trip(tmp_path):
    source = tmp_path / "source"
    source.write_text(
        "##TITLE=Serialization Test\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "##$FLOAT=0.0\n"
        "##$VALUES=( 2 )\n"
        "0.0 0.0\n"
        "##END=\n"
    )
    jcamp = JCAMPDX(source)

    jcamp.get_parameter("FLOAT").value = 1.25
    jcamp.get_parameter("VALUES").value = [2.5, 3.75]

    output = tmp_path / "round-trip"
    jcamp.write(output)
    restored = JCAMPDX(output)

    assert restored.get_value("FLOAT") == 1.25
    assert restored.get_value("VALUES") == [2.5, 3.75]
    assert "1.250000e+00" in output.read_text()


def test_jcampdx_data_parameter_setter_round_trip(tmp_path):
    source = tmp_path / "source-data"
    source.write_text(
        "##TITLE=XY Data\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "##$POINTS=(XY..XY)\n"
        "1.0, 2.0\n"
        "3.0, 4.0\n"
        "##END=\n"
    )
    jcamp = JCAMPDX(source)
    expected = np.array([[5.0, 6.0], [7.0, 8.0]])

    jcamp.get_parameter("POINTS").value = expected
    assert np.array_equal(jcamp.get_value("POINTS"), expected)

    output = tmp_path / "round-trip-data"
    jcamp.write(output)
    assert np.array_equal(JCAMPDX(output).get_value("POINTS"), expected)


def test_geometry_parameter_setter_stores_raw_value():
    parameter = GeometryParameter("##$GEOMETRY", "", "old", "4.24")

    parameter.value = "(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1, 2, 3))"

    assert parameter.val_str == "(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1, 2, 3))"
    assert str(parameter).endswith(parameter.val_str)


def test_generic_parameter_from_values_preserves_constructor_fields():
    parameter = GenericParameter.from_values("4.24", "FLOAT", None, 1.25, user_defined=True)

    assert parameter.key_str == "##$FLOAT"
    assert parameter.size is None
    assert parameter.val_str == "1.250000e+00"
    assert parameter.version == "4.24"
    assert parameter.value == 1.25


def test_parameter_subclass_constructors_support_named_fields():
    generic = GenericParameter(key_str="##$VALUES", size_str="", val_str="1 2", version="5.0")
    data = DataParameter(key_str="##$POINTS", size_str="(XY..XY)", val_str="1, 2\n3, 4", version="5.0")

    assert np.array_equal(generic.value, np.array([1, 2]))
    assert np.array_equal(data.value, np.array([[1, 2], [3, 4]]))


def test_wrap_lines_respects_78_columns_and_preserves_tokens():
    line = "##$LONG=" + " ".join(["1234567890"] * 20)

    wrapped = JCAMPDX.wrap_lines(line)

    assert all(len(part) <= 78 for part in wrapped.splitlines())
    assert wrapped.replace("\n", "") == line


def test_parse_value_does_not_treat_unclosed_parenthesis_as_list():
    value = GenericParameter.parse_value("(not a closed tuple")

    assert isinstance(value, np.ndarray)
    assert np.array_equal(value, np.array(["(not", "a", "closed", "tuple"]))


def test_jcampdx_size_parsing_accepts_compact_and_padded_brackets(tmp_path):
    path = tmp_path / "sizes"
    path.write_text(
        "##TITLE=Size Test\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "##$COMPACT=(2)\n"
        "1 2\n"
        "##$PADDED=(   2   )\n"
        "3 4\n"
        "##$MATRIX=(2, 3)\n"
        "1 2 3 4 5 6\n"
        "##END=\n"
    )
    jcamp = JCAMPDX(path)

    assert jcamp.get_parameter("COMPACT").size == (2,)
    assert jcamp.get_parameter("PADDED").size == (2,)
    assert jcamp.get_parameter("MATRIX").size == (2, 3)


def test_jcampdx_round_trip_preserves_comments_and_end_marker(tmp_path):
    source = tmp_path / "comments"
    source.write_text(
        "##TITLE=Comment Test\n"
        "##JCAMPDX=4.24\n"
        "##DATATYPE=Parameter Values\n"
        "$$ comment attached to VALUE\n"
        "$$ second comment\n"
        "##$VALUE=1\n"
        "$$ comment attached to OTHER\n"
        "##$OTHER=2\n"
        "##END=\n"
    )
    jcamp = JCAMPDX(source)

    output = tmp_path / "round-trip-comments"
    jcamp.write(output)
    serialized = output.read_text()

    assert "$$ comment attached to VALUE\n$$ second comment\n##$VALUE=1" in serialized
    assert "$$ comment attached to OTHER\n##$OTHER=2" in serialized
    assert serialized.endswith("##END=")
    assert JCAMPDX(output).get_value("VALUE") == 1
    assert JCAMPDX(output).get_value("OTHER") == 2
