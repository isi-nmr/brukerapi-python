import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np

from .exceptions import ConditionNotMet, InvalidDataset, MissingProperty, UnknownAcqSchemeException
from .paths import read_array

config_paths = {"core": Path(__file__).parents[0] / "config", "custom": Path(__file__).parents[0] / "config"}

# BART's ``enum mri_dims`` indices.  Keeping these local avoids making BART a
# runtime dependency while allowing callers to request arrays in its layout.
BART_DIMS = 16
BART_READ_DIM = 0
BART_PHS1_DIM = 1
BART_PHS2_DIM = 2
BART_COIL_DIM = 3
BART_TIME_DIM = 9
BART_TIME2_DIM = 10
BART_SLICE_DIM = 13
BART_AVG_DIM = 14
BART_BATCH_DIM = 15
BART_DIM_BY_TYPE = {
    "k_space_encode_step_0": BART_READ_DIM,
    "k_space_encode_step_1": BART_PHS1_DIM,
    "k_space_encode_step_2": BART_PHS2_DIM,
    "channel": BART_COIL_DIM,
    "repetition": BART_TIME_DIM,
    "echo": BART_TIME2_DIM,
    # spec 5.2: the NI axis counts acquisition objects (slices x echoes x movie
    # frames), which is what BART calls its slice dimension for a plain
    # multi-slice scan.
    "object": BART_SLICE_DIM,
    "slice": BART_SLICE_DIM,
    "average": BART_AVG_DIM,
}

# properties required for loading of the data array for each dataset type
REQUIRED_PROPERTIES = {
    "fid": ["numpy_dtype", "channels", "block_size", "acq_length", "scheme_id", "block_count", "encoding_space", "permute", "k_space", "encoded_dim", "shape_storage", "dim_type"],
    "fid_proc": [
        "numpy_dtype",
        "channels",
        "block_size",
        "acq_length",
        "scheme_id",
        "block_count",
        "encoding_space",
        "permute",
        "k_space",
        "encoded_dim",
        "shape_storage",
        "dim_type"
    ],
    "2dseq": [
        "pv_version",
        "numpy_dtype",
        "shape_frames",
        "is_single_slice",
        "shape_fg",
        "shape_block",
        "encoded_dim",
        "shape_storage",
        "shape_final",
        "num_slice_packages",
        "slope",
        "offset",
        "dim_type",
    ],
    "rawdata": ["numpy_dtype", "job_desc", "channels", "shape_storage", "dim_type"],
    "traj": ["numpy_dtype", "scheme_id", "traj_type", "shape_storage", "permute", "final", "dim_type"],
}


class Schema:
    """Base class for all schemes"""

    def __init__(self, dataset):
        # Check whether the dataset contains all required properties.
        for property in REQUIRED_PROPERTIES[dataset.type]:
            if not hasattr(dataset, property):
                if property == "scheme_id":
                    pulprog = dataset._parameter_value("PULPROG", "<missing>")
                    method = dataset._parameter_value("Method", "<missing>")
                    raise UnknownAcqSchemeException(f"PULPROG={pulprog!r}, Method={method!r}; pass scheme_id= to override")
                raise MissingProperty(property)
        self._dataset = dataset
        shape = getattr(dataset, "shape_final", None) or getattr(dataset, "k_space", None) or getattr(dataset, "final", None) or getattr(dataset, "shape_storage", None)
        dim_type = getattr(dataset, "dim_type", None)
        if shape is not None and dim_type is not None and len(dim_type) != len(shape):
            warnings.warn(
                f"dim_type {dim_type} does not describe shape {tuple(shape)} for {dataset.path}",
                RuntimeWarning,
                stacklevel=2,
            )

    def permutation_inverse(self, permutation):
        """Get permutation inverse to the input permutation

        :param inverse permutation: list
        :return:
        """
        inverse = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse[p] = i
        return inverse

    def value_filter(self, value):
        if isinstance(value, str):
            if value == "Yes":
                return True
            if value == "No":
                return False
            return value
        return value

    def validate_conditions(self):
        for condition in self._meta["conditions"]:
            # substitute parameters in expression string
            for sub_params in self._sub_params:
                condition_f = condition.replace(sub_params, f"self._sub_params['{sub_params}']")
            if not eval(condition_f):
                raise ConditionNotMet(condition_f)

    def _get_ra_k_space_info(self, layouts, slice_full):
        k_space = []
        k_space_offset = []

        for slc_, size_ in zip(slice_full, layouts["k_space"]):
            if isinstance(slc_, slice):
                start = slc_.start if slc_.start else 0
                stop = slc_.stop if slc_.stop else size_
            elif isinstance(slc_, int):
                start = slc_
                stop = slc_ + 1
            k_space.append(stop - start)
            k_space_offset.append(start)
        return tuple(k_space), np.array(k_space_offset)

    @staticmethod
    def _as_bart(data, axes):
        """Place a compact k-space array into BART's 16-dimensional layout."""
        if data.ndim != len(axes) or len(set(axes)) != len(axes):
            raise ValueError("BART axis mapping must assign each compact axis exactly once")
        source_to_bart = tuple(axes) + tuple(axis for axis in range(BART_DIMS) if axis not in axes)
        padded = np.reshape(data, data.shape + (1,) * (BART_DIMS - data.ndim))
        return np.transpose(padded, np.argsort(source_to_bart))


class SchemaFid(Schema):
    """Raw ordered FID/k-space schema.

    This reader applies storage trimming, dimensional permutation, RARE/EPI
    phase-line ordering, and EPI odd-line mirroring. It does not perform a
    full reconstruction: ramp-sampling regridding and ``RECO_qopts``
    quadrature corrections remain the caller's responsibility.
    """

    @property
    def acquisition_factor(self):
        """Number of stored scalar samples per logical sample."""
        return 1 if str(self._dataset._parameter_value("AQ_mod", "")).lower() == "qf" else 2

    def _declared_or_inferred(self, parameter, declared_value, inferred, warning_flag):
        """Prefer a loaded format declaration, retaining scheme inference."""
        if declared_value is None:
            return inferred
        if declared_value != inferred and not getattr(self._dataset, warning_flag, False):
            warnings.warn(
                f"{parameter}={self._dataset._parameter_value(parameter)!r} disagrees with "
                f"scheme_id={self._dataset.scheme_id!r} for {self._dataset.path}; "
                f"using the declared value",
                RuntimeWarning,
                stacklevel=3,
            )
            setattr(self._dataset, warning_flag, True)
        return declared_value

    @property
    def continuous_train(self):
        """Whether §5.3 declares a phase-factor continuous acquisition train."""
        value = self._dataset._parameter_value("ACQ_scan_size")
        declared = None if value is None else str(value).casefold() == "acq_phase_factor_scans"
        return self._declared_or_inferred(
            "ACQ_scan_size",
            declared,
            "EPI" in self._dataset.scheme_id,
            "_warned_scan_size_disagreement",
        )

    @property
    def mirror_odd_lines(self):
        """Whether a selected reconstruction requests §6.2 odd-line reversal."""
        value = self._dataset._parameter_value("RECO_inp_order")
        declared = None if value is None else str(value).upper() == "REV_ALT_ROWS"
        return self._declared_or_inferred(
            "RECO_inp_order",
            declared,
            "EPI" in self._dataset.scheme_id,
            "_warned_reco_input_order_disagreement",
        )

    @property
    def layouts(self):
        """Dictionary of possible logical layouts of data

        - encoding_space
        - permute
        - k_space

        :return: layouts: dict
        """

        layouts = {"storage": (self._dataset.block_size,) + (self._dataset.block_count,)}
        layouts["encoding_space"] = self._dataset.encoding_space
        layouts["permute"] = self._dataset.permute
        layouts["inverse_permute"] = self.permutation_inverse(layouts["permute"])
        layouts["k_space"] = self._dataset.k_space

        if self.acquisition_factor == 1:
            stored_samples = self._dataset.acq_length * self._dataset.block_count
            for name in ("encoding_space", "k_space"):
                logical_samples = int(np.prod(layouts[name]))
                if stored_samples % logical_samples:
                    raise InvalidDataset(
                        f"real-only AQ_mod=qf sample count {stored_samples} is incompatible with {name} layout {layouts[name]}"
                    )
                ratio = stored_samples // logical_samples
                if ratio > 1:
                    layouts[name] = (layouts[name][0] * ratio,) + tuple(layouts[name][1:])

        layouts["encoding_permuted"] = tuple(np.array(layouts["encoding_space"])[np.array(layouts["permute"])])

        if self.continuous_train:
            discarded = self._dataset.block_size - self._dataset.acq_length
            block_format = self._dataset._parameter_value("GO_block_size")
            scan_shift = self._dataset._parameter_value("ACQ_scan_shift", 0)
            if block_format == "Standard_KBlock_Format" or scan_shift >= 0:
                layouts["acquisition_position"] = (0, self._dataset.acq_length)
            else:
                layouts["acquisition_position"] = (discarded, self._dataset.acq_length)
        else:
            layouts["acquisition_position"] = (0, self._dataset.acq_length)

        return layouts

    def deserialize(self, data, layouts):
        data = self._decode_raw_stream(data, layouts)

        # Form encoding space
        data = self._acquisitions_to_encode(data, layouts)

        # Permute acquisition dimensions
        data = self._encode_to_permute(data, layouts)

        # Form k-space
        data = self._permute_to_kspace(data, layouts)

        # Typically for RARE, or EPI
        data = self._reorder_fid_lines(data, dir="FW")

        if self.mirror_odd_lines:
            data = self._mirror_odd_lines(data)

        data = self._reorder_objects(data, dir="FW")

        return data

    def raw(self):
        """Return decoded FID acquisitions as ``(sample, shot, receiver)``.

        This representation retains acquisition order.  It deliberately does
        not apply encoding-space reshaping, phase-line sorting, EPI mirroring,
        or object-order correction.
        """
        stored = self._dataset._read_binary_file(
            self._dataset.path,
            self._dataset.numpy_dtype,
            self._dataset.shape_storage,
        )
        data = self._decode_raw_stream(stored, self.layouts)
        receivers = int(self._dataset.channels)
        if data.shape[0] % receivers:
            raise InvalidDataset(
                f"decoded FID sample count {data.shape[0]} is not divisible by "
                f"the receiver count {receivers} for {self._dataset.path}"
            )
        samples = data.shape[0] // receivers
        return np.transpose(
            np.reshape(data, (samples, receivers, data.shape[1]), order="F"),
            (0, 2, 1),
        )

    def to_kspace(self, data=None, *, bart=False):
        """Return the decoded FID k-space, optionally in BART's layout."""
        if data is None:
            data = self._dataset.data
        if not bart:
            return data

        try:
            axes = tuple(BART_DIM_BY_TYPE[label] for label in self._dataset.dim_type)
        except KeyError as error:
            raise UnknownAcqSchemeException(
                f"cannot map FID axis {error.args[0]!r} to BART for {self._dataset.path}"
            ) from error
        return self._as_bart(data, axes)

    def _reorder_objects(self, data, dir="FW"):
        """Apply ACQ_obj_order to the axis that counts acquisition objects (spec 5.2)."""
        dim_type = getattr(self._dataset, "dim_type", ())
        axis_label = next((label for label in ("object", "slice") if label in dim_type), None)
        if axis_label is None:
            return data

        try:
            object_order = np.atleast_1d(self._dataset["ACQ_obj_order"].value).astype(int)
        except KeyError:
            return data

        axis = dim_type.index(axis_label)
        if object_order.size != data.shape[axis] or np.array_equal(object_order, np.arange(object_order.size)):
            return data

        indices = np.argsort(object_order) if dir == "FW" else object_order
        return np.take(data, indices, axis=axis)

    def _acquisition_trim(self, data, layouts):
        acquisition_offset = layouts["acquisition_position"][0]
        acquisition_length = layouts["acquisition_position"][1]
        block_length = self.layouts["storage"][0]

        if acquisition_offset > 0:
            # trim on channel level acquisition
            blocks = layouts["storage"][-1]
            channels = self._dataset.channels
            acquisition_offset = acquisition_offset // channels
            acquisition_length = acquisition_length // channels
            data = np.reshape(data, (-1, channels, blocks), order="F")
            return np.reshape(data[acquisition_offset : acquisition_offset + acquisition_length, :, :], (acquisition_length * channels, blocks), order="F")
        # trim on acq level
        if acquisition_length != block_length:
            discarded = data[acquisition_length:, :]
            if self._dataset._parameter_value("GO_block_size") == "Standard_KBlock_Format" and np.any(discarded):
                warnings.warn(
                    f"Expected trailing K-block padding to be zero for {self._dataset.path}, but found nonzero samples",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return data[0:acquisition_length, :]
        return data

    def _decode_raw_stream(self, data, layouts):
        """Trim storage padding and decode interleaved complex samples."""
        data = self._acquisition_trim(data, layouts)
        if self.acquisition_factor == 2:
            return data[0::2, ...] + 1j * data[1::2, ...]
        return data

    def _acquisitions_to_encode(self, data, layouts):
        return np.reshape(data, layouts["encoding_space"], order="F")

    def _encode_to_permute(self, data, layouts):
        return np.transpose(data, layouts["permute"])

    def _permute_to_kspace(self, data, layouts):
        return np.reshape(data, layouts["k_space"], order="F")

    def _reorder_fid_lines(self, data, dir="FW"):
        """
        Function to sort phase encoding lines using PVM_EncSteps1
        :param data ndarray in k-space layout:
        :return:
        """
        # TODO when to use?

        if self._dataset.scheme_id in {"CSI", "SPECTROSCOPY"}:
            return data

        # Create local copies of variables
        try:
            PVM_EncSteps1 = self._dataset["PVM_EncSteps1"].value
        except KeyError:
            return data

        # Order encoding steps for sorting
        PVM_EncSteps1_sorted = np.argsort(PVM_EncSteps1)

        if dir == "BW":
            PVM_EncSteps1_sorted = self.permutation_inverse(PVM_EncSteps1_sorted)

        if data.shape[1] != len(PVM_EncSteps1_sorted):
            raise InvalidDataset(
                f"phase-encode reorder length {len(PVM_EncSteps1_sorted)} does not match k-space axis length {data.shape[1]} for scheme {self._dataset.scheme_id}"
            )

        if np.array_equal(PVM_EncSteps1_sorted, np.arange(len(PVM_EncSteps1_sorted))):
            return data

        for index in np.ndindex(data.shape[2:]):
            index_f = list(index)
            index_f.insert(0, slice(0, data.shape[1]))
            index_f.insert(0, slice(0, data.shape[0]))
            index_f = tuple(index_f)
            tmp = data[index_f]
            data[index_f] = tmp[:, PVM_EncSteps1_sorted]

        return data

    def _mirror_odd_lines(self, data):
        # Both FW and BW run are the same
        # Order encoding steps for sorting

        for index in np.ndindex(data.shape[2:]):
            index_odd = list(index)
            index_odd.insert(0, slice(1, data.shape[1], 2))
            index_odd.insert(0, slice(0, data.shape[0]))
            index_odd = tuple(index_odd)
            tmp = data[index_odd]
            data[index_odd] = tmp[::-1, :]
        return data

    def serialize(self, data, layouts):
        data = self._reorder_objects(data, dir="BW")

        if self.mirror_odd_lines:
            data = self._mirror_odd_lines(data)

        data = self._reorder_fid_lines(data, dir="BW")

        data = np.reshape(data, layouts["encoding_permuted"], order="F")

        data = np.transpose(data, layouts["inverse_permute"])

        data = np.reshape(
            data,
            (layouts["acquisition_position"][1] // self.acquisition_factor, layouts["storage"][1]),
            order="F",
        )

        data_ = np.zeros(layouts["storage"], dtype=self._dataset.numpy_dtype, order="F")

        if self.acquisition_factor == 1:
            start = layouts["acquisition_position"][0]
            stop = start + layouts["acquisition_position"][1]
            data_[start:stop, :] = data.real
        elif layouts["acquisition_position"][0] > 0:
            channels = layouts["k_space"][self._dataset.dim_type.index("channel")]
            data = np.reshape(data, (-1, channels, data.shape[-1]), order="F")
            data_ = np.reshape(data_, (-1, channels, data_.shape[-1]), order="F")
            data_[layouts["acquisition_position"][0] // channels :: 2, :, :] = data.real
            data_[layouts["acquisition_position"][0] // channels + 1 :: 2, :, :] = data.imag
            data = np.reshape(data, (-1, data.shape[-1]), order="F")
            data_ = np.reshape(data_, (-1, data_.shape[-1]), order="F")
        elif layouts["acquisition_position"][1] != layouts["storage"][0]:
            data_[0 : layouts["acquisition_position"][1] : 2, :] = data.real
            data_[1 : layouts["acquisition_position"][1] + 1 : 2, :] = data.imag
        else:
            data_[0::2, :] = data.real
            data_[1::2, :] = data.imag

        return data_

    def ra(self, slice_):
        layouts, layouts_ra = self.get_ra_layouts(slice_)

        """
        random access
        """
        array_ra = np.zeros(layouts_ra["storage"], dtype=self._dataset.numpy_dtype)
        fp = read_array(self._dataset.path, self._dataset.numpy_dtype, layouts["storage"])

        for index_ra in np.ndindex(layouts_ra["k_space"][1:]):
            # index of line in the original k_space
            index_full = tuple(i + o for i, o in zip(index_ra, layouts_ra["k_space_offset"][1:]))
            index_full = self.index_to_data(layouts, (0,) + index_full)
            index_ra_f = self.index_to_data(layouts_ra, (0,) + index_ra)
            array_ra[index_ra_f] = np.array(fp[index_full])

        layouts_ra["k_space"] = (layouts_ra["k_space"][0] // self.acquisition_factor,) + layouts_ra["k_space"][1:]
        layouts_ra["encoding_space"] = (layouts_ra["encoding_space"][0] // self.acquisition_factor,) + layouts_ra["encoding_space"][1:]

        array_ra = self.deserialize(array_ra, layouts_ra)

        singletons = tuple(i for i, v in enumerate(slice_) if isinstance(v, int))

        return np.squeeze(array_ra, axis=singletons)

    def get_ra_layouts(self, slice_):
        layouts = deepcopy(self.layouts)
        layouts["k_space"] = (layouts["k_space"][0] * self.acquisition_factor,) + tuple(layouts["k_space"][1:])
        layouts["encoding_space"] = (layouts["encoding_space"][0] * self.acquisition_factor,) + tuple(layouts["encoding_space"][1:])
        layouts["inverse_permute"] = tuple(self.permutation_inverse(layouts["permute"]))
        layouts["encoding_permute"] = tuple(layouts["encoding_space"][i] for i in layouts["permute"])
        layouts["channel_index"] = self._dataset.dim_type.index("channel") if "channel" in self._dataset.dim_type else None
        layouts["channels"] = self._dataset.channels if layouts["channel_index"] is None else layouts["k_space"][layouts["channel_index"]]
        layouts["acquisition_position_ch"] = (layouts["acquisition_position"][0] // layouts["channels"], layouts["acquisition_position"][1] // layouts["channels"])
        layouts["storage_clear"] = (layouts["acquisition_position"][1], layouts["storage"][1])
        layouts["storage_clear_ch"] = (layouts["storage_clear"][0] // layouts["channels"], layouts["channels"], layouts["storage"][1])
        layouts["storage_ch"] = (layouts["storage"][0] // layouts["channels"], layouts["channels"], layouts["storage"][1])

        layouts_ra = deepcopy(layouts)

        layouts_ra["k_space"], layouts_ra["k_space_offset"] = self._get_ra_k_space_info(layouts, slice_)
        layouts_ra["channels"] = layouts["channels"] if layouts_ra["channel_index"] is None else layouts_ra["k_space"][layouts_ra["channel_index"]]
        layouts_ra["acquisition_position"] = (0, self._get_acquisition_length(layouts, layouts_ra["channels"]))  # delete offset
        # delete offset

        layouts_ra["encoding_space"], layouts_ra["storage"] = self._get_e_ra(layouts, layouts_ra)
        layouts_ra["encoding_permute"] = tuple(layouts_ra["encoding_space"][i] for i in layouts["permute"])

        return layouts, layouts_ra

    def _extrema_init(self, shape):
        min_index = np.array(shape)
        max_index = np.zeros(len(shape), dtype=int)
        return min_index, max_index

    def encode_extrema_update(self, min_enc_index, max_enc_index, enc_index):
        for i in range(len(min_enc_index)):
            min_enc_index[i] = min(min_enc_index[i], enc_index[i])
            max_enc_index[i] = max(max_enc_index[i], enc_index[i])

    def index_to_data(self, layout, index):
        # kspace to linear
        index = np.ravel_multi_index(index, layout["k_space"], order="F")

        # linear to encoding permuted
        index = np.unravel_index(index, layout["encoding_permute"], order="F")
        # permute
        index = tuple(index[i] for i in layout["inverse_permute"])
        # encoding space to linear
        index = np.ravel_multi_index(index, layout["encoding_space"], order="F")
        if layout["acquisition_position"][0] > 0:
            index = np.unravel_index(index, layout["storage_clear_ch"], order="F")
            index = (index[0] + layout["acquisition_position_ch"][0],) + index[1:]
            index = np.ravel_multi_index(index, layout["storage_ch"], order="F")
        elif layout["acquisition_position"][1] != layout["storage"][0]:
            index = np.unravel_index(index, layout["storage_clear"], order="F")
            index = np.ravel_multi_index(index, layout["storage"], order="F")

        index = np.unravel_index(index, layout["storage"], order="F")

        index = (slice(index[0], index[0] + layout["k_space"][0]), index[1])

        return index

    def _get_e_ra(self, layout_full, layout_ra):
        min_enc_index, max_enc_index = self._extrema_init(layout_full["encoding_space"][1:])
        storage_ra = []
        for index_ra in np.ndindex(layout_ra["k_space"][1:]):
            index_full = (0,) + tuple(i + o for i, o in zip(index_ra, layout_ra["k_space_offset"][1:]))

            """
            index_k_to_encode
            """

            index_full = np.ravel_multi_index(index_full, layout_full["k_space"], order="F")

            # linear to encoding permuted
            index_full = np.unravel_index(index_full, layout_full["encoding_permute"], order="F")
            # permute
            index_full = tuple(index_full[i] for i in layout_full["inverse_permute"])

            """
            Update encoding space extrema
            """
            self.encode_extrema_update(min_enc_index, max_enc_index, index_full[1:])

            """
            index_encode_to_data
            """
            index_full = np.ravel_multi_index(index_full, layout_full["encoding_space"], order="F")
            index_full = np.unravel_index(index_full, layout_full["storage_clear"], order="F")
            if index_full[1] not in storage_ra:
                storage_ra.append(index_full[1])

        encoding_space_ra = max_enc_index - min_enc_index + 1
        encoding_space_ra = (layout_full["encoding_space"][0],) + tuple(encoding_space_ra)

        storage_ra = (self._get_acquisition_length(layout_full, layout_ra["channels"]), len(storage_ra))

        return encoding_space_ra, storage_ra

    @staticmethod
    def _get_acquisition_length(layouts, channels):
        return layouts["acquisition_position"][1] // layouts["channels"] * channels

    def index_k_to_encode(self, layout, index):
        index = np.ravel_multi_index(index, layout["k_space"], order="F")
        # linear to encoding permuted
        index = np.unravel_index(index, layout["encoding_permute"], order="F")
        # permute
        index = tuple(index[i] for i in layout["inverse_permute"])
        return index

    def index_encode_to_data(self, layout, index):
        channel = index[layout["channel_index"]] + 1

        index = np.ravel_multi_index(index, layout["encoding_space"], order="F")
        index = np.unravel_index(index, layout["storage"], order="F")

        if layout["acquisition_position"][0] > 0:
            first = index[0] + (layout["acquisition_position"][0] // layout["channels"]) * channel
        else:
            first = index[0]
        index = (slice(first, first + layout["k_space"][0]), index[1])
        return index


class SchemaFidCompanion:
    """Decoder for auxiliary ``fid.<subtype>`` files."""

    def __init__(self, dataset, primary_schema=None):
        self._dataset = dataset
        self._primary_schema = primary_schema

    @property
    def layouts(self):
        if self._dataset.subtype == "orig" and self._primary_schema is not None:
            return self._primary_schema.layouts
        return {"storage": self._dataset.shape_storage}

    def deserialize(self, data, layouts):
        if self._dataset.subtype == "orig" and self._primary_schema is not None:
            return self._primary_schema.deserialize(data, layouts)
        decoded = data[0::2] + 1j * data[1::2]
        shape = getattr(self._dataset, "shape_final", None)
        return decoded if shape is None else np.reshape(decoded, shape, order="F")

    def serialize(self, data, layouts):
        if self._dataset.subtype == "orig" and self._primary_schema is not None:
            return self._primary_schema.serialize(data, layouts)
        data = np.reshape(np.asarray(data), (-1,), order="F")
        serialized = np.empty(data.size * 2, dtype=self._dataset.numpy_dtype)
        serialized[0::2] = np.real(data)
        serialized[1::2] = np.imag(data)
        return serialized


class SchemaTraj(Schema):
    @property
    def layouts(self):
        layouts = {}

        layouts["storage"] = self._dataset.shape_storage
        layouts["final"] = self._dataset.final
        layouts["permute"] = self._dataset.permute

        return layouts

    def deserialize(self, data, layouts):
        data = np.transpose(data, layouts["permute"])
        return np.reshape(data, layouts["final"], order="F")

    def serialize(self, data, layouts):
        permuted_storage = tuple(np.asarray(layouts["storage"])[layouts["permute"]])
        data = np.reshape(data, permuted_storage, order="F")
        data = np.transpose(data, self.permutation_inverse(layouts["permute"]))
        return np.reshape(data, layouts["storage"], order="F")


class SchemaRawdata(Schema):
    """PV-360 rawdata.jobN schema.

    The on-disk job records describe a complex sample stream, not its
    acquisition-space layout.  ``to_kspace`` deliberately supports only the
    Cartesian subset for which the method metadata proves that layout.
    """

    @property
    def layouts(self):
        layouts = {}
        layouts["raw"] = (int(self._dataset.shape_storage[0] / 2), self._dataset.channels, int(self._dataset.shape_storage[2]))
        layouts["shape_storage"] = self._dataset.shape_storage
        layouts["final"] = layouts["raw"]
        return layouts

    def deserialize(self, data, layouts):
        return data[0::2, ...] + 1j * data[1::2, ...]

    def raw(self):
        """Return decoded PV-360 acquisitions as ``(sample, shot, receiver)``."""
        return np.transpose(self._dataset._data, (0, 2, 1))

    def serialize(self, data, layouts):
        # storage array
        data_ = np.zeros(layouts["shape_storage"], dtype=self._dataset.numpy_dtype, order="F")

        # interlace real and imag along first axis
        data_[0::2, ...] = data.real
        data_[1::2, ...] = data.imag

        return data_

    def to_kspace(self, data=None, *, bart=False):
        """Return a Cartesian PV-360 raw-data job in k-space order.

        The returned axes are ``(readout, phase[, partition], object,
        repetition, channel)`` for 2-D and ``(readout, phase, partition,
        repetition, channel)`` for 3-D.  Retrospectively self-gated 2-D
        acquisitions retain their ``NI`` and ``NR`` axes and add an
        ``acquisition_cycle`` axis before the channel axis.  The method
        intentionally does not reconstruct EPI or non-Cartesian acquisitions.
        With ``bart=True``, the same data is returned in the 16-axis BART
        layout.
        """
        if data is None:
            data = self._dataset._data

        scheme_id = self._dataset._infer_scheme_id()
        if scheme_id is not None:
            raise UnknownAcqSchemeException(
                f"rawdata-to-k-space is currently supported only for Cartesian PV-360 jobs, "
                f"but {self._dataset.path} is {scheme_id}; use its acquisition-specific reader"
            )

        encoded_dim = self._dataset._parameter_value("ACQ_dim")
        matrix = np.atleast_1d(self._dataset._parameter_value("PVM_EncMatrix", []))
        if encoded_dim not in (2, 3) or matrix.size < encoded_dim:
            raise UnknownAcqSchemeException(
                f"cannot establish a Cartesian rawdata layout for {self._dataset.path}; "
                "pass data through an acquisition-specific reader"
            )

        matrix = tuple(int(value) for value in matrix[:encoded_dim])
        receivers = int(self._dataset.channels)
        objects = int(self._dataset._parameter_value("NI", 1))
        repetitions = int(self._dataset._parameter_value("NR", 1))
        readout, phase = matrix[:2]

        if encoded_dim == 2 and self._dataset._parameter_value("SelfGating") == "Yes":
            k_space = self._self_gated_k_space(
                data,
                readout,
                phase,
                receivers=receivers,
                objects=objects,
                repetitions=repetitions,
            )
            axes = (BART_READ_DIM, BART_PHS1_DIM, BART_TIME_DIM, BART_TIME2_DIM, BART_BATCH_DIM, BART_COIL_DIM)
            return self._as_bart(k_space, axes) if bart else k_space

        if encoded_dim == 2:
            encoding_space = (readout, receivers, phase, objects, repetitions)
            permute = (0, 2, 3, 4, 1)
        else:
            if objects != 1:
                raise UnknownAcqSchemeException(
                    f"cannot establish a 3-D Cartesian rawdata layout with NI={objects} for "
                    f"{self._dataset.path}"
                )
            encoding_space = (readout, receivers, phase, matrix[2], repetitions)
            permute = (0, 2, 3, 4, 1)

        if data.ndim != 3 or data.shape[0] != readout or data.shape[1] != receivers:
            raise InvalidDataset(
                f"rawdata sample layout {data.shape} does not match Cartesian metadata "
                f"(readout={readout}, receivers={receivers}) for {self._dataset.path}"
            )
        if data.size != int(np.prod(encoding_space)):
            raise InvalidDataset(
                f"rawdata contains {data.size} complex samples but Cartesian layout {encoding_space} "
                f"requires {int(np.prod(encoding_space))} for {self._dataset.path}"
            )

        k_space = np.transpose(np.reshape(data, encoding_space, order="F"), permute)
        k_space = self._reorder_phase_lines(k_space)
        k_space = self._reorder_objects(k_space)
        axes = (
            (BART_READ_DIM, BART_PHS1_DIM, BART_SLICE_DIM, BART_TIME_DIM, BART_COIL_DIM)
            if encoded_dim == 2
            else (BART_READ_DIM, BART_PHS1_DIM, BART_PHS2_DIM, BART_TIME_DIM, BART_COIL_DIM)
        )
        return self._as_bart(k_space, axes) if bart else k_space

    def _self_gated_k_space(self, data, readout, phase, *, receivers, objects, repetitions):
        """Arrange retrospectively gated Cartesian data before cine binning."""
        steps = np.atleast_1d(self._dataset._parameter_value("PVM_EncGenSteps1", []))
        if steps.size == 0 or steps.size % phase:
            raise InvalidDataset(
                f"self-gated phase-encode sequence has {steps.size} steps, which is incompatible "
                f"with phase size {phase} for {self._dataset.path}"
            )
        acquired_frames = steps.size // phase
        output_frames = int(self._dataset._parameter_value("PVM_NMovieFrames", objects))
        if output_frames != objects or acquired_frames % (objects * repetitions):
            raise InvalidDataset(
                f"self-gated dimensions (NI={objects}, NR={repetitions}, PVM_NMovieFrames={output_frames}) "
                f"do not describe {acquired_frames} acquired frames for {self._dataset.path}"
            )
        acquisition_cycles = acquired_frames // (objects * repetitions)
        layout = (readout, receivers, phase, acquired_frames)
        if data.ndim != 3 or data.shape[0] != readout or data.shape[1] != receivers or data.size != int(np.prod(layout)):
            raise InvalidDataset(
                f"rawdata sample layout {data.shape} does not match self-gated Cartesian layout "
                f"{layout} for {self._dataset.path}"
            )

        k_space = np.transpose(np.reshape(data, layout, order="F"), (0, 2, 3, 1))
        order = np.argsort(np.reshape(steps, (phase, acquired_frames), order="F"), axis=0)
        k_space = np.take_along_axis(k_space, order[None, :, :, None], axis=1)
        return np.reshape(
            k_space,
            (readout, phase, objects, acquisition_cycles, repetitions, receivers),
            order="F",
        )

    def _reorder_phase_lines(self, data):
        steps = self._dataset._parameter_value("PVM_EncSteps1")
        if steps is None:
            return data
        indices = np.argsort(np.atleast_1d(steps))
        if indices.size != data.shape[1]:
            raise InvalidDataset(
                f"phase-encode reorder length {indices.size} does not match k-space axis length "
                f"{data.shape[1]} for {self._dataset.path}"
            )
        return np.take(data, indices, axis=1)

    def _reorder_objects(self, data):
        if self._dataset._parameter_value("ACQ_dim") != 2 or data.ndim != 5:
            return data
        order = self._dataset._parameter_value("ACQ_obj_order")
        if order is None:
            return data
        indices = np.argsort(np.atleast_1d(order).astype(int))
        if indices.size != data.shape[2]:
            raise InvalidDataset(
                f"object-order length {indices.size} does not match k-space axis length "
                f"{data.shape[2]} for {self._dataset.path}"
            )
        return np.take(data, indices, axis=2)

class Schema2dseq(Schema):
    """
    Schema2dseq class

    - vector: data vector as obtained from binary file
    - frames: individual frames combined in
    - framegroups: aldasdasd

    """

    @property
    def layouts(self):
        return {
            "shape_fg": self._dataset.shape_fg,
            "shape_frames": self._dataset.shape_frames,
            "shape_block": self._dataset.shape_block,
            "shape_storage": self._dataset.shape_storage,
            "shape_final": self._dataset.shape_final,
        }

    def get_rel_fg_index(self, fg_type):
        try:
            return self._dataset.dim_type[self._dataset.encoded_dim :].index(fg_type)
        except ValueError:
            raise KeyError(f"Framegroup {fg_type} not found in dim_type") from ValueError

    def scale(self):
        data = self._split_complex_frames(self._dataset._data, self.layouts)
        data = self._apply_disk_slice_order(data)
        data = np.reshape(data, self._dataset.shape_storage, order="F")
        data = self._scale_frames(data, self.layouts, "FW")
        data = np.reshape(data, self._dataset.shape_final, order="F")
        data = self._apply_disk_slice_order(data)
        self._dataset.data = self._combine_complex_frames(data)

    def _apply_core_transposition(self, data, layouts, *, inverse=False):
        """Undo the per-frame dimension exchange recorded by VisuCoreTransposition.

        Spec 7.2: a nonzero value means frame f is stored with two of its
        dimensions exchanged relative to VisuCoreSize -- `n < VisuCoreDim`
        exchanges `n` and `n-1`, `VisuCoreDim` exchanges `0` and
        `VisuCoreDim - 1`. Such a frame has to be read in its real on-disk shape
        and swapped back, otherwise the Fortran-order reshape interleaves its
        rows.

        The exchange is skipped when the two dimensions have equal length. There
        the on-disk layout is unchanged, and the frames measure as already
        consistent with VisuCoreOrientation: on a 256x256 three-package
        localizer, sampling the intersection line of an untransposed and a
        transposed frame correlates 0.99 as delivered and -0.05 once swapped,
        while on a 110x120 localizer -- where the exchange does change the
        layout -- the same measurement goes from -0.27 to +0.64.
        """
        transposition = self._dataset._parameter_value("VisuCoreTransposition")
        if transposition is None:
            return data
        transposition = np.atleast_1d(np.asarray(transposition)).astype(int)
        if not transposition.any():
            return data

        block = tuple(int(size) for size in layouts["shape_block"])
        core_dim = len(block)
        frames = layouts.get("frame_index", range(data.shape[-1]))
        out = None
        for position, frame in enumerate(frames):
            value = int(transposition[frame]) if frame < transposition.size else 0
            if value == 0:
                continue
            first, second = (0, core_dim - 1) if value >= core_dim else (value - 1, value)
            if first == second or block[first] == block[second]:
                continue
            stored = list(block)
            stored[first], stored[second] = stored[second], stored[first]
            if out is None:
                out = np.array(data)
            if inverse:
                swapped = np.swapaxes(np.asarray(data[..., position]), first, second)
                out[..., position] = np.reshape(swapped.flatten(order="F"), block, order="F")
            else:
                frame_data = np.reshape(np.asarray(data[..., position]).flatten(order="F"), stored, order="F")
                out[..., position] = np.swapaxes(frame_data, first, second)
        return data if out is None else out

    def deserialize(self, data, layouts):
        data = self._apply_core_transposition(data, layouts)

        # scale
        if self._dataset._state["scale"]:
            data = self._scale_frames(data, layouts, "FW")

        # frames -> frame_groups
        data = self._frames_to_framegroups(data, layouts)
        data = self._apply_disk_slice_order(data)

        return self._combine_complex_frames(data)

    def _frame_group_axis(self, name):
        normalized_name = name.upper()
        for axis, dim_type in enumerate(self._dataset.dim_type):
            if str(dim_type).upper() == normalized_name:
                return axis
        return None

    def _apply_disk_slice_order(self, data):
        disk_order = str(self._dataset._parameter_value("VisuCoreDiskSliceOrder", "")).lower()
        if disk_order != "disk_reverse_slice_order":
            return data

        axis = self._frame_group_axis("FG_SLICE")
        if axis is None and getattr(self._dataset, "encoded_dim", None) == 3:
            # A 3-D VisuCore volume stores slices in its third encoded
            # dimension; it does not need an FG_SLICE frame group.
            axis = 2
        if axis is None or axis >= data.ndim:
            warnings.warn(
                "VisuCoreDiskSliceOrder requests reversed slices but no slice axis is identifiable "
                f"for {getattr(self._dataset, 'path', '<unknown>')}; leaving order unchanged",
                RuntimeWarning,
                stacklevel=2,
            )
            return data
        return np.flip(data, axis=axis)

    def _complex_frame_axis(self, data):
        if not self._dataset._state.get("combine_complex", True):
            return None

        axis = self._frame_group_axis("FG_COMPLEX")
        if axis is None:
            axis = getattr(self._dataset, "_combined_complex_axis", None)
            if axis is None:
                image_type = np.atleast_1d(self._dataset._parameter_value("RECO_image_type", []))
                if not any("COMPLEX_IMAGE" in str(value).upper() for value in image_type):
                    return None
                axis = data.ndim - 1

        if data.shape[axis] == 1:
            # a random-access selection of only the real or only the imaginary
            # component: there is nothing to combine, keep the axis as it is
            return None
        if data.shape[axis] != 2:
            raise InvalidDataset(
                f"complex 2dseq requires a two-element real/imag frame-group axis, got shape {data.shape} on axis {axis}"
            )
        return axis

    def _combine_complex_frames(self, data):
        axis = self._complex_frame_axis(data)
        if axis is None:
            return data
        real = np.take(data, 0, axis=axis)
        imaginary = np.take(data, 1, axis=axis)
        self._dataset._combined_complex_axis = axis
        label_axis = self._frame_group_axis("FG_COMPLEX")
        if label_axis is not None:
            del self._dataset.dim_type[label_axis]
        return real + 1j * imaginary

    def _split_complex_frames(self, data, layouts):
        raw_shape = tuple(layouts["shape_final"])
        if data.shape == raw_shape:
            return data

        axis = self._complex_frame_axis(np.empty(raw_shape))
        if axis is None:
            return data

        expected_shape = raw_shape[:axis] + raw_shape[axis + 1 :]
        if data.shape != expected_shape:
            raise InvalidDataset(f"complex 2dseq data shape {data.shape} does not match expected shape {expected_shape}")
        return np.stack((data.real, data.imag), axis=axis)

    def _scale_frames(self, data, layouts, dir):
        """

        :param data:
        :param layouts:
        :param dir:
        :return:
        """

        # dataset is created with scale state set to False
        if self._dataset._state["scale"] is False:
            return data

        # get a float copy of the data array
        data = data.astype(float)

        slope = self._dataset.slope if "mask" not in layouts else self._dataset.slope[layouts["mask"].flatten(order="F")]
        offset = self._dataset.offset if "mask" not in layouts else self._dataset.offset[layouts["mask"].flatten(order="F")]

        # spec 7.4: a parameter that is not frame-group dependent may carry a
        # single value, which then applies to every frame
        frames = data.shape[-1]
        slope = np.broadcast_to(np.atleast_1d(slope), (frames,)) if np.atleast_1d(slope).size == 1 else slope
        offset = np.broadcast_to(np.atleast_1d(offset), (frames,)) if np.atleast_1d(offset).size == 1 else offset

        for frame in range(data.shape[-1]):
            if dir == "FW":
                data[..., frame] *= float(slope[frame])
                data[..., frame] += float(offset[frame])
            elif dir == "BW":
                data[..., frame] -= float(offset[frame])
                data[..., frame] /= float(slope[frame])

        if dir == "BW":
            data = np.round(data)
            if np.issubdtype(self._dataset.numpy_dtype, np.integer):
                dtype_limits = np.iinfo(self._dataset.numpy_dtype)
                data = np.clip(data, dtype_limits.min, dtype_limits.max)

        return data

    def _frames_to_framegroups(self, data, layouts, mask=None):
        """

        :param data:
        :param layouts:
        :param mask:
        :return:
        """
        if mask:
            return np.reshape(data, (-1,) + layouts["shape_fg"], order="F")
        return np.reshape(data, layouts["shape_final"], order="F")

    def serialize(self, data, layout):
        data = self._split_complex_frames(data, layout)
        data = self._apply_disk_slice_order(data)
        data = self._framegroups_to_frames(data, layout)
        data = self._scale_frames(data, layout, "BW")
        return self._apply_core_transposition(data, layout, inverse=True)

    def _frames_to_vector(self, data):
        return data.flatten(order="F")

    def _framegroups_to_frames(self, data, layouts):
        if layouts.get("mask"):
            return np.reshape(data, (-1,) + layouts["shape_fg"], order="F")
        return np.reshape(data, layouts["shape_storage"], order="F")

    """
    Random access
    """

    def ra(self, slice_):
        """
        Random access to the data matrix.

        :param tuple slice_: Slice object(s) to select data in each dimension.
        :return: Selected subset of the data.
        :rtype: np.ndarray
        """

        layouts, layouts_ra = self._get_ra_layouts(slice_)

        array_ra = np.zeros(layouts_ra["shape_storage"], dtype=self._dataset.numpy_dtype)

        fp = read_array(self._dataset.path, self._dataset.numpy_dtype, layouts["shape_storage"])

        for slice_ra, slice_full in self._generate_ra_indices(layouts_ra, layouts):
            array_ra[slice_ra] = np.array(fp[slice_full])

        array_ra = self.deserialize(array_ra, layouts_ra)

        # Frame-group selection above does not alter encoded dimensions.
        # Apply their requested selection after deserializing the selected
        # frames, preserving singleton axes for the final squeeze below.
        encoded_slice = tuple(
            slice(index, index + 1) if isinstance(index, int) else index for index in slice_[: self._dataset.encoded_dim]
        )
        array_ra = array_ra[encoded_slice + (slice(None),) * (array_ra.ndim - self._dataset.encoded_dim)]

        singletons = tuple(i for i, v in enumerate(slice_) if isinstance(v, int))

        return np.squeeze(array_ra, axis=singletons)

    def _get_ra_layouts(self, slice_full):
        layouts = deepcopy(self.layouts)
        layouts_ra = deepcopy(layouts)

        layouts_ra["mask"] = np.zeros(layouts["shape_fg"], dtype=bool, order="F")
        layouts_ra["mask"][slice_full[self._dataset.encoded_dim :]] = True
        # Which frames of the whole dataset the selection covers: per-frame
        # parameters (VisuCoreTransposition) must be indexed by that absolute
        # frame number, not by the position within the selection.
        layouts_ra["frame_index"] = np.flatnonzero(layouts_ra["mask"].flatten(order="F"))
        layouts_ra["shape_fg"], layouts_ra["offset_fg"] = self._get_ra_shape(layouts_ra["mask"])
        layouts_ra["shape_frames"] = (np.prod(layouts_ra["shape_fg"], dtype=int),)
        layouts_ra["shape_storage"] = layouts_ra["shape_block"] + layouts_ra["shape_frames"]
        layouts_ra["shape_final"] = layouts_ra["shape_block"] + layouts_ra["shape_fg"]

        return layouts, layouts_ra

    def _get_ra_shape(self, mask):
        axes = []
        for axis in range(mask.ndim):
            axes.append(tuple(i for i in range(mask.ndim) if i != axis))

        ra_shape = []
        ra_offset = []
        for axis in axes:
            ra_shape.append(np.count_nonzero(np.count_nonzero(mask, axis=axis)))
            ra_offset.append(np.argmax(np.count_nonzero(mask, axis=axis)))

        return tuple(ra_shape), np.array(ra_offset)

    def _generate_ra_indices(self, layouts_ra, layouts):
        for index_ra in np.ndindex(layouts_ra["shape_final"][self._dataset.encoded_dim :]):
            index = tuple(np.array(index_ra) + layouts_ra["offset_fg"])
            index = tuple(0 for i in range(self._dataset.encoded_dim)) + index
            index_ra_f = tuple(0 for i in range(self._dataset.encoded_dim)) + index_ra

            index_ra_f = np.ravel_multi_index(index_ra_f, layouts_ra["shape_final"], order="F")
            index = np.ravel_multi_index(index, layouts["shape_final"], order="F")

            index_ra_f = np.unravel_index(index_ra_f, layouts_ra["shape_storage"], order="F")
            index = np.unravel_index(index, layouts["shape_storage"], order="F")

            slice_ra = tuple(slice(None) for i in range(self._dataset.encoded_dim)) + index_ra_f[self._dataset.encoded_dim :]
            slice_full = tuple(slice(None) for i in range(self._dataset.encoded_dim)) + index[self._dataset.encoded_dim :]
            yield slice_ra, slice_full
