import copy
import json
from copy import deepcopy
from pathlib import Path

from .dataset import Dataset
from .exceptions import (
    FilterEvalFalse,
    IncompleteDataset,
    InvalidJcampdxFile,
    JcampdxVersionError,
    NotADatasetDir,
    NotExperimentFolder,
    NotProcessingFolder,
    NotStudyFolder,
    UnsuportedDatasetType,
)
from .jcampdx import JCAMPDX

DEFAULT_DATASET_STATE = {"parameter_files": [], "property_files": [], "load": False}


class Folder:
    """A representation of a generic folder. It implements several functions to simplify the folder manipulation."""

    def __init__(
        self,
        path: str,
        parent: "Folder" = None,
        recursive: bool | None = None,  # noqa: FBT001
        dataset_index: list | None = None,
        dataset_state: dict = DEFAULT_DATASET_STATE,
    ):
        """The constructor for Folder class.

        :param path: path to a folder
        :param parent: parent :class:`.Folder` object
        :param recursive: recursively create sub-folders
        :param dataset_index: only data sets listed here will be indexed
        :return:
        """

        if recursive is None:
            recursive = True

        if dataset_index is None:
            dataset_index = ["fid", "2dseq", "ser", "rawdata"]

        self.path = Path(path)

        self.validate()

        self.parent = parent
        self._dataset_index = dataset_index
        self._set_dataset_state(dataset_state)
        self.children = self.make_tree(recursive=recursive)
        self.make_children_map()  # build lookup map after children exist

    def validate(self):
        """Validate whether the given path exists an leads to a folder.
        :return:
        :raises :obj:`.NotADirectoryError`:
        """
        if not self.path.is_dir() or not self.path.exists():
            raise NotADirectoryError

    def _set_dataset_state(self, passed):
        result = deepcopy(DEFAULT_DATASET_STATE)

        if "parameter_files" in passed:
            passed["parameter_files"] = result["parameter_files"] + passed["parameter_files"]

        if "property_files" in passed:
            passed["property_files"] = result["property_files"] + passed["property_files"]

        result.update(passed)
        self._dataset_state = result

    def __str__(self) -> str:
        return str(self.path)

    def make_children_map(self):
        """Build a dictionary for fast name lookups."""
        self._children_map = {child.path.name: child for child in self.children}

    def __getattr__(self, name: str):
        """Access individual files in folder. :obj:`.Dataset` and :obj:`.JCAMPDX` instances are not loaded, to access the
        data and parameters, to load the data, use context manager, or the `load()` function.

        Example:

        .. code-block:: python

            with folder.fid as fid
                data = fid.data
                te = fid.EffectiveTE

        :param name: Name of Dataset, JCAMPDX, or Folder
        :return:
        """
        if hasattr(self, "_children_map"):
            try:
                return self._children_map[name]
            except KeyError:
                pass
        else:
            self.make_children_map()
            if name in self._children_map:
                return self._children_map[name]

        raise KeyError(f"Child '{name}' not found in {self.path}")

    def __getitem__(self, name):
        """Access individual files in folder, dict style. :obj:`.Dataset` and :obj:`.JCAMPDX` instances are not loaded, to access the
        data and parameters, to load the data, use context manager, or the `load()` function.


        Example:

        .. code-block:: python

            with folder['fid'] as fid
                data = fid.data
                te = fid.EffectiveTE

        :param name: Name of :obj:`.Dataset`, :obj:`.JCAMPDX`, or :obj:`.Folder` object
        :return:
        """
        return self.__getattr__(name)

    def query(self, query):
        """Query each dataset in the folder recursively.

        :param query:
        :return:
        """

        self.query_pass(query, node=self)

        self.clean(node=self)

    def query_pass(self, query: str, node: "Folder" = None):
        children_out = []
        for child in node.children:
            if isinstance(child, Folder):
                children_out.append(self.query_pass(query, node=child))
            elif isinstance(child, Dataset):
                try:
                    child.load_parameters()
                    child.load_properties()
                    child.query(query)
                    child.unload()
                    children_out.append(child)
                except FilterEvalFalse:
                    pass
        node.children = children_out
        return node

    def clean(self, node: "Folder" = None) -> "Folder":
        """Remove empty folders from the tree

        :param node:
        :return: tree without empty folders
        """
        if node is None:
            node = self

        remove = []
        for child in node.children:
            if isinstance(child, Folder):
                self.clean(child)
                if not child.children:
                    remove.append(child)
        for child in remove:
            node.children.remove(child)

    def get_dataset_list(self) -> list:
        """List of :obj:`.Dataset` instances contained in folder"""
        return [x for x in self.children if isinstance(x, Dataset)]

    def get_dataset_list_rec(self) -> list:
        """List of :obj:`.Dataset` instances contained in folder"""
        return TypeFilter(Dataset).list(self)

    def get_jcampdx_list(self) -> list:
        """List of :obj:`.JCAMPDX` instances contained in folder"""
        return [x for x in self.children if isinstance(x, JCAMPDX)]

    def get_experiment_list(self) -> list:
        """List of :obj:`.Experiment` instances contained in folder and its sub-folders"""
        return TypeFilter(Experiment).list(self)

    def get_processing_list(self) -> list:
        """List of :obj:`.Processing` instances contained in folder and its sub-folders"""
        return TypeFilter(Processing).list(self)

    def get_study_list(self) -> list:
        """List of :obj:`.Study` instances contained in folder and its sub-folders"""
        return TypeFilter(Study).list(self)

    def make_tree(self, *, recursive: bool = True) -> list:
        """Build a folder tree with optimized traversal."""
        children = []
        entries = list(self.path.iterdir())

        for path in entries:
            if path.is_dir() and recursive:
                if Study.contains(path, ["subject"]):
                    children.append(Study(path, parent=self, recursive=recursive, dataset_index=self._dataset_index, dataset_state=self._dataset_state))
                    continue
                if Experiment.contains(path, ["acqp"]):
                    children.append(Experiment(path, parent=self, recursive=recursive, dataset_index=self._dataset_index, dataset_state=self._dataset_state))
                    continue
                if Processing.contains(path, ["visu_pars"]):
                    children.append(Processing(path, parent=self, recursive=recursive, dataset_index=self._dataset_index, dataset_state=self._dataset_state))
                    continue
                children.append(Folder(path, parent=self, recursive=recursive, dataset_index=self._dataset_index, dataset_state=self._dataset_state))
                continue

            if path.name in self._dataset_index or (path.name.partition(".")[0] in self._dataset_index and "rawdata" in path.name):
                try:
                    children.append(Dataset(path, **self._dataset_state))
                except (UnsuportedDatasetType, IncompleteDataset, NotADatasetDir):
                    continue

            try:
                children.append(JCAMPDX(path, load=False))
            except (InvalidJcampdxFile, JcampdxVersionError):
                continue

        return children

    @staticmethod
    def contains(path: str | Path, required: list) -> bool:
        """Checks whether folder specified by path contains all required files."""
        path = Path(path)
        required_set = set(required)
        existing_files = {f.name for f in path.iterdir()}
        return required_set.issubset(existing_files)

    def print(self, level=0, recursive=None):
        """Print structure of the :obj:`.Folder` instance.

        :param level: level of hierarchy
        :param recursive: print recursively
        :return:
        """

        if recursive is None:
            recursive = True

        if level == 0:
            prefix = ""
        else:
            prefix = "{} └--".format("  " * level)

        print(f"{prefix} {self.path.name} [{self.__class__.__name__}]")

        for child in self.children:
            if isinstance(child, Folder) and recursive:
                child.print(level=level + 1)
            else:
                print("{} {} [{}]".format("  " + prefix, child.path.name, child.__class__.__name__))

    def to_json(self, path=None):
        if path:
            with open(path, "w") as json_file:
                json.dump(self.to_json(), json_file, sort_keys=True, indent=4)
        else:
            return json.dumps(self.to_json(), sort_keys=True, indent=4)
        return None

    def report(self, path_out=None, format_=None, write=None, props=None, verbose=None):
        if write is None:
            write = True

        out = {}

        if format_ is None:
            format_ = "json"

        for dataset in self.get_dataset_list_rec():
            with dataset(add_parameters=["subject"]) as d:
                if write:
                    if path_out:
                        d.report(path=path_out / f"{d.id}.{format_}", props=props, verbose=verbose)
                    else:
                        d.report(path=d.path.parent / f"{d.id}.{format_}", props=props, verbose=verbose)
                else:
                    out[d.id] = d.to_json(props=props)

        if not write:
            return out
        return None


class Study(Folder):
    """Representation of the Bruker Study folder. The folder contains a subject info and a number of experiment folders.

    Tutorial :doc:`tutorials/how-to-study`

    """

    def __init__(
        self,
        path: str,
        parent: "Folder" = None,
        recursive: bool | None = None,  # noqa: FBT001
        dataset_index: list | None = None,
        dataset_state: dict = DEFAULT_DATASET_STATE,
    ):
        """The constructor for Study class.

        :param path: path to a folder
        :param parent: parent :class:`.Folder` object
        :param recursive: recursively create sub-folders
        :return:
        """

        if recursive is None:
            recursive = True

        if dataset_index is None:
            dataset_index = ["fid", "2dseq", "ser", "rawdata"]

        self.path = Path(path)
        self.validate()
        super().__init__(path, parent=parent, recursive=recursive, dataset_index=dataset_index, dataset_state=dataset_state)

    def validate(self):
        """Validate whether the given path exists an leads to a :class:`Study` folder.

        :raises: :obj:`.NotStudyFolder`: if the path does not lead to folder, or the folder does not contain a subject file
        """
        if not self.path.is_dir():
            raise NotStudyFolder

        if not self.contains(
            self.path,
            [
                "subject",
            ],
        ):
            raise NotStudyFolder

    def get_dataset(self, exp_id: str | None = None, proc_id: str | None = None) -> Dataset:
        """Get a :obj:`.Dataset` from the study folder. Fid data set is returned if `exp_id` is specified, 2dseq data set
        is returned if `exp_id` and `proc_id` are specified.

        :param exp_id: name of the experiment folder
        :param proc_id: name of the processing folder
        :return: fid, or 2dseq :obj:`.Dataset`
        """
        if exp_id:
            exp = self._get_exp(exp_id)

        if proc_id:
            return exp._get_proc(proc_id)["2dseq"]
        return exp["fid"]

    def _get_exp(self, exp_id):
        for exp in self.experiment_list:
            if exp.path.name == exp_id:
                return exp
        return None


class Experiment(Folder):
    """Representation of the Bruker Experiment folder. The folder can contain *fid*, *ser* a *rawdata.SUBTYPE* data sets.
    It can contain multiple :obj:`.Processing` instances.
    """

    def __init__(
        self,
        path: str,
        parent: "Folder" = None,
        recursive: bool | None = None,  # noqa: FBT001
        dataset_index: list | None = None,
        dataset_state: dict = DEFAULT_DATASET_STATE,
    ):
        """The constructor for Experiment class.

        :param path: path to a folder
        :param parent: parent :class:`.Folder` object
        :param recursive: recursively create sub-folders
        :return:
        """

        if recursive is None:
            recursive = True

        if dataset_index is None:
            dataset_index = ["fid", "ser", "rawdata"]

        self.path = Path(path)
        self.validate()
        super().__init__(path, parent=parent, recursive=recursive, dataset_index=dataset_index, dataset_state=dataset_state)

    def validate(self):
        """Validate whether the given path exists an leads to a :class:`Experiment` folder.

        :raises: :obj:`.NotExperimentFolder`: if the path does not lead to folder, or the folder does not contain an acqp file
        """
        if not self.path.is_dir():
            raise NotExperimentFolder

        if not self.contains(
            self.path,
            [
                "acqp",
            ],
        ):
            raise NotExperimentFolder

    def _get_proc(self, proc_id):
        for proc in self.processing_list:
            if proc.path.name == proc_id:
                return proc
        return None


class Processing(Folder):
    def __init__(self, path, parent=None, recursive=None, dataset_index=None, dataset_state: dict = DEFAULT_DATASET_STATE):
        """The constructor for Processing class.

        :param path: path to a folder
        :param parent: parent :class:`.Folder` object
        :param recursive: recursively create sub-folders
        :return:
        """

        if recursive is None:
            recursive = True

        if dataset_index is None:
            dataset_index = ["2dseq", "1r", "1i"]

        self.path = Path(path)
        self.validate()
        super().__init__(path, parent=parent, recursive=recursive, dataset_index=dataset_index, dataset_state=dataset_state)

    def validate(self):
        """Validate whether the given path exists an leads to a :class:`Processing` folder.

        :raises: :obj:`.NotProcessingFolder`: if the path does not lead to folder, or the folder does not contain an *visu_pars* file
        """
        if not self.path.is_dir():
            raise NotProcessingFolder

        if not self.contains(
            self.path,
            [
                "visu_pars",
            ],
        ):
            raise NotProcessingFolder


class Filter:
    def __init__(self, query, in_place=None, recursive=None):
        if in_place is None:
            in_place = True

        if recursive is None:
            recursive = True

        self.in_place = in_place
        self.recursive = recursive
        self.query = query

    def filter(self, folder):
        # either perform the filtering of the original folder, or make a copy
        if not self.in_place:
            folder = copy.deepcopy(folder)

        # perform filtering
        folder = self.filter_pass(folder)

        # remove empty children
        return folder.clean()

    def count(self, folder):
        count = 0
        q = []
        q.append(folder)
        while q:
            node = q.pop()
            try:
                self.filter_eval(node)
                count += 1
            except FilterEvalFalse:
                pass
            finally:
                if self.recursive and (isinstance(node, (Folder, Study))):
                    q += node.children
        return count

    def list(self, folder):
        list = []
        q = []
        q.append(folder)
        while q:
            node = q.pop()
            try:
                self.filter_eval(node)
                list.append(node)
            except FilterEvalFalse:
                pass
            finally:
                if self.recursive and isinstance(node, Folder):
                    q += node.children
        return list

    def filter_pass(self, node):
        children_out = []
        for child in node.children:
            if isinstance(child, Folder):
                children_out.append(self.filter_pass(child))
            else:
                try:
                    self.filter_eval(child)
                    children_out.append(child)
                except FilterEvalFalse:
                    pass
        node.children = children_out
        return node

    def filter_eval(self, node):
        if isinstance(node, Dataset):
            with node(add_properties=["subject"]) as n:
                n.query(self.query)
        else:
            raise FilterEvalFalse


class TypeFilter(Filter):
    def __init__(self, value, in_place=None, recursive=None):
        if in_place is None:
            in_place = True
        if recursive is None:
            recursive = True

        super().__init__(in_place, recursive)
        self.type = value

    def filter_eval(self, node):
        if not isinstance(node, self.type):
            raise FilterEvalFalse
