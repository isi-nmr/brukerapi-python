from .dataset import Dataset
from .jcampdx import JCAMPDX
from .exceptions import *
from pathlib import Path
import copy
import operator as op
import json
from random import random



class Folder:
    """A representation of a generic folder. It implements several functions to simplify the folder manipulation."""
    def __init__(
            self,
            path: str,
            parent: 'Folder' = None,
            recursive: bool = True,
            dataset_index: list = ['fid','2dseq','ser','rawdata']
    ):
        """The constructor for Folder class.

        :param path: path to a folder
        :param parent: parent :class:`.Folder` object
        :param recursive: recursively create sub-folders
        :param dataset_index: only data sets listed here will be indexed
        :return:
        """
        self.path = Path(path)
        self.validate()

        self.parent = parent
        self._dataset_index = dataset_index
        self.children = self.make_tree(recursive=recursive)

    def validate(self):
        """Validate whether the given path exists an leads to a folder.
        :return:
        :raises :obj:`.NotADirectoryError`:
        """
        if not self.path.is_dir() or not self.path.exists():
            raise NotADirectoryError

    def __str__(self) -> str:
        return str(self.path)

    def __getattr__(
            self,
            name: str
    ):
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
        for child in self.children:
            if child.path.name == name:
                return child
        raise KeyError

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

    @property
    def dataset_list(self) -> list:
        """List of :obj:`.Dataset` instances contained in folder"""
        return [x for x in self.children if isinstance(x, Dataset)]

    @property
    def dataset_list_rec(self) -> list:
        """List of :obj:`.Dataset` instances contained in folder"""
        return TypeFilter(Dataset).list(self)

    @property
    def jcampdx_list(self) -> list:
        """List of :obj:`.JCAMPDX` instances contained in folder"""
        return [x for x in self.children if isinstance(x, JCAMPDX)]

    @property
    def experiment_list(self) -> list:
        """List of :obj:`.Experiment` instances contained in folder and its sub-folders"""
        return TypeFilter(Experiment).list(self)

    @property
    def processing_list(self) -> list:
        """List of :obj:`.Processing` instances contained in folder and its sub-folders"""
        return TypeFilter(Processing).list(self)

    @property
    def study_list(self) -> list:
        """List of :obj:`.Study` instances contained in folder and its sub-folders"""
        return TypeFilter(Study).list(self)

    def make_tree(
            self,
            recursive: bool = True
    ) -> list:
        """Make a directory tree containing brukerapi objects only

        :param self:
        :param recursive: explore all levels of hierarchy
        :return:
        """
        children = []
        for file in self.path.iterdir():
            path = self.path / file

            if path.is_dir() and recursive:
                # try create Study
                try:
                    children.append(Study(path, parent=self, recursive=recursive))
                    continue
                except NotStudyFolder:
                    pass
                # try create Experiment
                try:
                    children.append(Experiment(path, parent=self, recursive=recursive))
                    continue
                except NotExperimentFolder:
                    pass
                #try create Processing
                try:
                    children.append(Processing(path, parent=self, recursive=recursive))
                    continue
                except NotProcessingFolder:
                    pass
                children.append(Folder(path, parent=self, recursive=recursive, dataset_index=self._dataset_index))
                continue
            try:
                if path.name in self._dataset_index:
                    children.append(Dataset(path, load=False))
                    continue
            except (UnsuportedDatasetType, IncompleteDataset, NotADatasetDir):
                pass
            try:
                children.append(JCAMPDX(path, load=False))
                continue
            except (InvalidJcampdxFile, JcampdxVersionError):
                pass
        return children

    @staticmethod
    def contains(
            path: str,
            required: list
    ) -> bool:
        """Checks whether folder specified by path contains files listed in required.

        :param path: path to a folder
        :param required: list of required files
        :return:
        """
        for file in path.iterdir():
            try:
                required.remove(file.name)
            except ValueError:
                pass

        if required:
            return False
        else:
            return True

    def print(self, level=0, recursive=True):
        """Print structure of the :obj:`.Folder` instance.

        :param level: level of hierarchy
        :param recursive: print recursively
        :return:
        """
        if level == 0:
            prefix=''
        else:
            prefix = '{} └--'.format('  ' * level)

        print('{} {} [{}]'.format(prefix,self.path.name, self.__class__.__name__))

        for child in self.children:
            if isinstance(child, Folder) and recursive:
                child.print(level=level+1)
            else:
                print('{} {} [{}]'.format('  '+prefix,child.path.name, child.__class__.__name__))

    def clean(self, node: 'Folder' = None) -> 'Folder':
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

    def to_json(self, path=None):
        if path:
            with open(path, 'w') as json_file:
                json.dump(self.to_json(), json_file, sort_keys=True, indent=4)
        else:
            return json.dumps(self.to_json(), sort_keys=True, indent=4)

    def report(self, path_out=None, format_=None, write=True, props=None, verbose=None):

        out = {}

        if format_ is None:
            format_ = 'json'

        for dataset in self.dataset_list_rec:
            with dataset(add_parameters=['subject']) as d:
                if write:
                    if path_out:
                        d.report(path=path_out/'{}.{}'.format(d.id, format_), props=props, verbose=verbose)
                    else:
                        d.report(path=d.path.parent/'{}.{}'.format(d.id, format_), props=props, verbose=verbose)
                else:
                    out[d.id]=d.to_json(props=props)

        if not write:
            return out


class Study(Folder):
    """Representation of the Bruker Study folder. The folder contains a subject info and a number of experiment folders.

    Tutorial :doc:`tutorials/how-to-study`

    """
    def __init__(
            self,
            path: str,
            parent: 'Folder' = None,
            recursive: bool = True
    ):
        """The constructor for Study class.

        :param path: path to a folder
        :param parent: parent :class:`.Folder` object
        :param recursive: recursively create sub-folders
        :return:
        """
        self.path = Path(path)
        self.validate()
        super(Study, self).__init__(path, parent=parent, recursive=recursive)

    def validate(self):
        """Validate whether the given path exists an leads to a :class:`Study` folder.

        :raises: :obj:`.NotStudyFolder`: if the path does not lead to folder, or the folder does not contain a subject file
        """
        if not self.path.is_dir():
            raise NotStudyFolder

        if not self.contains(self.path, ['subject',]):
            raise NotStudyFolder

    def get_dataset(
            self,
            exp_id: str = None,
            proc_id: str = None
    ) -> Dataset:
        """Get a :obj:`.Dataset` from the study folder. Fid data set is returned if `exp_id` is specified, 2dseq data set
        is returned if `exp_id` and `proc_id` are specified.

        :param exp_id: name of the experiment folder
        :param proc_id: name of the processing folder
        :return: fid, or 2dseq :obj:`.Dataset`
        """
        if exp_id:
            exp = self._get_exp(exp_id)

        if proc_id:
            return exp._get_proc(proc_id)['2dseq']
        else:
            return exp['fid']

    def _get_exp(self, exp_id):
        for exp in self.experiment_list:
            if exp.path.name == exp_id:
                return exp


class Experiment(Folder):
    """Representation of the Bruker Experiment folder. The folder can contain *fid*, *ser* a *rawdata.SUBTYPE* data sets.
    It can contain multiple :obj:`.Processing` instances.
    """
    def __init__(
            self,
            path: str,
            parent: 'Folder' = None,
            recursive: bool = True,
            dataset_index: list = ['fid','ser', 'rawdata']
    ):
        """The constructor for Experiment class.

        :param path: path to a folder
        :param parent: parent :class:`.Folder` object
        :param recursive: recursively create sub-folders
        :return:
        """
        self.path = Path(path)
        self.validate()
        super(Experiment, self).__init__(path, parent=parent, recursive=recursive, dataset_index=dataset_index)

    def validate(self):
        """Validate whether the given path exists an leads to a :class:`Experiment` folder.

        :raises: :obj:`.NotExperimentFolder`: if the path does not lead to folder, or the folder does not contain an acqp file
        """
        if not self.path.is_dir():
            raise NotExperimentFolder

        if not self.contains(self.path, ['acqp', ]):
            raise NotExperimentFolder

    def _get_proc(self, proc_id):
        for proc in self.processing_list:
            if proc.path.name == proc_id:
                return proc


class Processing(Folder):
    def __init__(self, path, parent=None, recursive=True, dataset_index=['2dseq','1r','1i']):
        """The constructor for Processing class.

        :param path: path to a folder
        :param parent: parent :class:`.Folder` object
        :param recursive: recursively create sub-folders
        :return:
        """
        self.path = Path(path)
        self.validate()
        super(Processing, self).__init__(path, parent=parent, recursive=recursive, dataset_index=dataset_index)

    def validate(self):
        """Validate whether the given path exists an leads to a :class:`Processing` folder.

        :raises: :obj:`.NotProcessingFolder`: if the path does not lead to folder, or the folder does not contain an *visu_pars* file
        """
        if not self.path.is_dir():
            raise NotProcessingFolder

        if not self.contains(self.path, ['visu_pars',]):
            raise NotProcessingFolder


class Filter:
    def __init__(self, query, in_place=True, recursive=True):
        self.in_place = in_place
        self.recursive = recursive
        self.query = query

    def filter(self, folder):

        # either perform the filtering of the original folder, or make a copy
        if self.in_place:
            folder = folder
        else:
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
                count +=1
            except FilterEvalFalse:
                pass
            finally:
                if self.recursive:
                    if isinstance(node, Folder) or isinstance(node, Study):
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
                if self.recursive:
                    if isinstance(node, Folder):
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
            with node(add_properties=['subject']) as n:
                n.query(self.query)
        else:
            raise FilterEvalFalse


class TypeFilter(Filter):
    def __init__(self, value, in_place=True, recursive=True):
        super(TypeFilter, self).__init__(in_place, recursive)
        self.type = value

    def filter_eval(self, node):
        if not isinstance(node, self.type):
            raise FilterEvalFalse