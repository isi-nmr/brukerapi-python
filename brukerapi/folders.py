from .dataset import Dataset, SUPPORTED
from .jcampdx import JCAMPDX
from .exceptions import *
from pathlib import Path

import os


class Folder():
    def __init__(self, path, recursive=True, dataset_index=['fid','2dseq','ser','rawdata']):

        self.path = Path(path)
        self.validate(self.path)

        self._parent = self.path.parent
        self._dataset_index = dataset_index
        self._children = self.get_children(recursive=recursive)

    def validate(self, path):
        if not self.path.is_dir() or not self.path.exists():
            raise NotADirectoryError

    def __str__(self):
        return str(self.path)

    def __iter__(self, node=None):
        if node is None:
            node = self

        for child in node._children:
            if isinstance(child, Folder):
                yield from self.__iter__(node=child)
            elif isinstance(child, Dataset) or isinstance(child, JCAMPDX):
                yield child

    def __getattr__(self, name):
        for child in self._children:
            if child.path.name == name:
                return child
        raise KeyError

    def __getitem__(self, name):
        return self.__getattr__(name)

    @property
    def dataset_list(self):
        return [x for x in self._children if isinstance(x, Dataset)]

    @property
    def jcampdx_list(self):
        return [x for x in self._children if isinstance(x, JCAMPDX)]

    @property
    def experiment_list(self):
        from .filters import TypeFilter
        return TypeFilter(Experiment).list(self)

    @property
    def processing_list(self):
        from .filters import TypeFilter
        return TypeFilter(Processing).list(self)

    @property
    def study_list(self):
        from .filters import TypeFilter
        return TypeFilter(Study).list(self)

    def iter_all(self, node=None):
        if node is None:
            node = self

        for child in node._children:
            if isinstance(child, Folder):
                yield from self.iter_all(node=child)
            else:
                yield child.path.name, child

    def get_children(self, recursive=True):
        children = []
        for file in self.path.iterdir():
            path = self.path / file

            if path.is_dir() and recursive:
                # try create Study
                try:
                    children.append(Study(path, recursive=recursive))
                    continue
                except NotStudyFolder:
                    pass
                # try create Experiment
                try:
                    children.append(Experiment(path, recursive=recursive))
                    continue
                except NotExperimentFolder:
                    pass
                #try create Processing
                try:
                    children.append(Processing(path, recursive=recursive))
                    continue
                except NotProcessingFolder:
                    pass

                children.append(Folder(path, recursive=recursive, dataset_index=self._dataset_index))
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
            except InvalidJcampdxFile:
                pass

        return children

    def contains(self, path, required):
        for file in path.iterdir():
            try:
                required.remove(file.name)
            except ValueError:
                pass

        if required:
            return False
        else:
            return True

class Study(Folder):
    def __init__(self, path, recursive=True):
        path = Path(path)

        if not path.is_dir():
            raise NotStudyFolder

        if not self.contains(path, ['subject',]):
            raise NotStudyFolder

        super(Study, self).__init__(path, recursive=recursive)

    def get_dataset(self, exp_id=None, proc_id=None):

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
    def __init__(self, path, recursive=True, dataset_index = ['fid','ser', 'rawdata']):
        path = Path(path)

        if not path.is_dir():
            raise NotExperimentFolder

        if not self.contains(path, ['acqp', ]):
            raise NotExperimentFolder

        super(Experiment, self).__init__(path, recursive=recursive, dataset_index=dataset_index)

    def _get_proc(self, proc_id):
        for proc in self.processing_list:
            if proc.path.name == proc_id:
                return proc


class Processing(Folder):
    def __init__(self, path, recursive=True, dataset_index=['2dseq','1r','1i']):
        path = Path(path)

        if not path.is_dir():
            raise NotProcessingFolder

        if not self.contains(path, ['visu_pars',]):
            raise NotProcessingFolder

        super(Processing, self).__init__(path, recursive=recursive, dataset_index=dataset_index)