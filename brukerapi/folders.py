from .dataset import Dataset, SUPPORTED
from .jcampdx import JCAMPDX
from .exceptions import *
from pathlib import Path
import copy
import operator as op
import os


class Folder():
    """A representation of a generic folder. It implements several.

    """
    def __init__(self, path, parent=None, recursive=True, dataset_index=['fid','2dseq','ser','rawdata']):

        self.path = Path(path)
        self.validate()

        self.parent = parent
        self._dataset_index = dataset_index
        self.children = self.make_tree(recursive=recursive)

    def validate(self):
        if not self.path.is_dir() or not self.path.exists():
            raise NotADirectoryError

    def __str__(self):
        return str(self.path)

    def __getattr__(self, name):
        for child in self.children:
            if child.path.name == name:
                return child
        raise KeyError

    def __getitem__(self, name):
        return self.__getattr__(name)

    @property
    def dataset_list(self):
        return [x for x in self.children if isinstance(x, Dataset)]

    @property
    def jcampdx_list(self):
        return [x for x in self.children if isinstance(x, JCAMPDX)]

    @property
    def experiment_list(self):
        return TypeFilter(Experiment).list(self)

    @property
    def processing_list(self):
        return TypeFilter(Processing).list(self)

    @property
    def study_list(self):
        return TypeFilter(Study).list(self)

    def make_tree(self, recursive=True):
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
            except InvalidJcampdxFile:
                pass
        return children

    @staticmethod
    def contains(path, required):
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
        """
        Function to print structure of the folder recursively
        :param level:
        :param recursive:
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

    def filter(
            self,
            parameter: str = None,
            operator: str = None,
            value: [str, float, int, list, tuple] = None,
            type: [Dataset, JCAMPDX, 'Folder', 'Experiment', 'Study', 'Processing'] = None,
            name: str = None,
            in_place: bool = True
    ):
        """Filter the folder tree using several types of filters.

        ParameterFilter(parameter, operator, value)
        TypeFilter(type)
        NameFilter(name)

        :param parameter:
        :param operator:
        :param value:
        :param type:
        :param name:
        :param in_place:
        :return:
        """
        if parameter and value and operator:
            return ParameterFilter(parameter, operator, value, in_place=in_place).filter(self)
        if type:
            return TypeFilter(type, in_place=in_place).filter(self)
        if name:
            return NameFilter(name, in_place=in_place).filter(self)

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


class Study(Folder):
    def __init__(self, path, parent=None, recursive=True):
        path = Path(path)

        if not path.is_dir():
            raise NotStudyFolder

        if not self.contains(path, ['subject',]):
            raise NotStudyFolder

        super(Study, self).__init__(path, parent=parent, recursive=recursive)

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
    def __init__(self, path, parent=None, recursive=True, dataset_index = ['fid','ser', 'rawdata']):
        path = Path(path)

        if not path.is_dir():
            raise NotExperimentFolder

        if not self.contains(path, ['acqp', ]):
            raise NotExperimentFolder

        super(Experiment, self).__init__(path, parent=parent, recursive=recursive, dataset_index=dataset_index)

    def _get_proc(self, proc_id):
        for proc in self.processing_list:
            if proc.path.name == proc_id:
                return proc


class Processing(Folder):
    def __init__(self, path, parent=None, recursive=True, dataset_index=['2dseq','1r','1i']):
        path = Path(path)

        if not path.is_dir():
            raise NotProcessingFolder

        if not self.contains(path, ['visu_pars',]):
            raise NotProcessingFolder

        super(Processing, self).__init__(path, parent=parent, recursive=recursive, dataset_index=dataset_index)


class Filter:
    def __init__(self, in_place=True, recursive=True):
        self.in_place = in_place
        self.recursive = recursive

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


ops = {
    "==": op.eq,
        "<": op.lt,
        ">": op.gt,
        "<=": op.le,
        ">=": op.ge,
    }


class ParameterFilter(Filter):
    def __init__(self, parameter, operator, value, in_place=True):
        super(ParameterFilter, self).__init__(in_place=in_place)
        self.parameter = parameter
        self.value = value

        try:
            self.op = ops[operator]
        except KeyError as e:
            raise ValueError('Invalid operator {}'.format(operator))

    def filter_eval(self, node):
        """
        Filters out:
            - anything other than Datasets and JCAMPDX files.
        :param node:
        :return:
        """
        if not isinstance(node, Dataset) and not isinstance(node, JCAMPDX):
            raise FilterEvalFalse

        # TODO context manager, or specific parameter querry
        with node as n:
            try:
                value = n.get_value(self.parameter)
            except KeyError:
                raise FilterEvalFalse

        if not self.op(value,self.value) :
            raise FilterEvalFalse


class DatasetTypeFilter(Filter):
    def __init__(self, value, in_place=True):
        super(DatasetTypeFilter, self).__init__(in_place)
        self.value = value

    def filter_eval(self, node):
        if not isinstance(node, Dataset):
            raise FilterEvalFalse

        if node.type != self.value:
            raise FilterEvalFalse


class TypeFilter(Filter):
    def __init__(self, value, in_place=True, recursive=True):
        super(TypeFilter, self).__init__(in_place, recursive)
        self.type = value

    def filter_eval(self, node):
        if not isinstance(node, self.type):
            raise FilterEvalFalse


class NameFilter(Filter):
    def __init__(self, value, in_place=True, recursive=False):
        super(TypeFilter, self).__init__(in_place, recursive)
        self.name = value

    def filter_eval(self, node):
        if node.path.name != node:
            raise FilterEvalFalse