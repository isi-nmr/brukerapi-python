from .dataset import Dataset
from .study import *
from .jcampdx import JCAMPDX
from .exceptions import *
import operator

import copy

ops = { "==": operator.eq,
        "<": operator.lt,
        ">": operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        }


class Filter():
    def __init__(self, in_place=True):
        self.in_place = in_place

    def filter(self, folder):

        # either perform the filtering of the original folder, or make a copy
        if self.in_place:
            folder = folder
        else:
            folder = copy.deepcopy(folder)

        # perform filtering
        folder = self.filter_pass(folder)

        # remove empty children
        return self.clean(folder)

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
                if isinstance(node, Folder) or isinstance(node, Study):
                    q += node._children
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
                if isinstance(node, Folder) or isinstance(node, Study):
                    q += node._children
        return list

    def filter_pass(self, node):
        children_out = []
        for child in node._children:

            if isinstance(child, Folder):
                children_out.append(self.filter_pass(child))
            else:
                try:
                    self.filter_eval(child)
                    children_out.append(child)
                except FilterEvalFalse:
                    pass
        node._children = children_out
        return node



    def clean(self, node):
        return node

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
        node.load_parameters()
        try:
            value = node.get_value(self.parameter)
        except KeyError:
            raise FilterEvalFalse
        node.unload_parameters()

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
