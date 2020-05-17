from .dataset import Dataset, SUPPORTED
from .jcampdx import JCAMPDX
from .exceptions import *

from pathlib2 import Path

import os


class Folder():
    def __init__(self, path, recursive=True, dataset_index=['fid','2dseq','ser','rawdata']):

        self.path = Path(path)

        if not self.path.is_dir() or not self.path.exists():
            raise NotADirectoryError

        self._parent = self.path.parent
        self._dataset_index = dataset_index
        self._children = self.get_children(recursive=recursive)

    def __str__(self):
        return self.path.name

    def __iter__(self, node=None):
        if node is None:
            node = self

        for child in node._children:
            if isinstance(child, Folder):
                yield from self.__iter__(node=child)
            elif isinstance(child, Dataset) or isinstance(child, JCAMPDX):
                yield child

    @property
    def datasets(self):
        return [x for x in self._children if isinstance(x, Dataset)]

    @property
    def dataset(self):
        return self.datasets[0]

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
        for element in os.listdir(self.path):
            path = self.path / element

            if path.is_dir() and recursive:
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

    @property
    def is_scan_(self):
        if any(x.type == 'fid' for x in self.datasets):
            return True
        else:
            return False

    @property
    def is_reco_(self):
        if any(x.type == '2dseq' for x in self.datasets):
            return True
        else:
            return False

    @classmethod
    def is_scan(cls, path):
        folder = cls(path, recursive=False)
        return folder.is_scan_

    @classmethod
    def is_reco(cls, path):
        folder = cls(path, recursive=False)
        return folder.is_reco_


class Study(Folder):
    def __init__(self, path):
        super(Study, self).__init__(path, dataset_index=['fid','2dseq'])

        self.subject = JCAMPDX(self.path/'subject')

    def get_dataset(self, scan_id=None, reco_id=None):

        if scan_id:
            out = self._get_scan(scan_id)

        if reco_id:
            out = self._get_reco(out, reco_id)

            if not out:
                raise RecoNotFound(scan_id)

        return out.dataset

    def _get_scan(self, scan_id):
        scan = None
        for child in self._children:
            if isinstance(child, Folder):
                if child.is_scan_:
                    if child.path.name == scan_id:
                        scan = child
        if scan:
            return scan
        else:
            raise ScanNotFound(scan_id)

    def _get_reco(self, scan_folder, reco_id):
        proc = self._get_proc(scan_folder)

        reco = None

        for folder in proc._children:
            if folder.is_reco_:
                if folder.path.name == reco_id:
                    reco = folder

        if reco:
            return reco
        else:
            raise RecoNotFound(reco_id)

    def _get_proc(self, scan_folder):
        for child in scan_folder._children:
            if isinstance(child, Folder) and child.path.name == 'pdata':
                return child

    @property
    def scans(self):
        from .filters import DatasetTypeFilter
        return DatasetTypeFilter('fid').list(self)

    @property
    def recos(self):
        from .filters import DatasetTypeFilter
        return DatasetTypeFilter('2dseq').list(self)