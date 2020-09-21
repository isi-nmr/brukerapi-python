

class UnknownAcqSchemeException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'Unknown acquisition scheme, {}'.format(self.message)
        else:
            return 'Unknown acquisition scheme'


class UnsuportedDatasetType(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'Dataset type: {} is not supported'.format(self.message)
        else:
            return 'Dataset type is not supported'


class InvalidJcampdxFile(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{} is not valid JCAMP-DX file'.format(self.message)
        else:
            return 'Invalid JCAMP-DX file'


class ParameterNotFound(Exception):
    def __init__(self, *args):
        if args:
            self.key = args[0]
            self.path = args[1]
        else:
            self.message = None

    def __str__(self):
        if self.key and self.path:
            return '{} not found in {}'.format(self.key, self.path)
        else:
            return 'Parameter not found'


class JcampdxVersionError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '"{}" is not a valid JCAMP-DX version, supported versions are {}'.format(self.message, SUPPORTED_VERSIONS)
        else:
            return 'Not a valid JCAMP-DX version'


class JcampdxFileError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'Not a valid JCAMP-DX file {} '.format(self.message)
        else:
            return 'Not a valid JCAMP-DX file'


class JcampdxInvalidLine(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'Not a valid JCAMP-DX data line {} '.format(self.message)
        else:
            return 'Not a valid JCAMP-DX data line'


class DatasetTypeMissmatch(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'DatasetTypeMissmatch'


class IncompleteDataset(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'DatasetTypeMissmatch'


class ConditionNotMet(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'Not a valid JCAMP-DX version'


class SequenceNotMet(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'Message {}'.format(self.message)
        else:
            return 'Not a valid JCAMP-DX version'


class PvVersionNotMet(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return 'Message {}'.format(self.message)
        else:
            return 'Not a valid ParaVision version'


class FilterEvalFalse(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'FilterEvalFalse'


class NotADatasetDir(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'NotADatasetDir {}'.format(self.message)

class ScanNotFound(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'Scan: {} not found'.format(self.message)


class RecoNotFound(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'Reco: {} not found'.format(self.message)


class ParametersNotLoaded(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'ParametersNotLoaded'


class SchemeNotLoaded(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'SchemeNotLoaded'


class DataNotLoaded(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'DataNotLoaded'


class TrajNotLoaded(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'TrajNotLoaded'


class NotStudyFolder(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'Not a Bruker study folder.'


class NotExperimentFolder(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'Not a Bruker experiment folder.'


class NotProcessingFolder(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'Not a Bruker processing folder.'


class PropertyConditionNotMet(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'Not a Bruker processing folder.'
