class UnknownAcqSchemeException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Unknown acquisition scheme, {self.message}"
        return "Unknown acquisition scheme"


class UnsuportedDatasetType(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Dataset type: {self.message} is not supported"
        return "Dataset type is not supported"


class InvalidJcampdxFile(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message} is not valid JCAMP-DX file"
        return "Invalid JCAMP-DX file"


class ParameterNotFound(Exception):
    def __init__(self, *args):
        if args:
            self.key = args[0]
            self.path = args[1]
        else:
            self.key = None
            self.path = None
            self.message = None

    def __str__(self):
        if self.key and self.path:
            return f"{self.key} not found in {self.path}"
        return "Parameter not found"


class JcampdxVersionError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f'"{self.message}" is not a valid JCAMP-DX version'
        return "Not a valid JCAMP-DX version"


class JcampdxFileError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Not a valid JCAMP-DX file {self.message} "
        return "Not a valid JCAMP-DX file"


class JcampdxInvalidLine(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Not a valid JCAMP-DX data line {self.message} "
        return "Not a valid JCAMP-DX data line"


class DatasetTypeMissmatch(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        return "DatasetTypeMissmatch"


class IncompleteDataset(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return self.message
        return "DatasetTypeMissmatch"


class ConditionNotMet(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "Not a valid JCAMP-DX version"


class SequenceNotMet(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Message {self.message}"
        return "Not a valid JCAMP-DX version"


class PvVersionNotMet(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Message {self.message}"
        return "Not a valid ParaVision version"


class FilterEvalFalse(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "FilterEvalFalse"


class NotADatasetDir(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return f"NotADatasetDir {self.message}"


class ScanNotFound(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return f"Scan: {self.message} not found"


class RecoNotFound(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return f"Reco: {self.message} not found"


class ParametersNotLoaded(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "ParametersNotLoaded"


class SchemeNotLoaded(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "SchemeNotLoaded"


class DataNotLoaded(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "DataNotLoaded"


class TrajNotLoaded(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "TrajNotLoaded"


class NotStudyFolder(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "Not a Bruker study folder."


class NotExperimentFolder(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "Not a Bruker experiment folder."


class NotProcessingFolder(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "Not a Bruker processing folder."


class PropertyConditionNotMet(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"{self.message}"
        return "Not a Bruker processing folder."


class FidSchemaUndefined(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        common = (
            "Schema was not identified for this dataset. This issue might occur in case of a pulse sequence. "
            "Please, contact authors to include the new sequence into the API configuration."
        )
        if self.message:
            return common + f"\n The name of pulse sequence used to measure this dataset is {self.message}"
        return common


class MissingProperty(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Dataset is missing the {self.message} property. We can offer some help, please contact us via https://github.com/isi-nmr/brukerapi-python"
        return "Dataset is missing one of the required properties. We can offer some help, please contact us via https://github.com/isi-nmr/brukerapi-python"
