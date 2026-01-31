import inspect

import pytest

from brukerapi import exceptions

exceptions_to_test = [
    ("UnknownAcqSchemeException", "Unknown acquisition scheme", "Unknown acquisition scheme, test"),
    ("UnsuportedDatasetType", "Dataset type is not supported", "Dataset type: test is not supported"),
    ("InvalidJcampdxFile", "Invalid JCAMP-DX file", "test is not valid JCAMP-DX file"),
    ("JcampdxVersionError", "Not a valid JCAMP-DX version", '"test" is not a valid JCAMP-DX version'),
]


def get_exception_classes(module):
    """Return all exception classes defined in the module."""
    return [cls for name, cls in inspect.getmembers(module, inspect.isclass) if issubclass(cls, Exception) and cls.__module__ == module.__name__]


@pytest.mark.parametrize("exc_class", get_exception_classes(exceptions))
def test_all_exceptions(exc_class):
    # Determine constructor signature
    sig = inspect.signature(exc_class.__init__)
    params = list(sig.parameters.values())[1:]  # skip 'self'

    # Build dummy arguments
    dummy_args = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            # Special case for ParameterNotFound
            if exc_class.__name__ == "ParameterNotFound":
                dummy_args.extend(["key", "path"])
            else:
                dummy_args.append("test")
        else:
            dummy_args.append("test")

    # Instantiate exception with args
    exc = exc_class(*dummy_args)
    s = str(exc)
    assert isinstance(s, str)
    assert len(s) > 0

    # Also test default constructor
    exc_default = exc_class()
    s_default = str(exc_default)
    assert isinstance(s_default, str)
