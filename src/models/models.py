from typing import List, TypedDict, Annotated
from enum import Enum
class PromptType(Enum):
    REQUIREMENT_PRESENCE = ("requirement_presence", "Checks if a specific requirement is implemented in the code.")
    """ 
    Identifies if a specific requirement is implemented in the code.
    If is implemented, returns the lines of code that implement the requirement.
    """
    ERROR_PRESENCE = ("error_presence", "Checks if a forbidden or erroneous pattern is present in the code.")
    """ 
    Identifies if a specific error is present in the code.
    If is implemented, returns the lines of code that implement the requirement.
    """
    STYLISTIC = ("stylistic", "Checks for universal code style violations.")
    """
    Identifies stylistic errors in the code that are universal and don't depend on specific assignment requirements.
    These include poor indentation, bad variable naming, infinite loops, code readability issues, etc.
    Returns an explanation if a style violation is found.
    """

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj


class Requirement(TypedDict):
    requirement: str
    function: str
    type: PromptType


class GeneratedPrompt(TypedDict):
    requirement: Requirement
    examples: str
    jinja_template: str
    index: int


class Submission(TypedDict):
    code: str


class Correction(TypedDict):
    requirement: Requirement
    result: str


class GroupedCode(TypedDict):
    function_name: str
    code: str
    submission_name: str
    line_numbers: List[str]


class ReferenceCorrection(TypedDict):
    """Reference correction containing a list of correction items"""
    corrections: List[str]
