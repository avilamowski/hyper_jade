
from enum import Enum


class PromptType(Enum):
	PRESENCE = ("presence", "Checks if something that must be present in the code is implemented.")
	CONCEPTUAL = ("conceptual", "Checks if a conceptual understanding or pattern is present in the code.")
	REQUIREMENT_PRESENCE = ("requirement_presence", "Checks if a specific requirement is implemented in the code.")
	ERROR_PRESENCE = ("error_presence", "Checks if a forbidden or erroneous pattern is present in the code.")

	def __new__(cls, value, description):
		obj = object.__new__(cls)
		obj._value_ = value
		obj.description = description
		return obj
