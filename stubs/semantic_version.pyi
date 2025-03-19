import re
from types import NotImplementedType
from typing import Any

class Version:
    version_re: re.Pattern
    partial_version_re: re.Pattern
    @classmethod
    def coerce(cls, version_string: str) -> Version: ...
    def __eq__(self, other: Any) -> bool | NotImplementedType: ...
    def __ne__(self, other: Any) -> bool | NotImplementedType: ...
    def __lt__(self, other: Any) -> bool | NotImplementedType: ...
    def __le__(self, other: Any) -> bool | NotImplementedType: ...
    def __gt__(self, other: Any) -> bool | NotImplementedType: ...
    def __ge__(self, other: Any) -> bool | NotImplementedType: ...
