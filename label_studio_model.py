from collections.abc import Sequence

from attrs import define

# from cattrs.preconf.json import make_converter


@define
class Value:
    id: int


@define
class ChoicesValue(Value):
    id: int


@define
class Result:
    id: int


@define
class ChoicesResult(Result):
    from_name: str
    to_name: str
    origin: str
    value: ChoicesValue
    type: str = "choices"


@define
class LabelStudioData:
    text: str


@define
class LabelStudioAnnotation:
    id: int
    result: Sequence[Result]


@define
class Preannotation:
    file_upload: str
    data: LabelStudioData
    predictions: Sequence[LabelStudioAnnotation]
