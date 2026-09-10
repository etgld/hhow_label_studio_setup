from collections.abc import Sequence

from attrs import define

# from cattrs.preconf.json import make_converter


@define
class Value:
    start: int
    end: int


@define
class ChoicesValue(Value):
    text: str
    choices: Sequence[str]


@define
class LabelsValue(Value):
    text: str
    labels: Sequence[str]


@define
class TextAreaValue(Value):
    text: str


@define
class Result:
    id: str
    value: Value
    from_name: str
    to_name: str
    type: str
    origin: str


@define
class ChoicesResult(Result):
    from_name: str
    value: ChoicesValue
    to_name: str = "text"
    origin: str = "manual"
    type: str = "choices"


@define
class LabelsResult(Result):
    from_name: str
    value: LabelsValue
    to_name: str = "text"
    origin: str = "manual"
    type: str = "labels"


@define
class TextAreaResult(Result):
    from_name: str
    value: TextAreaValue
    to_name: str = "text"
    origin: str = "manual"
    type: str = "textarea"


@define
class LabelStudioData:
    text: str


@define
class LabelStudioAnnotation:
    id: int
    result: Sequence[Result]


@define
class Preannotation:
    id: int
    file_upload: str
    data: LabelStudioData
    predictions: Sequence[LabelStudioAnnotation]
