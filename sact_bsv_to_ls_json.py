import argparse
import os
import pathlib
from collections.abc import Collection, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from itertools import chain
from operator import attrgetter

import cattrs
import polars as pl
from more_itertools import map_reduce, one

from label_studio_model import (
    LabelsResult,
    LabelStudioAnnotation,
    LabelStudioData,
    LabelsValue,
    Preannotation,
    Relation,
    Result,
)
from utils import get_salt_string

parser = argparse.ArgumentParser(description="")

USED_SALTS = set()

parser.add_argument("--output_dir", type=str)
parser.add_argument("--input_bsv_dir", type=str)
parser.add_argument("--input_text_dir", type=str)


class TLINK(StrEnum):
    BEFORE = "BEFORE"
    CONTAINS = "CONTAINS"
    OVERLAP = "OVERLAP"
    BEGINS_ON = "BEGINS-ON"
    ENDS_ON = "ENDS-ON"


class TIMEX3(StrEnum):
    DATE = "DATE"
    TIME = "TIME"
    DURATION = "DURATION"
    QUANTIFIER = "QUANTIFIER"
    PREPOSTEXP = "PREPOSTEXP"
    SET = "SET"
    INSTANT = "INSTANT"


@dataclass(frozen=True)
class Patient:
    identifier: str


@dataclass(frozen=True)
class TimeMention:
    span: tuple[int, int]
    time_type: TIMEX3

    def to_label_studio_value(self, id: str, note_text: str) -> LabelsResult:
        return LabelsResult(
            value=LabelsValue(
                start=self.span[0],
                end=self.span[1],
                text=note_text[self.span[0] : self.span[1]],
                labels=[self.time_type.value],
            ),
            id=id,
            from_name="TIMEX3",
        )


@dataclass(frozen=True)
class Medication:
    span: tuple[int, int]

    def to_label_studio_value(self, id: str, note_text: str) -> LabelsResult:
        return LabelsResult(
            value=LabelsValue(
                start=self.span[0],
                end=self.span[1],
                text=note_text[self.span[0] : self.span[1]],
                labels=["ASPECTUAL"],
            ),
            id=id,
            from_name="EVENT",
        )


@dataclass(frozen=True)
class Annotation:
    medication: Medication
    time_mention: TimeMention
    tlink: TLINK


@dataclass(frozen=True)
class Note:
    identifier: str
    text: str | None
    annotations: frozenset[Annotation] | None


def get_session_unique_salt_string(tries: int = 10) -> str:
    try_index = 0
    while try_index < tries:
        salt_string = get_salt_string()
        if salt_string not in USED_SALTS:
            return salt_string
        try_index += 1
    raise IndexError(
        f"Somehow could not obtain a unique salt within {try_index + 1} tries"
    )


def get_relevant_bsv_files(input_bsv_dir: str) -> Iterable[pathlib.Path]:
    for root, _, files in os.walk(input_bsv_dir):
        for fn in files:
            if fn.endswith("medTimeSpans.bsv"):
                yield pathlib.Path(os.path.join(root, fn))


def get_relevant_text_files(input_text_dir: str) -> Iterable[pathlib.Path]:
    for root, _, files in os.walk(input_text_dir):
        root_path = pathlib.Path(root)
        if root_path.stem.lower().startswith("patient"):
            for fn in files:
                if fn.endswith(".txt"):
                    yield pathlib.Path(os.path.join(root, fn))


@cache
def str_to_span(span: str) -> tuple[int, int]:
    elems = span.split(",")
    try:
        begin, end = elems
        return int(begin), int(end)
    except ValueError:
        raise ValueError(f"Bad span: {span}")


def row_to_annotation(row: Mapping[str, str]) -> Annotation:
    medication_span_column = " Medication Span "
    time_span_column = " Time Span "
    time_type_column = " Time Type "
    temporal_relation_column = " Temporal Relation "
    if any(
        row.get(key) is None
        for key in (medication_span_column, time_span_column, time_type_column)
    ):
        raise ValueError(f"Bad row: {row}")
    medication = Medication(span=str_to_span(row.get(medication_span_column)))
    time_mention = TimeMention(
        span=str_to_span(row.get(time_span_column)),
        time_type=TIMEX3(row.get(time_type_column)),
    )
    tlink_category = row.get(temporal_relation_column)
    if tlink_category is None:
        raise ValueError(f"Missing TLINK {row.get(temporal_relation_column)}")
    tlink = TLINK(tlink_category.removesuffix("-1"))
    return Annotation(medication=medication, time_mention=time_mention, tlink=tlink)


def get_annotations(path: pathlib.Path) -> frozenset[Annotation]:
    df = pl.read_csv(path, separator="|").filter(~pl.all_horizontal(pl.all().is_null()))
    return frozenset({row_to_annotation(row) for row in df.to_dicts()})


def get_patient(path: pathlib.Path) -> Patient:
    return Patient(identifier=path.parent.stem)


def get_note_with_annotations(path: pathlib.Path) -> Note:
    return Note(
        identifier=path.stem.removesuffix("_medTimeSpans"),
        text=None,
        annotations=get_annotations(path),
    )


def get_note_with_text(path: pathlib.Path) -> Note:
    with open(path, mode="r") as f:
        text = f.read()
    return Note(identifier=path.stem, text=text, annotations=None)


def get_annotated_notes(
    paths: Iterable[pathlib.Path],
) -> AbstractSet[Note]:
    return {get_note_with_annotations(path) for path in paths}


def get_notes_with_text(
    paths: Iterable[pathlib.Path],
) -> AbstractSet[Note]:
    return {get_note_with_text(path) for path in paths}


def bsv_files_to_annotation_maps(
    bsv_paths: Iterable[pathlib.Path],
) -> Mapping[Patient, AbstractSet[Note]]:
    return map_reduce(bsv_paths, keyfunc=get_patient, reducefunc=get_annotated_notes)


def note_to_note_text_maps(
    note_paths: Iterable[pathlib.Path],
) -> Mapping[Patient, AbstractSet[Note]]:
    return map_reduce(note_paths, keyfunc=get_patient, reducefunc=get_notes_with_text)


def safe_get[T](notes: Collection[Note], attribute: str) -> T:
    try:
        return one(
            list(map(attrgetter(attribute), notes)),
            too_long=ValueError,
            too_short=ValueError,
        )
    except ValueError:
        raise ValueError(f"Problem with attribute {attribute} in notes {notes}")


def note_cluster(notes: Collection[Note]) -> Note:
    if len(notes) == 1:
        return next(iter(notes))
    elif len(notes) == 2:
        with_text = next((note for note in notes if note.text is not None), None)
        with_annotations = next(
            (note for note in notes if note.annotations is not None), None
        )
        if (
            with_text is None and with_annotations is None
        ) or with_text == with_annotations:
            raise ValueError(f"Issue with note cluster {notes}")
        return Note(
            identifier=safe_get(notes, "idenfitifier"),
            text=safe_get(notes, "text"),
            annotations=safe_get(notes, "annotations"),
        )
    else:
        raise ValueError(f"Problematic note cluster {notes}")


def merge_notes(
    text_notes: AbstractSet[Note], annotation_notes: AbstractSet[Note]
) -> AbstractSet[Note]:
    return set(
        map_reduce(
            chain(text_notes, annotation_notes),
            keyfunc=attrgetter("identifier"),
            reducefunc=note_cluster,
        ).values()
    )


def merge_note_mappings(
    text_note_mappings: Mapping[Patient, AbstractSet[Note]],
    annotation_note_mappings: Mapping[Patient, AbstractSet[Note]],
) -> Mapping[Patient, AbstractSet[Note]]:
    return {
        patient: merge_notes(
            text_notes=text_note_mappings.get(patient, set()),
            annotation_notes=annotation_note_mappings.get(patient, set()),
        )
        for patient in text_note_mappings.keys() | annotation_note_mappings.keys()
    }


def local_annotations_to_label_studio_results(
    note: Note,
) -> Sequence[Result | Relation]:
    if note.annotations is None and note.text is None:
        raise ValueError(f"Bad note missing everything {note}")
    elif note.text is None:
        raise ValueError(f"Note missing text {note}")
    elif note.annotations is None:
        return []
    unique_medications = {annotation.medication for annotation in note.annotations}
    unique_times = {annotation.time_mention for annotation in note.annotations}
    unique_medication_to_label_studio_label = {}
    unique_time_to_label_studio_label = {}
    for unique_medication in unique_medications:
        unique_medication_to_label_studio_label[unique_medication] = (
            unique_medication.to_label_studio_value(
                id=get_session_unique_salt_string(), note_text=note.text
            )
        )
    for unique_time in unique_times:
        unique_time_to_label_studio_label[unique_time] = (
            unique_time.to_label_studio_value(
                id=get_session_unique_salt_string(), note_text=note.text
            )
        )
    relations = set()
    for annotation in note.annotations:
        ls_medication = unique_medication_to_label_studio_label[annotation.medication]
        ls_time = unique_time_to_label_studio_label[annotation.time_mention]
        relations.add(
            Relation(
                from_id=ls_medication.id,
                to_id=ls_time.id,
                labels=[annotation.tlink.value],
            )
        )

    return list(
        chain(
            unique_medication_to_label_studio_label.values(),
            unique_time_to_label_studio_label.values(),
            relations,
        )
    )


def local_annotations_to_label_studio_annotation(
    note: Note,
) -> Sequence[LabelStudioAnnotation]:
    return [
        LabelStudioAnnotation(
            id=-1,  # Temporary
            result=local_annotations_to_label_studio_results(note=note),
        )
    ]


def note_to_pre_annotation(patient: Patient, note: Note, index: int) -> Preannotation:
    return Preannotation(
        id=index,
        file_upload=f"{patient.identifier}_{note.identifier}",
        data=LabelStudioData(text=note.text if note.text is not None else "ERROR"),
        predictions=local_annotations_to_label_studio_annotation(note=note),
    )


def reset_annotation_id(preannotation: Preannotation, index: int) -> int:
    if len(preannotation.predictions) > 0:
        preannotation.predictions[0].id = index
        return index + 1
    return index


def annotated_notes_to_preannotation(
    patient_to_notes: Mapping[Patient, AbstractSet[Note]],
) -> Sequence[Preannotation]:
    index = 1
    result = []
    for patient in sorted(patient_to_notes.keys(), key=attrgetter("identifier")):
        for note in sorted(patient_to_notes[patient], key=attrgetter("identifier")):
            result.append(
                note_to_pre_annotation(patient=patient, note=note, index=index)
            )
    index += 1
    for preannotation in result:
        index = reset_annotation_id(preannotation=preannotation, index=index)
    return result


def convert_and_write(
    input_bsv_dir: str,
    input_text_dir: str,
    output_dir: str,
) -> None:
    relevant_bsv_files = get_relevant_bsv_files(input_bsv_dir=input_bsv_dir)
    relevant_text_files = get_relevant_text_files(input_text_dir=input_text_dir)
    annotation_note_mappings = bsv_files_to_annotation_maps(
        bsv_paths=relevant_bsv_files
    )
    text_note_mappings = note_to_note_text_maps(note_paths=relevant_text_files)
    patient_to_notes = merge_note_mappings(
        text_note_mappings=text_note_mappings,
        annotation_note_mappings=annotation_note_mappings,
    )
    preannotations = annotated_notes_to_preannotation(patient_to_notes=patient_to_notes)
    with open(os.path.join(output_dir, "result.json"), mode="w") as f:
        f.write(cattrs.unstructure(preannotations))


def main():
    args = parser.parse_args()
    convert_and_write(
        input_bsv_dir=args.input_bsv_dir,
        input_text_dir=args.input_text_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
