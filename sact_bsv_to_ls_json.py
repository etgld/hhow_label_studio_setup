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
from more_itertools import map_reduce

from label_studio_model import LabelStudioAnnotation, LabelStudioData, Preannotation

parser = argparse.ArgumentParser(description="")


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


@dataclass
class Patient:
    identifier: str


@dataclass
class TimeMention:
    span: tuple[int, int]
    time_type: TIMEX3


@dataclass
class Medication:
    span: tuple[int, int]


@dataclass
class Annotation:
    medication: Medication
    time_mention: TimeMention
    tlink: TLINK


@dataclass(frozen=True)
class Note:
    identifier: str
    text: str | None
    annotations: AbstractSet[Annotation] | None


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
    if any(
        row.get(key) is None
        for key in (" Medication Span ", " Time Span ", " Time Type ")
    ):
        raise ValueError(f"Bad row: {row}")
    medication = Medication(span=str_to_span(row.get(" Medication Span ")))
    time_mention = TimeMention(
        span=str_to_span(row.get(" Time Span ")),
        time_type=TIMEX3(row.get(" Time Type ")),
    )
    tlink = TLINK(row.get(" Temporal Relation "))
    return Annotation(medication=medication, time_mention=time_mention, tlink=tlink)


def get_annotations(path: pathlib.Path) -> AbstractSet[Annotation]:
    df = pl.read_csv(path, separator="|").filter(~pl.all_horizontal(pl.all().is_null()))
    return {row_to_annotation(row) for row in df.to_dicts()}


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
    return Note(
        identifier=path.stem.removesuffix("_medTimeSpans"), text=text, annotations=None
    )


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


def note_cluster(notes: Collection[Note]) -> Note:
    if len(notes) == 1:
        return next(iter(notes))
    elif len(notes) == 2:
        with_text = next(note for note in notes if note.text is not None)
        with_annotations = next(note for note in notes if note.annotations is not None)
        if (
            with_text is None and with_annotations is None
        ) or with_text == with_annotations:
            raise ValueError(f"Issue with note cluster {notes}")
        return Note(
            identifier=with_text.identifier,
            text=with_text.text,
            annotations=with_annotations.annotations,
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


def annotations_to_prediction(
    annotations: AbstractSet[Annotation],
) -> Sequence[LabelStudioAnnotation]:
    return []


def note_to_pre_annotation(patient: Patient, note: Note, index: int) -> Preannotation:
    return Preannotation(
        id=index,
        file_upload=f"{patient.identifier}_{note.identifier}",
        data=LabelStudioData(text=Note.text if Note.text is not None else "ERROR"),
        predictions=annotations_to_prediction(
            annotations=Note.annotations if Note.annotations is not None else set()
        ),
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
