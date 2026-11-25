"""Schemas for KineSelectSR selection and evaluation artifacts."""
from importlib.resources import files
import json
from jsonschema import Draft202012Validator


def validate_selection_record(payload: dict) -> None:
    schema_file = files(__name__).joinpath("selection_record.schema.json")
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)
