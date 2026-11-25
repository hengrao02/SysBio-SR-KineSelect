"""Packaged contracts for current runtime results, independent of dataset releases."""
from importlib.resources import files
import json
from jsonschema import Draft202012Validator


def result_validator() -> Draft202012Validator:
    schema_file = files(__name__).joinpath("result_record.schema.json")
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)
