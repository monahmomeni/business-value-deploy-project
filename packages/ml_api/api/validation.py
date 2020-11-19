from marshmallow import Schema, fields
from marshmallow import ValidationError

import typing as t
import json


class InvalidInputError(Exception):
    """Invalid model input."""


class CustomerActivityDataRequestSchema(Schema):
    people_id = fields.Str()
    activity_id = fields.Str()
    date_x = fields.Date(format='%Y-%m-%d')
    activity_category = fields.Str()
    char_1_x = fields.Str(allow_none=True)
    char_2_x = fields.Str(allow_none=True)
    char_3_x = fields.Str(allow_none=True)
    char_4_x = fields.Str(allow_none=True)
    char_5_x = fields.Str(allow_none=True)
    char_6_x = fields.Str(allow_none=True)
    char_7_x = fields.Str(allow_none=True)
    char_8_x = fields.Str(allow_none=True)
    char_9_x = fields.Str(allow_none=True)
    char_10_x = fields.Str(allow_none=True)
    char_1_y = fields.Str()
    group_1 = fields.Str()
    char_2_y = fields.Str()
    date_y = fields.Date(format='%Y-%m-%d')
    char_3_y = fields.Str()
    char_4_y = fields.Str()
    char_5_y = fields.Str()
    char_6_y = fields.Str()
    char_7_y = fields.Str()
    char_8_y = fields.Str()
    char_9_y = fields.Str()
    char_10_y = fields.Bool()
    char_11 = fields.Bool()
    char_12 = fields.Bool()
    char_13 = fields.Bool()
    char_14 = fields.Bool()
    char_15 = fields.Bool()
    char_16 = fields.Bool()
    char_17 = fields.Bool()
    char_18 = fields.Bool()
    char_19 = fields.Bool()
    char_20 = fields.Bool()
    char_21 = fields.Bool()
    char_22 = fields.Bool()
    char_23 = fields.Bool()
    char_24 = fields.Bool()
    char_25 = fields.Bool()
    char_26 = fields.Bool()
    char_27 = fields.Bool()
    char_28 = fields.Bool()
    char_29 = fields.Bool()
    char_30 = fields.Bool()
    char_31 = fields.Bool()
    char_32 = fields.Bool()
    char_33 = fields.Bool()
    char_34 = fields.Bool()
    char_35 = fields.Bool()
    char_36 = fields.Bool()
    char_37 = fields.Bool()
    char_38 = fields.Integer()


def _filter_error_rows(errors: dict,
                       validated_input: t.List[dict]
                       ) -> t.List[dict]:
    """Remove input data rows with errors."""

    indexes = errors.keys()
    # delete them in reverse order so that you
    # don't throw off the subsequent indexes.
    for index in sorted(indexes, reverse=True):
        del validated_input[index]

    return validated_input


def validate_inputs(input_data):
    """Check prediction inputs against schema."""
    # set many=True to allow passing in a list
    schema = CustomerActivityDataRequestSchema(many=True)

    errors = None
    try:
        schema.load(input_data)
    except ValidationError as exc:
        errors = exc.messages

    if errors:
        validated_input = _filter_error_rows(
            errors=errors,
            validated_input=input_data)
    else:
        validated_input = input_data

    return validated_input, errors
