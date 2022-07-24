# -*- coding: utf-8 -*-
"""Module to decode protocol buffers without the corresponding ``.proto`` file."""
from __future__ import annotations

import base64
import dataclasses
import json
import struct
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from enum import Enum
from typing import Any, TypeVar


E = TypeVar('E', bound=Enum)  # used in Field.as_enum()


class WireType(Enum):
    """Wire type of a :class:`Field`."""
    VARINT = 0
    FIXED_64_BIT = 1
    DELIMITED = 2
    GROUP_START = 3
    GROUP_END = 4
    FIXED_32_BIT = 5


class BytesCursor:
    """Helper class to read bytes and varints from a byte string."""
    def __init__(self, data: bytes):
        self.data = data
        self.cursor = 0

    def __bool__(self) -> bool:
        return self.cursor < len(self.data)

    def current(self) -> int:
        return self.data[self.cursor]

    def read(self) -> int:
        value: int = self.current()
        self.cursor += 1
        return value

    def read_n(self, n: int) -> bytes:
        value: bytes = self.data[self.cursor:self.cursor + n]
        self.cursor += n
        return value

    def read_varint(self) -> int:
        value: int = 0
        i: int = 0
        while self.current() >= 128:
            value += self.read() % 128 << (7 * i)
            i += 1
        value += self.read() % 128 << (7 * i)
        return value


class Convertible(ABC):
    """Interface for converting the values of :class:`Field`-like objects to other types."""
    @abstractmethod
    def auto(self):
        """Returns this field's value as an automatically detected type."""
        raise NotImplementedError

    def as_varint(self) -> int:
        """Returns this field's value as a varint."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to varint')

    def as_zigzag(self) -> int:
        """Returns this field's value as a ZigZag-encoded varint."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to zigzag')

    def as_bool(self) -> bool:
        """Returns this field's value as a :class:`bool`."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to bool')

    def as_enum(self, enum: Callable[[int], E]) -> E:
        """Returns this field's value the provided :class:`Enum` type."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to enum')

    def as_message(self) -> Message:
        """Returns this field's value as a :class:`Message`."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to Message')

    def as_string(self) -> str:
        """Returns this field's value as a :class:`str`."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to str')

    def as_bytes(self) -> bytes:
        """Returns this field's value as a :class:`bytes`."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to bytes')

    def as_packed_varint(self) -> tuple[int, ...]:
        """Returns this field's value as a packed repeated field of varint values."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to packed varint')

    def as_packed_fixed(self, fmt) -> tuple[int | float, ...]:
        """Returns this field's value as a packed repeated field of fixed-width values specified by :obj:`fmt`."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to packed fixed')

    def as_signed(self) -> int:
        """Returns this field's value as a signed fixed-width integer."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to signed')

    def as_unsigned(self) -> int:
        """Returns this field's value as an unsigned fixed-width integer."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to unsigned')

    def as_float(self) -> float:
        """Returns this field's value as a float."""
        raise TypeError(f'cannot convert {self.__class__.__name__} to float')


@dataclasses.dataclass(frozen=True)
class Field(Convertible, ABC):
    """Field of a :class:`Message`."""
    tag: int
    raw: int | bytes | None

    def __post_init__(self):
        if not 1 <= self.tag <= 2 ** 29 - 1:
            raise ValueError('field tag out of range')

    @classmethod
    def from_cursor(cls, cursor: BytesCursor) -> Field:
        """Reads the key and corresponding value from a :class:`BytesCursor`.
        Returns a :class:`Field` object of the appropriate subclass based on the wire type."""
        key = cursor.read_varint()
        tag = key >> 3
        wire_type = WireType(key % 8)
        if wire_type == WireType.VARINT:
            return VarintField(tag, cursor.read_varint())
        elif wire_type == WireType.FIXED_32_BIT:
            return Fixed32BitField(tag, cursor.read_n(4))
        elif wire_type == WireType.FIXED_64_BIT:
            return Fixed64BitField(tag, cursor.read_n(8))
        elif wire_type == WireType.DELIMITED:
            return DelimitedField(tag, cursor.read_n(cursor.read_varint()))
        elif wire_type == WireType.GROUP_START:
            return GroupStartField(tag, None)
        elif wire_type == WireType.GROUP_END:
            return GroupEndField(tag, None)
        else:
            raise ValueError(f'unknown wire type {wire_type}')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.tag!r}, {self.raw!r})'

    def __getitem__(self, tag: int) -> FieldCollection:
        """Returns a :class:`FieldCollection` of any nested :class:`Field` objects with the specified tag when
        interpreted as a :class:`Message`."""
        return self.as_message()[tag]


class VarintField(Field):
    """:class:`Field` of a :class:`Message` with wire type 0 (varint)."""
    raw: int

    def auto(self):
        return self.as_varint()

    def as_varint(self) -> int:
        return self.raw

    def as_zigzag(self) -> int:
        if self.raw % 2 == 0:
            return self.raw >> 1
        else:
            return -((self.raw + 1) >> 1)

    def as_bool(self) -> bool:
        if self.raw == 0:
            return False
        elif self.raw == 1:
            return True
        raise ValueError(f'expected 0 or 1, but found {self.raw} when converting to bool')

    def as_enum(self, enum: Callable[[int], E]) -> E:
        return enum(self.raw)


class DelimitedField(Field):
    """:class:`Field` of a :class:`Message` with wire type 2 (length-delimited)."""
    raw: bytes

    def __len__(self) -> int:
        return len(self.raw)

    def auto(self) -> Message | str | bytes:
        # Return empty string for zero-length bytes
        if len(self.raw) == 0:
            return ''

        try:
            # Try ASCII string if bytes start with a printable character
            if self.raw[0] >= 0x20:
                return self.raw.decode('ascii')
        except UnicodeDecodeError:
            pass

        try:
            # Try protobuf message
            return self.as_message()
        except ValueError:
            try:
                # Try UTF-8 string
                return self.as_string()
            except UnicodeDecodeError:
                # Fallback to bytes
                return self.as_bytes()

    def as_message(self) -> Message:
        return Message.from_bytes(self.raw)

    def as_string(self) -> str:
        return self.raw.decode('utf-8')

    def as_bytes(self) -> bytes:
        return self.raw

    def as_packed_varint(self) -> tuple[int, ...]:
        unpacked = []
        cursor = BytesCursor(self.raw)
        while cursor:
            unpacked.append(cursor.read_varint())
        return tuple(unpacked)

    def as_packed_fixed(self, fmt) -> tuple[int | float, ...]:
        return struct.unpack(fmt, self.raw)


class FixedNBitField(Field, ABC):
    """:class:`Field` of a :class:`Message` with wire type 1 (64-bit) or 5 (32-bit)."""
    raw: bytes

    def auto(self) -> int:
        return self.as_signed()

    @abstractmethod
    def as_signed(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def as_unsigned(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def as_float(self) -> float:
        raise NotImplementedError


class Fixed32BitField(FixedNBitField):
    """:class:`Field` of a :class:`Message` with wire type 5 (32-bit)."""
    raw: bytes

    def as_signed(self) -> int:
        return struct.unpack('<l', self.raw)[0]  # long

    def as_unsigned(self) -> int:
        return struct.unpack('<L', self.raw)[0]  # unsigned long

    def as_float(self) -> float:
        return struct.unpack('<f', self.raw)[0]  # float


class Fixed64BitField(FixedNBitField):
    """:class:`Field` of a :class:`Message` with wire type 1 (64-bit)."""
    raw: bytes

    def as_signed(self) -> int:
        return struct.unpack('<q', self.raw)[0]  # long long

    def as_unsigned(self) -> int:
        return struct.unpack('<Q', self.raw)[0]  # unsigned long long

    def as_float(self) -> float:
        return struct.unpack('<d', self.raw)[0]  # double


class GroupStartField(Field):
    """:class:`Field` of a :class:`Message` with wire type 3 (start group)."""
    raw: None

    def auto(self):
        return 'group_start'


class GroupEndField(Field):
    """:class:`Field` of a :class:`Message` with wire type 4 (end group)."""
    raw: None

    def auto(self):
        return 'group_end'


@dataclasses.dataclass(frozen=True)
class FieldCollection(Convertible):
    """Wrapper that contains zero or more :class:`Field` objects. If it contains zero or one Field,
    the methods of the :class:`Convertible` interface can be used to convert it to another type."""
    fields: Sequence[Field]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.fields!r})'

    def __iter__(self) -> Iterator[Field]:
        """Returns an iterator over each :class:`Field` in the collection."""
        return (field for field in self.fields)

    def __getitem__(self, tag: int) -> FieldCollection:
        """If the collection contains only a single :class:`Field` that can be interpreted as a :class:`Message`,
        returns any nested :class:`Field` objects with the specified tag. Otherwise, raises :class:`ValueError`."""
        if len(self.fields) == 0:
            return FieldCollection([])
        elif len(self.fields) == 1:
            return self.fields[0].as_message()[tag]
        else:
            raise ValueError(f'expected 0 or 1 element with tag {tag}, but found {len(self.fields)}')

    def auto(self) -> int:
        if len(self.fields) == 1:
            return self.fields[0].auto()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with 0 or more than 1 element automatically')

    def as_varint(self) -> int:
        if len(self.fields) == 0:
            return 0
        elif len(self.fields) == 1:
            return self.fields[0].as_varint()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to varint')

    def as_zigzag(self) -> int:
        if len(self.fields) == 0:
            return 0
        elif len(self.fields) == 1:
            return self.fields[0].as_zigzag()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to zigzag')

    def as_bool(self) -> bool:
        if len(self.fields) == 0:
            return False
        elif len(self.fields) == 1:
            return self.fields[0].as_bool()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to bool')

    def as_enum(self, enum: Callable[[int], E]) -> E:
        if len(self.fields) == 0:
            return enum(0)
        elif len(self.fields) == 1:
            return self.fields[0].as_enum(enum)
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to enum')

    def as_message(self) -> Message:
        if len(self.fields) == 0:
            return Message.from_bytes(b'')
        elif len(self.fields) == 1:
            return self.fields[0].as_message()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to Message')

    def as_string(self) -> str:
        if len(self.fields) == 0:
            return ''
        elif len(self.fields) == 1:
            return self.fields[0].as_string()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to str')

    def as_bytes(self) -> bytes:
        if len(self.fields) == 0:
            return b''
        elif len(self.fields) == 1:
            return self.fields[0].as_bytes()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to bytes')

    def as_packed_varint(self) -> tuple[int, ...]:
        if len(self.fields) == 0:
            return tuple()
        elif len(self.fields) == 1:
            return self.fields[0].as_packed_varint()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to packed varint')

    def as_packed_fixed(self, fmt) -> tuple[int | float, ...]:
        if len(self.fields) == 0:
            return tuple()
        elif len(self.fields) == 1:
            return self.fields[0].as_packed_fixed(fmt)
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to packed fixed')

    def as_signed(self) -> int:
        if len(self.fields) == 0:
            return 0
        elif len(self.fields) == 1:
            return self.fields[0].as_signed()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to signed')

    def as_unsigned(self) -> int:
        if len(self.fields) == 0:
            return 0
        elif len(self.fields) == 1:
            return self.fields[0].as_unsigned()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to unsigned')

    def as_float(self) -> float:
        if len(self.fields) == 0:
            return 0.0
        elif len(self.fields) == 1:
            return self.fields[0].as_float()
        else:
            raise ValueError(f'cannot convert {self.__class__.__name__} with more than 1 element to float')


class JSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any | None:
        if isinstance(o, bytes):
            return base64.standard_b64encode(o).decode('ascii')
        return super().default(o)


class Message(FieldCollection):
    """A protocol buffer message."""
    def __getitem__(self, tag: int) -> FieldCollection:
        """Returns any :class:`Field` objects with the specified tag."""
        return FieldCollection([field for field in self.fields if field.tag == tag])

    @classmethod
    def from_cursor(cls, cursor: BytesCursor) -> Message:
        """Returns a :class:`Message` with the :class:`Field` objects read from the provided :class:`BytesCursor`."""
        try:
            fields = []
            while cursor:
                fields.append(Field.from_cursor(cursor))
            return cls(fields)
        except (IndexError, ValueError):
            raise ValueError('invalid protobuf')

    @classmethod
    def from_bytes(cls, data: bytes) -> Message:
        """Returns a :class:`Message` with the :class:`Field` objects read from the provided ``data``."""
        return cls.from_cursor(BytesCursor(data))

    def as_dict(self, *, recursive: bool = True) -> dict[int, Any]:
        """Converts the message into a :class:`dict` by calling the :func:`Field.auto()` method of each :class:`Field`.

        :param recursive: whether to recursively convert nested messages into dicts
        :return: dict with the corresponding fields"""
        output: dict[int, Any] = {}
        for field in self.fields:
            value: Any = field.auto()
            if isinstance(value, Message):
                if recursive:
                    value = value.as_dict(recursive=recursive)
                else:
                    value = field.as_bytes()

            if field.tag not in output:
                output[field.tag] = value
            elif field.tag in output and not isinstance(output[field.tag], list):
                output[field.tag] = [output[field.tag], value]
            else:
                output[field.tag].append(value)
        return output

    def as_json(self, *, recursive: bool = True, default=JSONEncoder, **kwargs) -> str:
        """Converts the message into JSON. Additional parameters are passed to :func:`json.dumps()`.

        :param recursive: whether to recursively convert nested messages into JSON objects
        :param default: defaults to a custom JSONEncoder
        :return: str with the object serialized as JSON"""
        return json.dumps(self.as_dict(recursive=recursive), default=default, **kwargs)
