"""Instruction parsing: turn a prompt into an (object, source, destination) triple."""

from .instruction_parser import ParsedInstruction, parse_instruction, parse_instructions

__all__ = ["ParsedInstruction", "parse_instruction", "parse_instructions"]
