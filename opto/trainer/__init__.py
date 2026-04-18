"""Trainer package public facade."""

from importlib import import_module

__all__ = ["train"]


def __getattr__(name):
	if name == "train":
		return import_module("opto.trainer.train").train
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")