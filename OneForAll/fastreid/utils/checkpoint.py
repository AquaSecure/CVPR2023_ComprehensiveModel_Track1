
# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import logging
import os
from collections import defaultdict
from typing import Any
from typing import Optional, List, Dict, NamedTuple, Tuple, Iterable
from subprocess import Popen

import numpy as np
import paddle
from termcolor import colored

from fastreid.utils.file_io import PathManager


class _IncompatibleKeys(
    NamedTuple(
        # pyre-fixme[10]: Name `IncompatibleKeys` is used but not defined.
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
            ("incorrect_shapes", List[Tuple]),
        ],
    )
):
    pass


class Checkpointer(object):
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.