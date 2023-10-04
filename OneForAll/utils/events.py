"""utils/events.py
"""
import datetime
import json
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional
from fvcore.common.history_buffer import HistoryBuffer

from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.utils.file_io import PathManager


class CommonMetricSacredWriter(EventWriter):
    """CommonMetricSacredWriter
    """
    def 