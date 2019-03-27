# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""GQA scene graph evaluation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import os
import shutil
import uuid

from core.config import cfg
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import DEVKIT_DIR
from utils.io import save_object

logger = logging.getLogger(__name__)


def evaluate_boxes(
    gqa_dataset,
    all_boxes,
    output_dir,
    use_salt=True,
    cleanup=True,
    use_matlab=False
):
    result = gqa_eval_allinone(gqa_dataset, all_boxes, output_dir)
    return result


def gqa_eval_allinone(gqa_dataset, all_boxes, output_dir):
    """
    evaluate gqa result; this code is similar with the one of voc_eval
    """

    return None
