# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""
Functions for downloading pre-trained Sana models
"""
import argparse
import os

import torch
from termcolor import colored
from torchvision.datasets.utils import download_url

from .hf_utils import hf_download_or_fpath

pretrained_models = {}


def find_model(model_name):
    """
    Finds a pre-trained G.pt model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if not model_name:
        raise ValueError("model_name cannot be None")

    if model_name in pretrained_models:  # Find/download our pre-trained G.pt checkpoints
        return download_model(model_name)

    # Load a custom Sana checkpoint:
    model_path = hf_download_or_fpath(model_name)
    if not model_path:
        raise ValueError(f"Failed to download or find model: {model_name}")
        
    assert os.path.isfile(model_path), f"Could not find Sana checkpoint at {model_path}"
    print(colored(f"[Sana] Loading model from {model_path}", attrs=["bold"]))
    return torch.load(model_path, map_location=lambda storage, loc: storage)


def download_model(model_name):
    """
    Downloads a pre-trained Sana model from the web.
    """
    if not model_name:
        raise ValueError("model_name cannot be None")
        
    assert model_name in pretrained_models
    local_path = f"output/pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        os.makedirs("output/pretrained_models", exist_ok=True)
        web_path = f""  # Add your actual web path here
        download_url(web_path, "output/pretrained_models/")
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_names", nargs="+", type=str, default=pretrained_models)
    args = parser.parse_args()
    model_names = set(args.model_names)

    # Download Sana checkpoints
    for model in model_names:
        download_model(model)
    print("Done.")