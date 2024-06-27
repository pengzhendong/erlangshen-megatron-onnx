# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
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

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from modelscope import snapshot_download


class BertONNX:
    def __init__(
        self,
        model_id="pengzhendong/chinese-roberta-wwm-ext-large-onnx",
        providers=["CPUExecutionProvider"],
    ):
        repo_dir = snapshot_download(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(repo_dir)
        self.session = ort.InferenceSession(
            f"{repo_dir}/model.onnx", providers=providers
        )

    def compute(self, text):
        inputs = self.tokenizer(text, return_tensors="np")
        tokens = inputs.tokens()
        onnx_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"],
        }
        return tokens, self.session.run(None, onnx_inputs)
