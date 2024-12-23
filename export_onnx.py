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

from pathlib import Path
from typing import Mapping, OrderedDict

import torch
from modelscope import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.onnx import OnnxConfig, export


class BertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("last_hidden_state", {0: "batch", 1: "sequence_length"}),
                ("hidden_states", {0: "batch", 2: "sequence_length"}),
                (
                    "attentions",
                    {0: "batch", 3: "sequence_length", 4: "sequence_length"},
                ),
            ]
        )


class ExportModel(PreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.model(input_ids, attention_mask, token_type_ids)
        return {
            "last_hidden_state": outputs["last_hidden_state"],
            "hidden_states": torch.stack(list(outputs["hidden_states"])),
            "attentions": torch.stack(list(outputs["attentions"])),
        }

    def call(self, input_ids=None, attention_mask=None, token_type_ids=None):
        self.forward(input_ids, attention_mask, token_type_ids)


def main():
    repo_dir = snapshot_download("pengzhendong/Erlangshen-MegatronBert-1.3B")
    config = AutoConfig.from_pretrained(repo_dir)
    config.output_attentions = True
    config.output_hidden_states = True
    model = AutoModel.from_pretrained(repo_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(repo_dir)

    model = ExportModel(config, model)
    config = BertOnnxConfig(config)
    onnx_path = Path("onnx/model.onnx")
    if not onnx_path.parent.exists():
        onnx_path.parent.mkdir(parents=True)
    onnx_inputs, onnx_outputs = export(tokenizer, model, config, config.default_onnx_opset, onnx_path)


if __name__ == "__main__":
    main()
