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

import click

from bert_onnx import BertONNX


@click.command()
@click.argument("text")
def main(text):
    model = BertONNX()
    tokens, outputs = model.compute(text)
    _, hidden_states, _ = outputs[0], outputs[1], outputs[2]
    print(tokens)
    print(hidden_states[-3][0])  # [num_layers, batch_size]


if __name__ == "__main__":
    main()
