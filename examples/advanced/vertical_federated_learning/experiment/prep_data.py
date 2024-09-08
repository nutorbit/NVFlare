# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shutil

import numpy as np
import pandas as pd


def main():
    data_path = "./data/bank.csv"
    output_path="/tmp/nvflare/vertical_xgb_data"
    output_file="bank.data.csv"
    partitions = [
        ["age", "job", "balance"],
        ["marital", "education", "default", "y"]
    ]
    
    df = pd.read_csv(data_path, sep=";")
    df["uid"] = df.index.to_series().map(lambda x: "uid_" + str(x))

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    for site in range(len(partitions)):
        df_split = df[partitions[site] + ["uid"]]
        df_split = df_split.sample(frac=1)
        print(f"site-{site+1} split cols {partitions[site]}")

        data_path = os.path.join(output_path, f"site-{site + 1}")
        if not os.path.exists(data_path):
            os.makedirs(data_path, exist_ok=True)

        df_split.to_csv(path_or_buf=os.path.join(data_path, output_file), index=False)


if __name__ == "__main__":
    main()