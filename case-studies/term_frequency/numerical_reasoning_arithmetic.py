# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""

import datasets


# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\

"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
    Generated dataset for testing numerical reasoning
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

class NumericalReasoningArithmetic(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="{}".format(num),
            version=datasets.Version("0.1.0"),
            description="Task with {} as the first operand".format(num)) \
                for num in range(0,100)
    ]

    DEFAULT_CONFIG_NAME = None

    def _info(self):
        features = datasets.Features(
            {
                "x1": datasets.Value("int32"),
                "x2": datasets.Value("int32"),
                "y_mul": datasets.Value("int32"),
                "y_add": datasets.Value("int32"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "validation"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test"
                },
            ),
        ]

    def _generate_examples(self, split):

        x1 = int(self.config.name)
        if split == "validation":
            key = -1
            for x in list(range(0,x1))+list(range(x1+1,100)):
                for x2 in range(1,51):
                    key += 1
                    yield key, {
                        "x1": x,
                        "x2": x2,
                        "y_mul": x*x2,
                        "y_add": x+x2,
                    }

        elif split == "test":
            for key, x2 in enumerate(range(1,51)):
                yield key, {
                    "x1": x1,
                    "x2": x2,
                    "y_mul": x1*x2,
                    "y_add": x1+x2,
                }
