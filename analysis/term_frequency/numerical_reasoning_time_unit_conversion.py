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

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
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
                "x": datasets.Value("int32"),
                "y_min_sec": datasets.Value("int32"),
                "y_hour_min": datasets.Value("int32"),
                "y_day_hour": datasets.Value("int32"),
                "y_week_day": datasets.Value("int32"),
                "y_month_week": datasets.Value("int32"),
                "y_year_month": datasets.Value("int32"),
                "y_decade_year": datasets.Value("int32"),
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

        x = int(self.config.name)
        if split == "validation":
            for key, _x in enumerate(list(range(0,x))+list(range(x+1,100))):
                yield key, {
                    "x": _x,
                    "y_min_sec": _x*60,
                    "y_hour_min": _x*60,
                    "y_day_hour": _x*24,
                    "y_week_day": _x*7,
                    "y_month_week": _x*4,
                    "y_year_month": _x*12,
                    "y_decade_year": _x*10,
                    }

        elif split == "test":
            # Because it's only 1 datapoint, 
            # let's do it over 10 times to get a better distribution
            for key in range(0,10):
                yield key, {
                    "x": x,
                    "y_min_sec": x*60,
                    "y_hour_min": x*60,
                    "y_day_hour": x*24,
                    "y_week_day": x*7,
                    "y_month_week": x*4,
                    "y_year_month": x*12,
                    "y_decade_year": x*10,
                    }
