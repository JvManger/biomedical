# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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

"""
The DDI corpus has been manually annotated with drugs and pharmacokinetics and
pharmacodynamics interactions. It contains 1025 documents from two different
sources: DrugBank database and MedLine.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import kb_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks
from .bigbiohub import parse_brat_file
from .bigbiohub import brat_parse_to_bigbio_kb


_LANGUAGES = ['German']
_PUBMED = True
_LOCAL = True
_CITATION = """\
@article{HERREROZAZO2013914,
  title        = {
    The DDI corpus: An annotated corpus with pharmacological substances and
    drug-drug interactions
  },
  author       = {
    for test purposes
  },
  year         = 2023,
  journal      = {},
  volume       = ,
  number       = ,
  pages        = {},
  doi          = {},
  issn         = {},
  url          = {},
  keywords     = {}
}
"""

_DATASETNAME = "CustomGedisa"
_DISPLAYNAME = "Custom Gedisa"

_DESCRIPTION = """\
The DDI corpus has been manually annotated with drugs and pharmacokinetics and \
pharmacodynamics interactions. It contains 1025 documents from two different \
sources: DrugBank database and MedLine.
"""

#_HOMEPAGE = "https://github.com/isegura/DDICorpus"
_HOMEPAGE = ""
_LICENSE = 'Creative Commons Attribution Non Commercial 4.0 International'

_URLS = {
    _DATASETNAME: "D:\PythonProjekte\8000-Projektdaten\CustomGedisa/GedisaDDICorpus.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class DDICorpusDataset(datasets.GeneratorBasedBuilder):
    """DDI Corpus"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="CustomGedia_source",
            version=SOURCE_VERSION,
            description="Gedisa DDI Corpus source schema",
            schema="source",
            subset_id="CustomGedisa",
        ),
        BigBioConfig(
            name="CustomGedisa_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Gedisa DDI Corpus BigBio schema",
            schema="bigbio_kb",
            subset_id="CustomGedisa",
        ),
    ]

    DEFAULT_CONFIG_NAME = "CustomGedisa_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "offsets": datasets.Sequence(datasets.Value("int32")),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "head": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "tail": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "type": datasets.Value("string"),
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        standoff_dir = os.path.join(data_dir, "GedisaDDICorpus")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(standoff_dir, "Train"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(standoff_dir, "Test"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath: str, split: str) -> Tuple[int, Dict]:
        if self.config.schema == "source":
            for subdir, _, files in os.walk(filepath):
                for file in files:
                    # Ignore configuration files and annotation files - we just consider the brat text files
                    if not file.endswith(".txt"):
                        continue

                    brat_example = parse_brat_file(Path(subdir) / file)
                    source_example = self._to_source_example(brat_example)

                    yield source_example["document_id"], source_example

        elif self.config.schema == "bigbio_kb":
            for subdir, _, files in os.walk(filepath):
                for file in files:
                    # Ignore configuration files and annotation files - we just consider the brat text files
                    if not file.endswith(".txt"):
                        continue

                    # Read brat annotations for the given text file and convert example to the BigBio-KB format
                    brat_example = parse_brat_file(Path(subdir) / file)
                    kb_example = brat_parse_to_bigbio_kb(brat_example)
                    kb_example["id"] = kb_example["document_id"]

                    yield kb_example["id"], kb_example

    @staticmethod
    def _to_source_example(brat_example: Dict) -> Dict:
        source_example = {
            "document_id": brat_example["document_id"],
            "text": brat_example["text"],
            "relations": brat_example["relations"],
        }

        source_example["entities"] = []
        for entity_annotation in brat_example["text_bound_annotations"]:
            entity_ann = entity_annotation.copy()

            source_example["entities"].append(
                {
                    # These are lists in the parsed output, so just take the first element to
                    # match the source schema.
                    "offsets": entity_annotation["offsets"][0],
                    "text": entity_ann["text"][0],
                    "type": entity_ann["type"],
                    "id": entity_ann["id"],
                }
            )

        return source_example
