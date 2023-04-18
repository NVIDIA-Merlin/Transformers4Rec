from merlin.schema import Schema
from merlin.schema.io.tensorflow_metadata import TensorflowMetadata

from transformers4rec.torch.utils import torch_utils as tu

SCHEMA = """
{
  "feature": [
    {
      "name": "session_id",
      "type": "INT",
      "intDomain": {
        "name": "session_id",
        "max": "8523",
        "isCategorical": true
      },
      "annotation": {
        "tag": [
          "categorical"
        ],
        "extraMetadata": [
          {
            "num_buckets": null,
            "freq_threshold": 0,
            "max_size": 0,
            "start_index": 0,
            "cat_path": ".//categories/unique.session_id.parquet",
            "embedding_sizes": {
              "cardinality": 8524,
              "dimension": 254
            },
            "_dims": [
              [
                0,
                null
              ]
            ],
            "is_list": false,
            "is_ragged": false,
            "dtype_item_size": 64
          }
        ]
      }
    },
    {
      "name": "item_id-list",
      "valueCount": {
        "min": "10",
        "max": "10"
      },
      "type": "INT",
      "intDomain": {
        "name": "item_id",
        "max": "293",
        "isCategorical": true
      },
      "annotation": {
        "tag": [
          "list",
          "id",
          "item_id",
          "item",
          "categorical"
        ],
        "extraMetadata": [
          {
            "num_buckets": null,
            "freq_threshold": 0,
            "max_size": 0,
            "start_index": 0,
            "cat_path": ".//categories/unique.item_id.parquet",
            "embedding_sizes": {
              "cardinality": 294,
              "dimension": 39
            },
            "_dims": [
              [
                0,
                null
              ],
              10
            ],
            "is_list": true,
            "is_ragged": false,
            "dtype_item_size": 64
          }
        ]
      }
    },
    {
      "name": "category-list",
      "valueCount": {
        "min": "10",
        "max": "10"
      },
      "type": "INT",
      "intDomain": {
        "name": "category",
        "max": "180",
        "isCategorical": true
      },
      "annotation": {
        "tag": [
          "list",
          "categorical"
        ],
        "extraMetadata": [
          {
            "num_buckets": null,
            "freq_threshold": 0,
            "max_size": 0,
            "start_index": 0,
            "cat_path": ".//categories/unique.category.parquet",
            "embedding_sizes": {
              "cardinality": 181,
              "dimension": 29
            },
            "_dims": [
              [
                0,
                null
              ],
              10
            ],
            "is_list": true,
            "is_ragged": false,
            "dtype_item_size": 64
          }
        ]
      }
    },
    {
      "name": "click",
      "type": "INT",
      "annotation": {
        "tag": [
          "target"
        ],
        "extraMetadata": [
          {
            "_dims": [
              [
                0,
                null
              ]
            ],
            "is_list": false,
            "is_ragged": false,
            "dtype_item_size": 64
          }
        ]
      }
    },
    {
      "name": "day-first",
      "type": "INT",
      "annotation": {
        "tag": [
          "categorical"
        ],
        "extraMetadata": [
          {
            "_dims": [
              [
                0,
                null
              ]
            ],
            "is_list": false,
            "is_ragged": false,
            "dtype_item_size": 64
          }
        ]
      }
    }
  ]
}
"""


def test_get_output_sizes_from_schema():
    schema: Schema = TensorflowMetadata.from_json(SCHEMA).to_merlin_schema()

    sizes = tu.get_output_sizes_from_schema(schema)

    assert sizes["session_id"] == (-1,)
    assert sizes["item_id-list"] == (-1, 10)
    assert sizes["category-list"] == (-1, 10)
    assert sizes["click"] == (-1,)
    assert sizes["day-first"] == (-1,)
