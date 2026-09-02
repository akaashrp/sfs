from scripts.prep.run_qwen_bucketed_prompts import extract_prompt_metadata


def test_raw_bucket_metadata_is_preserved():
    record = {
        "prompt": "Question",
        "example_id": "dataset:train:7",
        "bucket": "alpaca",
    }

    assert extract_prompt_metadata(record) == {
        "example_id": "dataset:train:7",
        "bucket": "alpaca",
    }


def test_holdout_cache_metadata_is_not_nested_again():
    record = {
        "prompt": "Question",
        "request_id": "holdout-alpaca-02500",
        "prompt_index": 2500,
        "prompt_metadata": {
            "example_id": "dataset:train:2500",
            "dataset_id": "dataset",
        },
    }

    assert extract_prompt_metadata(record) == {
        "example_id": "dataset:train:2500",
        "dataset_id": "dataset",
    }
