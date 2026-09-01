import json
import tempfile
import unittest
from pathlib import Path

from IndicFinetuning.single_language_curriculum.curriculum import (
    CurriculumSample,
    assert_disjoint,
    build_stage_manifests,
    cumulative_interval_step,
    deduplicate,
    expressive_clean_subset,
    filter_ivr,
    parse_optional_float,
    split_ivr_speaker_disjoint,
    split_rasa_stratified,
    write_jsonl,
)
from IndicFinetuning.single_language_curriculum.export_hf_catalogs import (
    metadata_passes_ivr_filters,
    normalized_gender,
)
from IndicFinetuning.single_language_curriculum.config import MalayalamCurriculumConfig


def sample(source, index, speaker, **overrides):
    values = dict(
        id=f"{source}_{index}",
        source=source,
        text=f"text {source} {index}",
        audio_path=f"/{source}/{index}.wav",
        speaker_id=speaker,
        gender="female" if index % 2 else "male",
        duration=5.0,
        scenario="Extempore",
        cer=0.03,
        snr=30.0,
        c50=25.0,
        pitch_std=float(index % 10 + 1),
        speaking_rate=15.0,
    )
    values.update(overrides)
    return CurriculumSample(**values)


class CurriculumPlanTests(unittest.TestCase):
    def test_audio_interval_is_cumulative_across_stage_boundaries(self):
        self.assertEqual(cumulative_interval_step(250, 1750, 1000), 2000)
        self.assertIsNone(cumulative_interval_step(249, 1750, 1000))

    def test_default_run_is_one_curriculum_epoch_with_audio_every_1000_steps(self):
        config = MalayalamCurriculumConfig()
        self.assertTrue(all(stage.epochs == 1.0 for stage in config.stages))
        self.assertEqual(config.audio_sample_steps, 1000)
        self.assertTrue(config.audio_samples_on_steps)
        self.assertEqual((config.batch_size, config.grad_accum), (8, 2))

    def test_hf_metadata_normalization_and_filtering(self):
        class Args:
            ivr_min_duration = 2.0
            ivr_max_duration = 20.0
            max_ivr_cer = 0.15
            min_ivr_snr = 20.0
            min_ivr_c50 = 10.0

        schema = {"duration": "duration", "cer": "cer", "snr": "snr", "c50": "c50"}
        row = {"duration": 8.0, "cer": "tensor(0.0387)", "snr": 30.0, "c50": 25.0}
        self.assertTrue(metadata_passes_ivr_filters(row, schema, Args()))
        self.assertEqual(normalized_gender("Female"), "female")

    def test_parse_tensor_style_cer(self):
        self.assertEqual(parse_optional_float("tensor(0.0387)"), 0.0387)
        self.assertIsNone(parse_optional_float("unknown"))

    def test_filters_and_deduplicates(self):
        rows = [
            sample("ivr", 1, "a"),
            sample("ivr", 2, "a", text="text ivr 1"),
            sample("ivr", 3, "b", snr=10.0),
            sample("ivr", 4, "b", cer=0.2),
        ]
        self.assertEqual([row.id for row in filter_ivr(deduplicate(rows))], ["ivr_1"])

    def test_same_transcript_from_different_speakers_is_preserved(self):
        rows = [
            sample("rasa", 1, "female", text="shared sentence"),
            sample("rasa", 2, "male", text="shared sentence"),
        ]
        self.assertEqual(len(deduplicate(rows)), 2)

    def test_rasa_split_keeps_each_speaker_represented(self):
        rows = [sample("rasa", index, f"speaker_{index % 2}") for index in range(200)]
        splits = split_rasa_stratified(rows, seed=42)
        assert_disjoint(splits, speaker_disjoint=False)
        self.assertEqual({len(value) for value in splits.values()}, {10, 180})
        for split_rows in splits.values():
            self.assertEqual({row.speaker_id for row in split_rows}, {"speaker_0", "speaker_1"})

    def test_ivr_split_is_speaker_disjoint(self):
        rows = [sample("ivr", index, f"speaker_{index // 10}") for index in range(200)]
        splits = split_ivr_speaker_disjoint(rows, seed=42)
        assert_disjoint(splits, speaker_disjoint=True)
        self.assertEqual(sum(map(len, splits.values())), len(rows))

    def test_fixed_stage_source_ratios(self):
        rasa = [sample("rasa", index, f"r{index % 2}") for index in range(33)]
        ivr = [sample("ivr", index, f"i{index % 10}") for index in range(31)]
        expressive = expressive_clean_subset(ivr)
        stages = build_stage_manifests(rasa, ivr, expressive, seed=42)

        self.assertEqual({row.source for row in stages["stage1"]}, {"ivr"})
        stage2_rasa = sum(row.source == "rasa" for row in stages["stage2"])
        stage2_ivr = sum(row.source == "ivr" for row in stages["stage2"])
        self.assertEqual(stage2_rasa, stage2_ivr)
        stage3_rasa = sum(row.source == "rasa" for row in stages["stage3"])
        stage3_ivr = sum(row.source == "ivr" for row in stages["stage3"])
        self.assertEqual(stage3_rasa, 33)
        self.assertEqual(stage3_ivr, 9)

    def test_jsonl_is_utf8_and_round_trippable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            row = sample("rasa", 1, "speaker", text="ഇത് മലയാളം")
            write_jsonl(path, [row])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["text"], "ഇത് മലയാളം")


if __name__ == "__main__":
    unittest.main()
