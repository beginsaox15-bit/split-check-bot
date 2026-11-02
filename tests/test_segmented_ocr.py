import time
import unittest
from unittest.mock import patch

from PIL import Image
import numpy as np
import cv2
from bot.services.receipt_preprocessing import rectify_receipt

from bot.services.receipt_ocr_layout import OCRLine, is_long_receipt, segment_bounds, merge_segments
from bot.services.tesseract_receipt_recognition import (
    OCRCandidate, choose_strategy, make_candidate, recognize_segments,
)
from bot.services.receipt_models import RecognitionUnavailable


def line(text, top, segment=0, margin=50):
    return OCRLine(text, top, top + 18, 90, 2, 2, 12, segment, margin)


class SegmentedOCRTests(unittest.TestCase):
    def test_narrow_long_receipt_can_be_cropped_from_wide_photo(self):
        photo = np.full((1400, 1400), 35, np.uint8)
        cv2.rectangle(photo, (600, 80), (800, 1320), 235, -1)
        receipt, applied = rectify_receipt(photo)
        self.assertTrue(applied)
        self.assertTrue(is_long_receipt((receipt.shape[1], receipt.shape[0])))

    def test_detection_depends_on_aspect_ratio(self):
        self.assertTrue(is_long_receipt((400, 1600)))
        self.assertTrue(is_long_receipt((1600, 6400)))
        self.assertFalse(is_long_receipt((1200, 1692)))

    def test_adaptive_segments_cover_all_pixels_and_overlap(self):
        with Image.new('L', (400, 2500), 255) as image:
            bounds = segment_bounds(image)
        self.assertGreater(len(bounds), 1)
        self.assertLessEqual(len(bounds), 12)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], 2500)
        self.assertTrue(all(right[0] < left[1] for left, right in zip(bounds, bounds[1:])))

    def test_line_on_boundary_kept_from_complete_neighbor(self):
        parts = [(0, 700, [OCRLine('Тартар 85', 690, 700, 50, 2, 2, 8, 0, 0)]),
                 (600, 1300, [line('Тартар 850.00', 690, 1)])]
        merged = merge_segments(parts)
        self.assertEqual([item.text for item in merged], ['Тартар 850.00'])

    def test_only_observation_at_boundary_is_not_lost(self):
        parts = [(0, 700, [OCRLine('Тартар 85', 690, 700, 50, 2, 2, 8, 0, 0)]),
                 (600, 1300, [])]
        self.assertEqual([item.text for item in merge_segments(parts)], ['Тартар 85'])

    def test_two_clipped_observations_are_not_both_deleted(self):
        parts = [(0, 700, [OCRLine('Чай 10', 680, 700, 50, 2, 2, 8, 0, 0)]),
                 (690, 1000, [OCRLine('Чай 100', 690, 710, 50, 2, 2, 8, 1, 0)])]
        self.assertTrue(merge_segments(parts))

    def test_near_identical_overlap_line_deduplicated(self):
        merged = merge_segments([(0, 700, [line('Лимонад 850.00', 650, margin=32)]),
                                 (600, 1300, [line('Лимонад! 850.00', 651, 1, 60)])])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, 'Лимонад! 850.00')

    def test_identical_real_goods_on_two_rows_remain_two(self):
        merged = merge_segments([
            (0, 700, [line('Кофе 100.00', 620), line('Кофе 100.00', 660)]),
            (600, 1300, [line('Кофе 100.00', 620, 1), line('Кофе 100.00', 660, 1)]),
        ])
        self.assertEqual(len(merged), 2)
        self.assertEqual([item.top for item in merged], [620, 660])

    def test_identical_text_outside_overlap_not_globally_deduplicated(self):
        merged = merge_segments([(0, 700, [line('Кофе 100.00', 200)]),
                                 (600, 1300, [line('Кофе 100.00', 900, 1)])])
        self.assertEqual(len(merged), 2)

    def test_different_numbers_not_fuzzily_combined(self):
        merged = merge_segments([(0, 700, [line('Кофе 100.00', 650)]),
                                 (600, 1300, [line('Кофе 1000.00', 651, 1)])])
        self.assertEqual(len(merged), 2)

    def test_russian_guest_line_order_preserved(self):
        parts = [(0, 700, [line('Гость 1', 100), line('Чай 330.00', 620)]),
                 (600, 1300, [line('Чай 330.00', 620, 1), line('Гость 2', 900, 1), line('Суп 350.00', 1000, 1)])]
        self.assertEqual([item.text for item in merge_segments(parts)], ['Гость 1', 'Чай 330.00', 'Гость 2', 'Суп 350.00'])

    def test_full_wins_when_better_or_tied(self):
        full = OCRCandidate('full', 'grayscale', 6, 100, True, 3)
        segmented = OCRCandidate('segmented', 'grayscale', 4, 90, True, 4, strategy='segmented')
        self.assertIs(choose_strategy(full, segmented), full)
        segmented.score = 100
        self.assertIs(choose_strategy(full, segmented), full)
        segmented.score = 100.9
        self.assertIs(choose_strategy(full, segmented), full)

    def test_segmented_wins_on_quality_not_character_count(self):
        good = [line('Бургер 850.00', 100), line('Чай 330.00', 150), line('ИТОГО 1180.00', 200)]
        full = make_candidate([OCRLine('мусор' * 200, 0, 10, 10, 0, 200, 0)], 'grayscale', 6)
        segmented = make_candidate(good, 'adaptive', 4)
        segmented.strategy = 'segmented'
        self.assertLess(len(segmented.text), len(full.text))
        self.assertIs(choose_strategy(full, segmented), segmented)

    def test_partial_segment_processing_cannot_replace_full(self):
        full = OCRCandidate('full', 'grayscale', 6, 100, True, 3)
        partial = OCRCandidate('partial', 'grayscale', 6, 1000, True, 10, complete=False)
        self.assertIs(choose_strategy(full, partial), full)
        full.complete = False
        self.assertIs(choose_strategy(full, partial), full)
        partial.complete = True
        self.assertIs(choose_strategy(full, partial), partial)
        with Image.new('L', (400, 1100), 255) as image:
            with patch('bot.services.tesseract_receipt_recognition.segment_bounds', return_value=[(0, 600), (500, 1100)]), \
                 patch('bot.services.tesseract_receipt_recognition.recognize_variants', side_effect=[full, RecognitionUnavailable()]):
                self.assertIsNone(recognize_segments(image, time.monotonic() + 10, (6, 4, 11)))
