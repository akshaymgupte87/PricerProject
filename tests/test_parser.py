import unittest

from pricer.parser import get_weight, parse


class WeightParsingTests(unittest.TestCase):
    def test_accepts_singular_and_plural_pounds(self):
        self.assertEqual(get_weight({"Item Weight": "1 pound"}), 1.0)
        self.assertEqual(get_weight({"Item Weight": "2 pounds"}), 2.0)

    def test_accepts_common_abbreviations(self):
        self.assertEqual(get_weight({"Item Weight": "16 oz"}), 1.0)
        self.assertEqual(get_weight({"Item Weight": "1 lb."}), 1.0)

    def test_accepts_hundredths_of_pounds(self):
        self.assertEqual(get_weight({"Item Weight": "25 hundredths pounds"}), 0.25)
        self.assertEqual(get_weight({"Item Weight": "25 hundredths of pounds"}), 0.25)

    def test_malformed_weights_return_zero(self):
        malformed = [
            None,
            {},
            {"Item Weight": "bad pounds"},
            {"Item Weight": "1 hundredths"},
        ]
        for details in malformed:
            with self.subTest(details=details):
                self.assertEqual(get_weight(details), 0.0)


class DetailsParsingTests(unittest.TestCase):
    @staticmethod
    def datapoint(details):
        return {
            "price": "10",
            "title": "x" * 700,
            "description": [],
            "features": [],
            "details": details,
        }

    def test_invalid_or_null_details_do_not_reject_an_otherwise_valid_item(self):
        for details in (None, "not-json", "[]"):
            with self.subTest(details=details):
                item = parse(self.datapoint(details), "test")
                self.assertIsNotNone(item)
                self.assertEqual(item.weight, 0.0)


if __name__ == "__main__":
    unittest.main()
