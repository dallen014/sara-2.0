import unittest
from src.interpret import explain_prediction


class TestInterpret(unittest.TestCase):
    def setUp(self):
        """Example ticket data for testing."""
        self.subject = "   Billing issue:   duplicate   charge   "
        self.body = """
            Hello Support,

            I was   charged   twice on my last   invoice.
            The total on my   credit card   doesn't match the receipt.

            Please  refund   the   extra  charge   and update my billing address:
            123   Main  St,   Springfield   

            You can   reach me at   (555)   123-4567
            or   john.doe@example.com

            Thanks,
            Alex  Smith
            """

    def test_explain_prediction_returns_list(self):
        explanation = explain_prediction(self.subject, self.body, num_features=5)

        # Check type and non-empty
        self.assertIsInstance(explanation, list, "Output should be a list")
        self.assertGreater(len(explanation), 0, "Explanation should not be empty")

        # Check each element is a (word, weight) tuple
        for word, weight in explanation:
            self.assertIsInstance(word, str, "Word should be a string")
            self.assertIsInstance(weight, float, "Weight should be a float")

        # Print explanation for manual inspection
        print("\nLIME Explanation Output:")
        for word, weight in explanation:
            print(f"{word}: {weight:.4f}")


if __name__ == "__main__":
    unittest.main()
