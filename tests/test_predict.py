import unittest
from src.predict import predict_ticket


class TestPredict(unittest.TestCase):
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

    def test_predict_ticket_returns_list(self):
        preds = predict_ticket(self.subject, self.body)
        self.assertIsInstance(preds, list, "Output should be a list")

    def test_predict_ticket_top3_format(self):
        preds = predict_ticket(self.subject, self.body)

        # Print results for inspection
        print("\nPredicted Queues and Scores:")
        for label, score in preds:
            print(f"{label}: {score:.4f}")

        # Ensure exactly 3 predictions are returned
        self.assertEqual(len(preds), 3, "Should return top 3 predictions")

        for label, score in preds:
            self.assertIsInstance(label, str, "Label should be a string")
            self.assertIsInstance(score, float, "Score should be a float")
            self.assertGreaterEqual(score, 0.0, "Score should be >= 0.0")
            self.assertLessEqual(score, 1.0, "Score should be <= 1.0")

    def test_predict_ticket_sorted_scores(self):
        preds = predict_ticket(self.subject, self.body)
        scores = [score for _, score in preds]

        # Ensure scores are in descending order
        self.assertEqual(
            scores,
            sorted(scores, reverse=True),
            "Scores should be sorted in descending order",
        )


if __name__ == "__main__":
    unittest.main()
