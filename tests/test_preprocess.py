import unittest
import numpy as np
from src.preprocess import (
    redact_pii,
    clean_text,
    combine_subject_body,
    preprocess_and_transform,
)


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        """Run before each test."""
        self.subject = "Reset Password Request"
        self.body = (
            "Hi Support,\n\n"
            "I can't log in to my account. "
            "My email is johndoe@gmail.com and my phone is 555-123-4567. "
            "My IP is 192.168.0.1. "
            "My credit card is 4111 1111 1111 1111. "
            "My address is 123 Main St. "
            "Thanks, John Doe"
        )

    def test_redact_pii(self):
        """Test that redact_pii removes common sensitive patterns."""
        text = f"{self.subject} {self.body}"
        redacted = redact_pii(text)

        # Validate each pattern
        self.assertIn("[EMAIL_REDACTED]", redacted, "Email not redacted")
        self.assertIn("[PHONE_REDACTED]", redacted, "Phone not redacted")
        self.assertIn("[IP_REDACTED]", redacted, "IP not redacted")
        self.assertIn("[CREDIT_CARD_REDACTED]", redacted, "Credit card not redacted")
        self.assertIn("[ADDRESS_REDACTED]", redacted, "Address not redacted")
        self.assertIn("<NAME>", redacted, "Name not redacted")

    def test_clean_text(self):
        raw = "Hello   \n\n   World"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "Hello World")

    def test_combine_subject_body(self):
        combined = combine_subject_body(self.subject, self.body)
        self.assertIn("Reset Password Request", combined)
        self.assertIn("Hi Support", combined)

    def test_pipeline_returns_numpy_array(self):
        arr = preprocess_and_transform(self.subject, self.body)
        self.assertIsInstance(arr, np.ndarray)

    def test_pipeline_shape_matches_embedding_dim(self):
        arr = preprocess_and_transform(self.subject, self.body)
        # should be (1, embedding_dim), where embedding_dim ~768 for SBERT mpnet
        self.assertEqual(len(arr.shape), 2)
        self.assertGreater(arr.shape[1], 0)


if __name__ == "__main__":
    unittest.main()
