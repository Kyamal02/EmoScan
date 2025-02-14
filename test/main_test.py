import pytest
from app.main import load_sentiment_model

#
class TestSentimentAnalysis:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = load_sentiment_model()
        self.test_cases = [
            ("Это лучшее приложение!", "positive"),
            ("Ненавижу этот сервис!", "negative"),
            ("сегодня без изменений", "neutral")
        ]
#
#
    def test_model_loading(self):
        assert self.model is not None
        assert self.model.task == "text-classification"

    def test_sentiment_prediction(self):
        for text, expected_label in self.test_cases:
            result = self.model(text)[0]
            assert result['label'] == expected_label
            assert 0 <= result['score'] <= 1

    def test_empty_input(self):
        with pytest.raises(Exception):
            self.model("")

if __name__ == "__main__":
    pytest.main()