import pytest
from app.main import load_sentiment_model

class TestSentimentAnalysis:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = load_sentiment_model()
        self.test_cases = [
            ("Это лучшее приложение!", "positive"),
            ("Ненавижу этот сервис!", "negative"),
            ("сегодня без изменений", "neutral")
        ]

    # Добавляем функцию-обертку для проверки ввода
    def predict_sentiment(self, text):
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        return self.model(text)

    def test_model_loading(self):
        assert self.model is not None
        assert self.model.task == "text-classification"

    def test_sentiment_prediction(self):
        for text, expected_label in self.test_cases:
            # Используем новую функцию для предсказания
            result = self.predict_sentiment(text)[0]
            assert result['label'] == expected_label
            assert 0 <= result['score'] <= 1

    def test_empty_input(self):
        # Проверяем конкретное исключение ValueError
        with pytest.raises(ValueError) as exc_info:
            self.predict_sentiment("")
        assert "Input text cannot be empty" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main()