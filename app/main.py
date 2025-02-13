import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


@st.cache_resource
def load_sentiment_model():
    model_name = "SiberianNLP/rubert-base-cased-sentiment-rusentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def main():
    st.title("📝 Анализатор тональности русского текста")
    st.write("Введите текст для анализа эмоциональной окраски (позитивный/нейтральный/негативный)")

    classifier = load_sentiment_model()

    text = st.text_area("Введите ваш текст:", height=150)

    if st.button("Проанализировать") and text:
        with st.spinner("Анализируем..."):
            try:
                result = classifier(text)[0]
                label = result['label']
                score = result['score']

                emoji_dict = {
                    'neutral': "😐",
                    'positive': "😊",
                    'negative': "😠"
                }
                emoji = emoji_dict.get(label, "🤔")
                st.subheader(f"Результат: {label.capitalize()} {emoji}")

                st.metric("Уверенность модели", f"{score * 100:.1f}%")

                with torch.no_grad():
                    inputs = classifier.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                    outputs = classifier.model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

                st.write("Распределение вероятностей:")
                labels = ['neutral', 'positive', 'negative']
                probabilities = {label: probs[i].item() for i, label in enumerate(labels)}

                cols = st.columns(3)
                for col, (label, prob) in zip(cols, probabilities.items()):
                    with col:
                        st.write(f"{label.capitalize()}")
                        st.progress(prob, text=f"{prob * 100:.1f}%")
                        st.caption(f"{emoji_dict[label]} {prob * 100:.1f}%")

            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")
    elif not text:
        st.warning("Пожалуйста, введите текст для анализа")


if __name__ == "__main__":
    main()