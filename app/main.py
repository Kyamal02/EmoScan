import streamlit as st
from transformers import pipeline
import torch


@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="cointegrated/rubert-tiny-sentiment-balanced",
        device_map="auto" if torch.cuda.is_available() else None,
        max_length=512,
        truncation=True
    )


def main():
    st.title("📝 Анализатор тональности русского текста")
    st.write("Введите текст для анализа эмоциональной окраски (нейтральный/позитивный/негативный)")

    try:
        classifier = load_sentiment_model()
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        return

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

                # Для отображения распределения вероятностей
                all_scores = classifier(text, return_all_scores=True)[0]
                probabilities = {item['label']: item['score'] for item in all_scores}
#
                st.write("Распределение вероятностей:")
                cols = st.columns(3)
                for col, (label, prob) in zip(cols, probabilities.items()):
                    with col:
                        st.write(f"{label.capitalize()}")
                        st.progress(prob, text=f"{prob * 100:.1f}%")
                        st.caption(f"{emoji_dict.get(label, '')} {prob * 100:.1f}%")

            except Exception as e:
                st.error(f"Произошла ошибка: {str(e)}")
    elif not text:
        st.warning("Пожалуйста, введите текст для анализа")


if __name__ == "__main__":
    main()