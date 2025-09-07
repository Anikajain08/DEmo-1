import streamlit as st
import joblib
import os

def main():
    st.title("Fake News Detector")
    st.write("Enter a News to check whether it is Fake or Real: ")

    news_input = st.text_area("News Article:", "")

    if st.button("Check News"):
        if news_input.strip():
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                vectorizer_path = os.path.join(base_dir, "vectorizer.jb")
                model_path = os.path.join(base_dir, "lr_model.jb")
                vectorizer = joblib.load(vectorizer_path)
                model = joblib.load(model_path)
                transform_input = vectorizer.transform([news_input])
                prediction = model.predict(transform_input)
                if prediction[0] == 1:
                    st.success("THE NEWS IS REAL!!")
                else:
                    st.success("THE NEWS IS FAKE!!")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("PLEASE ENTER SOME TEXT TO ANALYZE.")

if __name__ == "__main__":
    main()
