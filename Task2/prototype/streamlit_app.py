import streamlit as st
import requests

st.set_page_config(page_title="Cyclone Technical Assistant")
st.title("Cyclone Separator Technical Q&A")

API_URL = "http://localhost:8000/ask"

query = st.text_input("Ask a question about cyclone separator operation, maintenance, or safety:")

if st.button("Ask") and query:
    with st.spinner("Searching documents and generating answer..."):
        response = requests.post(API_URL, json={"question": query})
        result = response.json()

    st.subheader("Answer")
    st.write(result["answer"])
    st.caption(f"Status: {result['status']}")

    if result["citations"]:
        st.subheader("Sources")
        for doc, page in result["citations"]:
            st.write(f"- {doc}, page {page}")
            