import streamlit as st 
import pandas as pd
st.title("Résultat de LLama guard modèl 🦙")


st.subheader("Testing : ")
st.video('/Users/chaimanemir/Desktop/detection_pytorch/hate-speech-detect-pytorch/st_/pages/for_st_app.mp4')

st.subheader("Résultat : ")
resultat = pd.read_csv("/Users/chaimanemir/Desktop/detection_pytorch/hate-speech-detect-pytorch/st_/pages/resultat_llama.csv")
st.write(resultat)