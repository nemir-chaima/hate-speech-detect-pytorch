from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import pandas as pd

#st.markdown(
  #      """
  #      <h1 style='text-align: center;'>Welcome to 'No Toxic Messages App'</h1>
  #      """,
  #      unsafe_allow_html=True  # Permet l'utilisation de HTML dans Streamlit
  #  )
#st.image('no_hate.png')

#torch.cuda.empty_cache()

#if 'messages' not in st.session_state:
 #   st.session_state.messages =[]

#for message in st.session_state.messages:

#    with st.chat_message(message['role']):
#        st.markdown(message['content'])
#        st.markdown(f"{message['prediction']}")
        
model_id = "meta-llama/Meta-Llama-Guard-2-8B"

device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
dic_classes = {'1':'Violent_crimes','2':'Non-violent Crimes',
               '3': 'Sex-Related Crimes', '4': 'Child Sexual Exploitation',
               '5': 'Specialized Advice','6':'Privacy','7':'Intellectual Property',
               '8': 'Indiscriminate Xeapons', '9':'Hate',
               '10': 'Suicide & Self-harm', "11": 'Sexual Content'}

#prompt = st.chat_input('Saisir un message ..')
#if prompt:
#results = []
for i in range(15):
    text= input("Saisir le message : ")
    predictions = moderate([{"role": "user", "content": text }])


    
    if predictions[0]=='u':
            #print(predictions[8])
            print(' Message innaproprié')
            classe_predite = dic_classes[predictions[8]]
            print("il est classé comme :", classe_predite)
            #results.append((text,' Message innaproprié',classe_predite))
    else :
            print('Votre message est clean')
            #results.append((text,' Message clean', 'clean'))

#result_df = pd.DataFrame(results, columns=['text', 'resulat','classe'])
#result_df.to_csv('resultat_llama.csv', index=False)