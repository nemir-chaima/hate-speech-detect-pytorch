from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

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

#prompt = st.chat_input('Saisir un message ..')
#if prompt:
text= input("saisir le message : ")
predictions = moderate([{"role": "user", "content": text }])

#st.session_state.messages.append({'role':'user', 'content': prompt, 'prediction':f'{ predictions}'})

    #st.write(predictions)
print(predictions)


print(predictions[7])
print(type(predictions))