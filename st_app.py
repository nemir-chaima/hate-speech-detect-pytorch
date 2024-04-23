import streamlit as st 
from notebook_pytorch import  trained_model
from notebook_pytorch import tokenizer
from notebook_pytorch import model



CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
trained_model = trained_model
trained_model.freeze()

test_example = st.text_input('Saisissez un message à envoyer', 'I dont like you, I hate your texts those are really bullshit!')
#test_example = "I dont like you, I hate your texts those are really bullshit!"


encoding = tokenizer.encode_plus(
    test_example,
    add_special_tokens=True,
    max_length=128,
    return_token_type_ids=False,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
)

model.eval()
_, preds = model(encoding["input_ids"], encoding["attention_mask"])
preds = preds.flatten().detach().numpy()

predictions = []
for idx, label in enumerate(CLASSES):
    if preds[idx] > 0.5:
        predictions.append((label, round(preds[idx]*100, 2)))

predictions