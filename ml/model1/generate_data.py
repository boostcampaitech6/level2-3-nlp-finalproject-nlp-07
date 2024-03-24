from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch
import pickle
import pandas as pd
from tqdm import tqdm

# data = pd.read_csv('./song_data.csv')


def make_emotion(singer, song_name, tokenizer):
    conversation = [ {'role': 'system', 'content': "Respond with one of the following 13 emotions: ['nostalgia',  'love',   'excitement',  'anger',  'happiness',  'calmness',  'sadness',  'gratitude',  'loneliness',  'anticipation']. The format of the answer should be 'answer: emotion"},
                    {'role': 'user', 'content': f"Please tell me the emotion of the song '{song_name}' by '{singer}'."} ] 
    
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    return prompt

def use_model(prompt, model,tokenizer):
    max_length = 400
    generated_sequences = model(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        truncation= True,
    )
    return generated_sequences[0]["generated_text"].replace(prompt, "")


def main():

    data= pd.read_csV("Data's Directory")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    access_token ='Your Acess Token'
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=access_token)
    language_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token=access_token)

    
    text_generation_pipeline = TextGenerationPipeline(
        model=language_model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=device,
    )

    emotions = []
    for i in tqdm(range(len(data))):
        prompt = make_emotion(data.iloc[i]['singer'],data.iloc[i]['song_name'], tokenizer)
        result = use_model(prompt,text_generation_pipeline, tokenizer )
        emotions.append(result)
    data['emotion'] = emotions

    data.to_csv('Save Directory',index = False)


if __name__ == '__main__':
    main()