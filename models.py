import whisper
import google.generativeai as genai
import re
import os
import numpy as np
import torch
from samplings import top_p_sampling, temperature_sampling
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
import pretty_midi
import soundfile as sf
import librosa

def transcriber(filepath:str)->str:
    model = whisper.load_model("tiny")
    result = model.transcribe(filepath)
    result = result["text"]
    return result


def get_new_lyrics(lyrics:str)->str:
    genai.configure(api_key="AIzaSyCKGBx9lc5yntM7TZcp53_ogHtqsXjRdRo")
    model = genai.GenerativeModel("gemini-1.5-flash")
    song_type = model.generate_content(f"In one word, tell me what type of song has the lyrics: {lyrics}").text
    song_genre = model.generate_content(f"In one word, tell me what genre of song has the lyrics: {lyrics}").text
    new_lyrics = model.generate_content(f"Given lyrics {lyrics}, write a new verse for the song")
    return [song_type, song_genre, new_lyrics]

def text_to_music(prompt:str)->list:
    tokenizer = AutoTokenizer.from_pretrained('sander-wood/text-to-music')
    model = AutoModelForSeq2SeqLM.from_pretrained('sander-wood/text-to-music')
    max_length = 1024
    top_p = 0.9
    temperature = 1.0
    
    input_ids = tokenizer(prompt, 
                      return_tensors='pt', 
                      truncation=True, 
                      max_length=max_length)['input_ids']

    decoder_start_token_id = model.config.decoder_start_token_id
    eos_token_id = model.config.eos_token_id

    decoder_input_ids = torch.tensor([[decoder_start_token_id]])

    for t_idx in range(max_length):
        outputs = model(input_ids=input_ids, 
                        decoder_input_ids=decoder_input_ids)
        probs = outputs.logits[0][-1]
        probs = torch.nn.Softmax(dim=-1)(probs).detach().numpy()
        sampled_id = temperature_sampling(probs=top_p_sampling(probs, 
                                                            top_p=top_p, 
                                                            return_probs=True),
                                        temperature=temperature)
        decoder_input_ids = torch.cat((decoder_input_ids, torch.tensor([[sampled_id]])), 1)
        if sampled_id!=eos_token_id:
            continue
        else:
            tune = "X:1\n"
            tune += tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
            print(tune)
            break
    return tune
    
def musicgen(prompt:str):
    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")
    model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    inputs = processor(
        text=[f"{prompt}"],
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
    sampling_rate = model.config.audio_encoder.sampling_rate
    '''audio_data = audio_values[0].numpy().astype(np.float32)
    audio_data = audio_data.squeeze()
    sf.write('generated_audio.wav', audio_data, sampling_rate)'''
    return [audio_values, sampling_rate]

if __name__ == "__main__":
    lyrics = transcriber("../uploads/"+os.listdir('../uploads')[0])
    print(lyrics)