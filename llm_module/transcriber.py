import whisper
model = whisper.load_model("tiny")
result = model.transcribe("Jason Mraz - I Won't Give Up.mp3")
print(result["text"])