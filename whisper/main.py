import whisper # type: ignore
import datetime
model_name: str = input("Enter model name from the list [tiny, base, small, medium, turbo, large]: ")
print(f"Please wait text is being extracted with model {model_name}...")
start_time = datetime.datetime.now()
model = whisper.load_model(model_name)

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("deepak.m4a")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)

detected_language = max(probs, key=probs.get)

# decode the audio
options = whisper.DecodingOptions(language="en")
result = whisper.decode(model, mel, options)

end_time =  datetime.datetime.now()
file_name = f"{model_name}_output.txt"

with open(file_name, "w") as file:
    file.write(f"Total time taken (in sec): {(end_time - start_time).total_seconds()}\n")
    file.write(f"Detected language: {detected_language}\n")
    file.write(f"result:\n{result.text}")

print(f"The result is saved in {file_name}")