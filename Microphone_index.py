import speech_recognition as sr

print("Available microphone devices:")
for i, microphone_name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {microphone_name}")
