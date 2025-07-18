I wrote this program in VS Code, so I don't know if it'll work elsewhere. This program will wait for the wake words "Cyn" or "Hey Cyn", take a picture, and compare the face(s) in the frame to the one(s) in the FRS (Facial Recognition System). Plus, If you name the file 
image(s) of the face(s) as the name of the person and make it a jpg or a png, Cyn will use the file name as the name for the person who's face is in the picture. It'll also listen to your selected microphone, use Google STT and Whisper TOGETHER for a super accurate voice 
transcription, and send the text to an AI model that will impersonate Cyn in it's response. However, there's things you need to do first. Go read requirements.txt for well...the requirements. But that's not all. You need to use Microphone_index.py to find the index for 
your selected microphone, and change MICROPHONE_INDEX (line 63) to the index of your microphone (or set it to None if you're not sure). You'll also need to use webcam_index_tester.py to find the index of your selected webcam, and change the integer to the index of your 
webcam (line 226). Once you have both of those, you'll need to make a folder to put your different things in. The folder can be named whatever you want it to be named, but here's what you need to make/put in it:
1. Cyn_AI.py (obviously)
2. cyn_system_prompt.txt
3. cyn_memory.json (Don't mind the last few things I put in the memory file, my AI kept glitching)
4. config.json
5. A picture/multiple pictures of faces you want to add to the FRS named something like Jacob.png or Abby.jpg

Also, to change Cyn's personalized messages to any known people or other general stuff, use the config.json file.
