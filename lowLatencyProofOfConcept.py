import datetime
import io
import logging
import os
import queue
import threading
from elevenlabslib import *
import keyring
import speech_recognition as sr
import faster_whisper
import pyaudio
import nltk
import openai

#Config data (fill this in)
openAIAPIKey = None
elevenAPIKey = None
elevenLabsVoiceName = "Rachel"
latencyOptimizationLevel=3      #4 is slightly faster but can mispronounce dates and numbers. Not worth it imo.
whisperModel = "base.en"



#Initialize openAI and elevenlabslib with the given API keys (or pull them from keyring for me because I already have them stored there)
openai.api_key = openAIAPIKey or keyring.get_password("openai_chat_app", "openai_key")
user = ElevenLabsUser(elevenAPIKey or keyring.get_password("openai_chat_app", "elevenlabs_key"))

#Get the voice
voice = user.get_voices_by_name(elevenLabsVoiceName)[0]


#Variables used for the parallelization of the TTS calls
eventQueue = queue.Queue()
readyForPlaybackEvent = threading.Event()
readyForPlaybackEvent.set()

#pyAudio backend, used to get info about the audio devices
pyABackend = pyaudio.PyAudio()
outputDeviceIndex = -1

#Load the chosen model using faster-whisper.
model = faster_whisper.WhisperModel(whisperModel, device="cuda", compute_type="float16")


#Check if we have already downloaded the required nltk data, otherwise download it.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
logging.basicConfig(level=logging.DEBUG)
#Eligible characters for the end of a sentence (used for openAI output streaming):
sentenceEndCharacters = [".","?","!"]

#Object to keep track of and print latency information
latencyData = dict()

def main():
    #Microphone setup for speech recognition
    defaultInputInfo = pyABackend.get_default_input_device_info()
    print(f"Default input device: {defaultInputInfo['name']} - {defaultInputInfo['index']}")
    microphoneInfo = get_portaudio_device_info_from_name(choose_from_list_of_strings("Please choose your input device.", get_list_of_portaudio_devices("input")))
    srMic = sr.Microphone(device_index=microphoneInfo["index"], sample_rate=int(microphoneInfo["defaultSampleRate"]))
    recognizer = sr.Recognizer()

    #Audio output setup
    defaultOutputInfo = pyABackend.get_default_output_device_info()
    print(f"Default output device: {defaultOutputInfo['name']} - {defaultOutputInfo['index']}")
    outputInfo = get_portaudio_device_info_from_name(choose_from_list_of_strings("Please choose your output device.", get_list_of_portaudio_devices("output")))
    global outputDeviceIndex
    outputDeviceIndex = outputInfo["index"]

    # These values are specific to MY microphone - you may have to change them for yours for it to work well.
    recognizer.pause_threshold = 0.5
    recognizer.energy_threshold = 250
    recognizer.dynamic_energy_threshold = False




    # Start the thread that handles the TTS parallelization.
    # For more information, see the comments in that function.
    threading.Thread(target=waitForPlaybackReady).start()


    #Listen to voice input using speechRecognition
    with srMic as source:
        print("Listening to voice input...")
        audio = recognizer.listen(source)

    print("Complete sentence identified.")
    latencyData["start_time"] = datetime.datetime.now()

    #Transcribe the audio using faster-whisper
    recognizedText = ""
    segments, info = model.transcribe(io.BytesIO(audio.get_wav_data()), language="en", beam_size=5)
    print(info)
    for segment in segments:
        recognizedText += " " + segment.text

    recognizedText = recognizedText.strip()
    print(f"Recognized text:{recognizedText}")
    latencyData["recognize_time"] = datetime.datetime.now()


    #Set up the communication with the openAI API.
    messages = [{"role": "user", "content": recognizedText}]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True, max_tokens=1024)

    #Helper variables.
    totalText = ""
    textToBeSpoken = ""
    sentenceEndFlagged = False


    #This part is fairly critical. I'll explain what it does in short.
    #-Streams the response from the openAI API
    #-If it identifies that the current chunk contains a character that MIGHT mark the end of a sentence, sentenceEndFlagged is set to true.
    #-On the next chunk, it will use nltk to check if the next now contains more than one sentence (The reason NLTK is used is to avoid false positives with things such as IP addresses).
    #-If NLTK identifies that the text contains more than one sentence, the sentences BESIDES the last one (which will still be incomplete) are forwarded to the TTS.
    #-The text contained within these sentences is then cut from textToBeSpoken, which is the variable that keeps track of all the text that has yet to be read out loud.

    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        if "content" in chunk_message:
            chunk_text = chunk_message['content']
            textToBeSpoken += chunk_text
            totalText += chunk_text
            #if message_text contains a sentence-ending character:
            if sentenceEndFlagged:
                sentences = nltk.sent_tokenize(textToBeSpoken)
                if len(sentences) > 1:
                    logging.debug("Detected new sentence!")
                    if "response_time" not in latencyData:
                        latencyData["response_time"] = datetime.datetime.now()
                    for index, sentence in enumerate(sentences):
                        if index < len(sentences)-1:
                            synthesizeAndPlayAudio(sentence)
                            # Remove the already synthesized sentence from the overall text.
                            textToBeSpoken = textToBeSpoken[len(sentence):].lstrip()

                    sentenceEndFlagged = False
            for character in sentenceEndCharacters:
                if character in chunk_text:
                    logging.debug("Sentence-ending chunk detected.")
                    #This is a sentence-ending section. Let's hand it over to NLTK to do the sentence splitting, to make sure it actually is a completed sentence.
                    sentenceEndFlagged = True


    #We're at the end of the message. Let's make sure we don't have any leftover text.
    if len(textToBeSpoken) > 0:
        #We have some text left over, let's speak it.
        sentences = nltk.sent_tokenize(textToBeSpoken)
        for sentence in sentences:
            synthesizeAndPlayAudio(sentence)


    #This just prints all the latency information.
    print(f"Complete response received: {totalText}")
    latencyData["complete_response_time"] = datetime.datetime.now()

    input("Press Enter once playback is finished.")
    print("Latency report:")
    print(f"Time taken from end of sentence to text recognition: {(latencyData['recognize_time']-latencyData['start_time']).total_seconds()}s")
    print(f"Time taken from recognition to the first completed sentence of the API's response: {(latencyData['response_time'] - latencyData['recognize_time']).total_seconds()}s")
    print(f"Time taken from first response to audio playback start: {(latencyData['speak_time'] - latencyData['response_time']).total_seconds()}s")

    print(f"\nTime taken from first sentence recognition to full response recieved (time saved by streaming the OpenAI output): {(latencyData['complete_response_time'] - latencyData['response_time']).total_seconds()}s")
    print(f"\nTotal time taken from end of sentence to playback start: {(latencyData['speak_time']-latencyData['start_time']).total_seconds()}s")

    input("Press Enter to exit.")


#The following section is the main latency-saving trick when it comes to the TTS.
#To summarize what happens, waitForPlaybackReady runs in its own separate thread.
#Its job is to detect when a playback has ended, and to signal to the next waiting thread that it's ready.

#Every time synthesizeAndPlayAudio is called, a new thread is created.
#This is actually the main reason I added callback functionality to elevenlabslib.
#Every time a new thread is created, a new event is also created and put into the eventQueue.
#By setting the startcallbackfunc to wait for this new event, we can control exactly when the playback starts, while letting the download progress in the background.
#Once the play back is over, readyForPlaybackEvent is raised, letting the waitForPlaybackReady thread pick a new event from the queue to raise.

def waitForPlaybackReady():
    while True:
        readyForPlaybackEvent.wait()
        readyForPlaybackEvent.clear()
        nextEvent = eventQueue.get()
        nextEvent.set()

def synthesizeAndPlayAudio(prompt) -> None:
    #Create a new event and add it to the eventQueue managed by the waitForPlaybackReady thread.
    newEvent = threading.Event()
    eventQueue.put(newEvent)
    def startcallbackfunc():
        #This function is run once the stream is ready to begin playback.
        #It waits for the event to be set by the waitForPlaybackReady thread.

        #NOTE: In pycharm specifically, for some reason breakpoints in this function do not work unless I use the breakpoint() function, so be warned.
        #breakpoint()
        newEvent.wait()
        logging.debug("Playing audio: " + prompt)
        if "speak_time" not in latencyData:
            latencyData["speak_time"] = datetime.datetime.now()
    def endcallbackfunc():
        #This function is run once the playback has ended.
        #It sets readyForPlaybackEvent, which allows the waitForPlaybackReady thread to pick a new event from the queue.
        logging.debug("Finished playing audio:" + prompt)
        readyForPlaybackEvent.set()
    playbackOptions = PlaybackOptions(runInBackground=True, onPlaybackStart=startcallbackfunc, onPlaybackEnd=endcallbackfunc, portaudioDeviceID=outputDeviceIndex)
    voice.generate_stream_audio_v2(prompt=prompt, playbackOptions=playbackOptions)




#UI stuff, not relevant to the main program.

def get_list_of_portaudio_devices(deviceType:str) -> list[str]:
    """
    Returns a list containing all the names of portaudio devices of the specified type.
    """
    if deviceType != "output" and deviceType != "input":
        raise ValueError("Invalid audio device type.")

    deviceNames = list()
    for hostAPI in range(pyABackend.get_host_api_count()):
        hostAPIinfo = pyABackend.get_host_api_info_by_index(hostAPI)
        for i in range(hostAPIinfo["deviceCount"]):
            device = pyABackend.get_device_info_by_host_api_device_index(hostAPIinfo["index"], i)
            if device["max" + deviceType[0].upper() + deviceType[1:] + "Channels"] > 0:
                deviceNames.append(f"{device['name']} (API: {hostAPIinfo['name']}) - {device['index']}")

    return deviceNames

def get_portaudio_device_info_from_name(deviceName:str):
    chosenDeviceID = int(deviceName[deviceName.rfind(" - ") + 3:])
    chosenDeviceInfo = pyABackend.get_device_info_by_index(chosenDeviceID)
    return chosenDeviceInfo

def choose_int(prompt, minValue, maxValue) -> int:
    print(prompt)
    chosenVoiceIndex = -1
    while not (minValue <= chosenVoiceIndex <= maxValue):
        try:
            chosenVoiceIndex = int(input("Input a number between " + str(minValue) +" and " + str(maxValue)+"\n"))
        except ValueError:
            print("Not a valid number.")
    return chosenVoiceIndex
def choose_from_list_of_strings(prompt, options:list[str]) -> str:
    print(prompt)
    if len(options) == 1:
        print("Choosing the only available option: " + options[0])
        return options[0]

    for index, option in enumerate(options):
        print(str(index+1) + ") " + option)

    chosenOption = choose_int("", 1, len(options)) - 1
    return options[chosenOption]



if __name__=="__main__":
    main()