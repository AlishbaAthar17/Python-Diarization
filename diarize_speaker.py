import asyncio
import os
from deepgram import Deepgram
from transformers import pipeline
from typing import Dict
# Import the functions from the summarizer module
from textrank import read_transcript, generate_summary

audio_file = input("Enter the file path for the recorded meeting: ")

async def speakerTime(speech_data: Dict):

    if 'results' in speech_data:
        transcript = speech_data['results']['channels'][0]['alternatives'][0]['words']

        speaker_time = {}
        speaker_sentences = [] #list to be used here not a dictionary
        
        current_speaker = -1 
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)

        #sentiment_analyzer = pipeline("sentiment-analysis")

        for speaker in transcript:
            speaker_number = speaker["speaker"]

            if speaker_number is not current_speaker: 
                current_speaker = speaker_number
                speaker_sentences.append([speaker_number, [], 0]) #This appends new list to the sentences list when a new speaker speaks. The speaker number updates in the above line and the words are appended to a new list

                try:
                    speaker_time[speaker_number][1] += 1
                except KeyError:
                    speaker_time[speaker_number] = [0,1]

            current_word = speaker["word"]
            speaker_sentences[-1][1].append(current_word)

            speaker_time[speaker_number][0] += speaker["end"] - speaker["start"] 
            speaker_sentences[-1][2] += speaker["end"] - speaker["start"]


        for speaker, sentences, time in speaker_sentences:
            sentence = ' '.join(sentences)
            sentiment = sentiment_analyzer(sentence)
            print(f"<Speaker {speaker}>: {' '.join(sentences)}")
            print(f"Sentiment for <Speaker {speaker}>: {sentiment[0]['label']}")
            print(f"<Speaker {speaker}>: {time}") 
            
        for speaker, (total_time, total_sentences) in speaker_time.items(): 
            print(f"<Speaker {speaker}> average time: {total_time/total_sentences} ")
            print(f"Total time of speech: {total_time}")
        

    return transcript

def generate_summary(joined_words):
    summarizer = pipeline('summarization')
    speech = joined_words
    print(summarizer(speech, max_length=150, min_length=30, do_sample=False, truncation=True))
    summary_new = str(summarizer(speech, max_length=150, min_length=30, do_sample=False, truncation=True))

    base_file_name = "Summary.txt"
    write_file_path = base_file_name
    count = 1
    while os.path.exists(write_file_path):
        write_file_path = f"{base_file_name[:-4]}_{count}.txt"
        count += 1

    with open(write_file_path, 'w') as write_file:
        write_file.write(summary_new)

    read_file_path = f"{base_file_name[:-4]}_read.txt"
    with open(read_file_path, 'w') as read_file:
        read_file.write(summary_new)
    

async def main():
    DEEPGRAM_API_KEY = "3e1e03d5a644d32de205bf201760d1bb42207ef5"
    deepgram = Deepgram(DEEPGRAM_API_KEY)

    try:
        with open(audio_file, 'rb') as audio:
            source_file = {'buffer': audio, 'mimetype': 'audio/mp3'}
            transcription = await deepgram.transcription.prerecorded(source_file, {'punctuate': True, 'diarize': True})

            if transcription is not None:
                speakers = await speakerTime(transcription)

                all_words = [speaker["word"] for speaker in transcription['results']['channels'][0]['alternatives'][0]['words']]
                joined_words = " ".join(all_words)
                print(len(all_words))
                base_file_name = "SpeechToText.txt"
                write_file_path = base_file_name
                count = 1
                while os.path.exists(write_file_path):
                    write_file_path = f"{base_file_name[:-4]}_{count}.txt"
                    count += 1

                with open(write_file_path, 'w') as write_file:
                    write_file.write(joined_words)

                # Create a new file for reading with a different name
                read_file_path = f"{base_file_name[:-4]}_read.txt"
                with open(read_file_path, 'w') as read_file:
                    read_file.write(joined_words)
                
                generate_summary(joined_words)

    except Exception as e:
        print(f"Error: {e}")
    
    # summarizer = pipeline('summarization')
    # speech = joined_words
    # print(summarizer(speech,max_length=150,min_length=30, do_sample=False, truncation=True))

# ...

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
asyncio.run(main())

def finalResult():
    print("Transcripts")
    print("Meeting Summary")
    print("Meeting Minutes")


# import asyncio #asyncio is a library to write concurrent code using the async/await syntax
# from deepgram import Deepgram #end-to-end deep learning architecture and AutoML training allows Deepgram to create highly accurate, use-case specific speech recognition models.
# from typing import Dict
# from transformers import pipeline
# # import assemblyai as aai
# # from tinytag import TinyTag

# audio_file = 'premier_broken-phone.mp3'

# # @asyncio.coroutine --> this is an outdated version to define asynchronous functions so it wont work
# # def speakerTime(speech_data: Dict):

# #This function below takes speech transcript data as input and calculates and prints speaking time statistics for each speaker
# async def speakerTime(speech_data: Dict):
#    if 'results' in speech_data:
#        transcript = speech_data['results']['channels'][0]['alternatives'][0]['words']

#        speaker_time = {}
#        speaker_words = []
       
#        current_speaker = -1 #to keep track of the current speaker

#        for speaker in transcript:
#            speaker_number = speaker["speaker"]

#            if speaker_number is not current_speaker: #If the speaker changes, update the current speaker and initialize their word list.
#                current_speaker = speaker_number
#                speaker_words.append([speaker_number, [], 0]) # 0 is the total amount of time per phrase for each speaker

#             #Try-Except error handling: to update a counter associated with a particular speaker number in a dictionary. If the speaker number is already present in the dictionary, it increments the count. If not, it creates a new entry for that speaker with a count of 1.
#                try:
#                    speaker_time[speaker_number][1] += 1
#                except KeyError:
#                    speaker_time[speaker_number] = [0,1]

#            current_word = speaker["word"]
#            speaker_words[-1][1].append(current_word)

#            speaker_time[speaker_number][0] += speaker["end"] - speaker["start"] # [0] gets the total time
#            speaker_words[-1][2] += speaker["end"] - speaker["start"]


#        for speaker, words, time in speaker_words:
#            print(f"<Speaker {speaker}>: {' '.join(words)}")
#            print(f"<Speaker {speaker}>: {time}") #time for one speaker
        
#        for speaker, (total_time, amount) in speaker_time.items(): # (unpacks)key goes into total_time, value goes into amount
#            print(f"<Speaker {speaker}> average time: {total_time/amount} ")
#            print(f"Total time of speech: {total_time}")
        

#    return transcript


# async def main():

#    DEEPGRAM_API_KEY = "YOUR_API_KEY"
#    deepgram = Deepgram(DEEPGRAM_API_KEY)

#    with open(audio_file, 'rb') as audio: #read in binary mode
#        source_file = {'buffer': audio, 'mimetype': 'audio/mp3'}
#        transcription = await deepgram.transcription.prerecorded(source_file, {'punctuate': True, 'diarize': True})
#         #The await keyword is used in asynchronous functions to pause the execution of the function until the awaited asynchronous operation is complete
#         #'diarize': True: It suggests performing speaker diarization, which is the process of distinguishing and labeling different speakers in the audio.
#         #'punctuate':True: It suggests adding punctuation to the texts extracted from the audio file
#        speakers = await speakerTime(transcription)

#         #Converts the whole Audio file to text
#        all_words = [speaker["word"] for speaker in transcription['results']['channels'][0]['alternatives'][0]['words']]
#        joined_words = " ".join(all_words)
#        print("\nSpeech-To-Text:")
#        print(joined_words)
#        print(len(joined_words))

#     #    for totalwords in joined_words:
#     #     if (len(joined_words) > 2000):
#     #         joined_words.split(maxsplit=3)
        

#        summarizer = pipeline('summarization')
#        speech = joined_words
#        print(summarizer(speech,max_length=150,min_length=30, do_sample=False, truncation=True)) 

#     #    with open('categorize.pkl', 'w') as f:
#     #     for lines in summarized_text:
#     #         f.write(str(lines))
#     #         f.write('\n')
       
#     #    audio = TinyTag.get("premier_broken-phone.mp3")
#     #    print("Genre:" + audio.genre)
  
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# asyncio.run(main())


