import os
from vosk import Model, KaldiRecognizer
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import wave
import json

class WordInfo:
  def __init__(self, word, speakers, start_frame, end_frame, person_talking=None):
    self.word = word
    self.speakers = speakers # List of speakers and the percentage of time they are talking
    self.start_frame = start_frame
    self.end_frame = end_frame
    self.person_talking = person_talking

class CreateTranscript:
    def __init__(self, selected_video_file, persons):
        self.selected_video_file = selected_video_file
        self.persons = persons
        self.frame_rate = 25
        self.selected_audio_file = None
        self.vosk_model_path = "MultiSpeech/FaceDetector/models/vosk-model-small-en-us-0.15"
        self.wordTranscriptList = []
        self.final_transcript = []

        self.extractAndConvertAudio(self.selected_video_file)
        self.createWordDict(self.persons, self.selected_audio_file, self.vosk_model_path)
        self.decideFinalResults(self.wordTranscriptList)
        self.final_transcript = self.createTranscript(self.wordTranscriptList)
        self.print(self.final_transcript)
    
    def extractAndConvertAudio(self, selected_video_file):
        """Extract audio from the video and convert it to a format that Vosk can use."""
        
        clip = VideoFileClip(selected_video_file) # Load video
        audio = clip.audio # Extract audio

        # Save audio as WAV
        audio_temp_path = self.selected_video_file[:-4] + "_temp.wav"
        audio_path = self.selected_video_file[:-4] + ".wav"
        audio.write_audiofile(audio_temp_path)
        
        audio = AudioSegment.from_wav(audio_temp_path) # Load audio with PyDub
        audio = audio.set_channels(1) # Convert audio to mono

        audio.export(audio_path, format="wav") # Save audio as Linear16 PCM WAV
        
        os.remove(audio_temp_path) # Remove the temporary audio file

        self.selected_audio_file = audio_path
        print(f"Extracted and converted audio saved to {audio_path}")

    def createWordDict(self, persons, selected_audio_file, vosk_model_path):
        """Create a dictionary of words and their start and end times from the audio file using Vosk."""

        # print(f"Using Vosk model at {vosk_model_path}")
        # print(f"Processing audio file {selected_audio_file}")

        with wave.open(selected_audio_file, "rb") as wf:
            
            if wf.getnchannels() != 1:
                raise ValueError("Audio file must be mono")
            if wf.getsampwidth() != 2:
                raise ValueError("Audio file must be 16-bit PCM")

            # Create a recognizer object
            vosk_model = Model(vosk_model_path)
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            rec.SetWords(True)

            # Read the audio data in chunks and process
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    accept_result = rec.Result()
                    results.append(accept_result)
                    # print(f"AcceptWaveform result: {result}")
                else:
                    partial_result = rec.PartialResult()
                    results.append(partial_result)
                    # print(f"Partial result: {partial_result}")
            
            # Get the final result
            final_result = rec.FinalResult()
            results.append(final_result)

            # Extract word-level information
            results_dict = []
            for result in results:
                res = json.loads(result)
                if 'result' in res:
                    for word in res['result']:
                        word_info = {
                            'word': word['word'],
                            'start': word['start'],
                            'end': word['end'],
                            'confidence': word['conf']
                        }
                        results_dict.append(word_info)
                    
            self.processResults(results_dict, persons)
        
        os.remove(selected_audio_file) # Remove the audio file after processing
        
        # print(f"Length of results_dict: {len(results_dict)}")
        # for word_info in results_dict:
        #     print(f"Word: {word_info['word']}, Start: {word_info['start']}, End: {word_info['end']}, Confidence: {word_info['confidence']}")

    def processResults(self, results_dict, persons):
        """Process the results from Vosk and create a list of WordInfo objects."""

        for result in results_dict:
            word = result["word"]
            start_frame = int(result["start"] * self.frame_rate)
            end_frame = int(result["end"] * self.frame_rate)
            personsTalking = []
            
            for person in persons:
                percentTalk = 0 # Percentage of time the person is talking
                faces = person.get_faces()
                frame_count = end_frame - start_frame

                if frame_count > 0: # This is to avoid division by 0
                    for frame_num in range(start_frame, end_frame): # Loop through each frame
                        for face in faces:
                            if face.get_frame_number() == frame_num and face.is_talking() == 2: # If the face is talking on current frame number
                                percentTalk += 1
                    percentTalk = percentTalk / frame_count
                else:
                    percentTalk = 0 # If the frame count is 0, set the percentage of time the person is talking to 0
                
                personsTalking.append((person.get_label(), percentTalk)) # Add the person and the percentage of time they are talking
        
            word_info = WordInfo(word, personsTalking, start_frame, end_frame) # Create a WordInfo object
            self.wordTranscriptList.append(word_info) # Add the WordInfo object to the list of wordTranscriptList

    def decideFinalResults(self, wordTranscriptList):
        """Decide the final results of the transcript by determining who is talking at each word."""
        
        for word_index, word in enumerate(wordTranscriptList):
     
            count_max_value_tuples = len([t for t in word.speakers if t[1] == (max(t[1] for t in word.speakers))]) # Count the number of tuples with the max value

            if count_max_value_tuples == 1: # If there is only one tuple with the max value ???????????????????????????
            
                if max(word.speakers, key=lambda pair: pair[1])[1] > 0.5: # If the max value is greater than 0.5
                    word.person_talking = max(word.speakers, key=lambda pair: pair[1])[0]
                
                else: 
                    word.person_talking = wordTranscriptList[word_index - 1].person_talking # Set the person talking to the person talking in the previous word
            
            else: 
                word.person_talking = wordTranscriptList[word_index - 1].person_talking # Set the person talking to the person talking in the previous word

    def createTranscript(self, wordTranscriptList):
        """Create a formatted transcript from the list of WordInfo objects."""
        
        wordTranscriptList.sort(key=lambda x: x.start_frame) # Sort words by start_frame to get chronological order

        # Group words by the person talking, taking time into account
        transcript = []
        current_speaker = None
        current_speech = []
        current_time_start = None

        for word_index in wordTranscriptList:
            if word_index.person_talking != current_speaker:
                if current_speaker is not None:
                    if current_speech:
                        transcript.append(
                            (current_speaker, current_time_start, word_index.start_frame, ' '.join(current_speech))
                        )
                current_speaker = word_index.person_talking
                current_speech = [word_index.word]
                current_time_start = word_index.start_frame
            else:
                current_speech.append(word_index.word)

        # Add the last group of words
        if current_speech:
            transcript.append(
                (current_speaker, current_time_start, wordTranscriptList[-1].end_frame, ' '.join(current_speech))
            )

        # Format the transcript
        formatted_transcript = []
        for speaker, start, end, speech in transcript:
            formatted_transcript.append(
                f"Time {start} to {end} - Person {speaker}:\n{speech}\n"
            )

        return '\n'.join(formatted_transcript)    

    def print(self, final_transcript):
        
        print(final_transcript)

        # for word in self.wordTranscriptList:
        #     print("-------------------------------------------------")
        #     print("Word:", word.word)
        #     print("Speakers:", word.speakers)
        #     print("Start Frame:", word.start_frame)
        #     print("End Frame:", word.end_frame)
        #     print("Person Talking:", word.person_talking)
        #     print()