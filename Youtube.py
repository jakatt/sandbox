# Initialisation
import os
import openai
import sys
import gradio as gr
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from pydub import AudioSegment
from pytube import YouTube
from transformers import pipeline
sys.path.append('../..')
from dotenv import load_dotenv, find_dotenv
_=load_dotenv(find_dotenv("setvar.env"))
openai.api_key=os.environ["OPENAI_API_KEY"]
save_dir="testdocs/"

def instructions_and_inputs_to_openai(text):
  separator="*****"
  messages =  [  
    {'role':'system', 
    'content':"""You are an assitant who answer in english./
    Provide a summary in telegraphic mode with numbered bullets points."""},    
    {'role':'user', 
    'content':"""Summarize the text between 5 asterisks in english:"""+separator+text+separator},  
    ] 
  response = get_completion_from_messages(messages, temperature=1)
  return response

def get_completion_from_messages(messages, 
                              model="gpt-3.5-turbo", 
                              temperature=0, 
                              max_tokens=500):
  response = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      temperature=temperature, # this is the degree of randomness of the model's output
      max_tokens=max_tokens, # the maximum number of tokens the model can ouptut 
  )
  return response.choices[0].message["content"]

def get_audio_and_video_files(url):
  m4afiles = os.listdir(path=save_dir)
  for file in m4afiles:
  # check if the file ends with .m4a
    if file.endswith(".m4a"):
      os.remove(save_dir+file)
  loader = GenericLoader(
  YoutubeAudioLoader([url],save_dir),
  OpenAIWhisperParser()
  )
  docs = loader.load()
  video = YouTube(url)
  stream = video.streams.get_highest_resolution()
  stream.download(output_path="testdocs",filename="file.mp4")
  audio_file_path = save_dir+"file.wav"
  video_file_path = save_dir+"file.mp4"
  for n in os.listdir(path=save_dir):
      if n.endswith("m4a"):
          sound = AudioSegment.from_file(save_dir+n, format='m4a')
          file_handle = sound.export(save_dir+"file.wav", format='wav')
#  with open("transcript.txt","w") as file:
#    file.write(docs[0].page_content)
  return video_file_path, docs[0].page_content, audio_file_path

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    #summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def myaiapp(url):
  video_file,audio_transcription,audio_file=get_audio_and_video_files(url)
  #Summary by OpenAI
  #response = instructions_and_inputs_to_openai(audio_transcription)

  #Summary disabled
  #response="Enable OpenAI to get the summary"
  
  #Summary by facebook/bart-large-cnn
  response=summarize_text(audio_transcription)
  return video_file, audio_transcription, audio_file, response

with gr.Blocks() as demo:
  gr.Markdown("""# Youtube transcription
    Transcript Youtube video in an audio file that can then be ingested in different manners (e.g.: summary, Q&A on the content...)""")
  with gr.Tab("Youtube"):
    yt_input= gr.Textbox(label="Youtube",placeholder="Type the Youtube video URL here")
    yt_outputs=[
      gr.Video(label="Video"),
      gr.Textbox(label="Audio transcript"),
      gr.Audio(label="Audio"),
      gr.Textbox(label="Summary")
    ]
    yt_button=gr.Button("Process")
    gr.Examples(["https://www.youtube.com/watch?v=0Cn9IBtazjs","https://www.youtube.com/watch?v=ZHjr3AdriWs&list=PLDrBFlreuiQuhpAFD6UTgec5MG7ArZ6eB&index=3"],yt_input)
  with gr.Tab("PDF"):  
    pdf_input= gr.Textbox(label="PDF",placeholder="Type the PDF file URL here")
    pdf_output=gr.Textbox(label="Video")
    pdf_button=gr.Button("Process") 
  yt_button.click(myaiapp,inputs=yt_input,outputs=yt_outputs)
  pdf_button.click(myaiapp,inputs=pdf_input,outputs=pdf_output)

if __name__ == "__main__":
    demo.launch()