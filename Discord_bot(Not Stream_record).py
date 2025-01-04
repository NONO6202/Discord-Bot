import os
import re
import io
import json
import wave
import random
import asyncio
import threading
from PIL import Image
from copy import copy
from requests import get
from datetime import datetime

import discord
from discord import FFmpegPCMAudio, app_commands
from discord.ext import tasks, voice_recv

from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genAI
from gtts import gTTS
import tiktoken

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

import numpy as np
import librosa
from scipy import signal
from pydub import AudioSegment
from samplerate import resample
import pyaudio
import soundfile

class pitch_conversion:
    class pitch_basics:
        def butter_lowpass(self, cutoff: float, fs: int, order: int = 3) -> tuple:
            b, a = signal.butter(order, cutoff, btype='low', fs=fs, analog=False)
            return (b, a)
        def frameize(self, x: np.array, N: int, H_a: int, hfilt: np.array) -> list:
            frames = []
            idx = 0
            while True:
                try: frames.append(hfilt * x[H_a*idx:H_a*idx+N])
                except: break
                idx += 1
            return frames
        def find_hfilt_norm(self, hfilt: np.array, H_s: int, delta: int = 0) -> np.array:
            hf_norm = copy(hfilt)
            N = len(hfilt)
            if (H_s + delta) < N and (H_s + delta) >= 0:
                hf_norm[(H_s+delta):] += hfilt[:N-(H_s+delta)]
                hf_norm[:N-(H_s+delta)] += hfilt[(H_s+delta):]
            return hf_norm
        def warp_spectrum(self, S: np.array, factor: float) -> np.array:
            out_S = np.array([
                np.interp((np.arange(0, len(s)) / len(s)) * factor,
                          (np.arange(0, len(s)) / len(s)),
                          s)
                for s in S.T], dtype=complex).T
            return out_S
    def butter_lowpass_filter(self, data: np.array, cutoff: float, fs: int, order: int = 3) -> np.array:
        b, a = self.pitch_basics().butter_lowpass(cutoff, fs, order=order)
        return signal.filtfilt(b, a, data)
    def scale_time(self, x: np.array, N: int, H_a: int, hfilt: np.array, alpha: float) -> np.array:
        frames = self.pitch_basics().frameize(x, N, H_a, hfilt)
        H_s = int(np.round(H_a * alpha))
        out_x = np.zeros(len(frames)*H_s+N)
        # time-scaling
        for i, frame in enumerate(frames):
            hfilt_norm = self.pitch_basics().find_hfilt_norm(hfilt, H_s)
            out_x[i*H_s:i*H_s+N] += frame/hfilt_norm
        return out_x
    def synthesize_pitch(self, x: np.array, sr: int, N: int, H_a: int, hfilt: np.array, alpha: float) -> np.array:
        syn_x = self.scale_time(x, N, H_a, hfilt, alpha)
        # apply anti-aliasing
        if alpha >= 1:
            syn_x = self.butter_lowpass_filter(syn_x, sr/2*(1/alpha)*0.6, fs=sr, order=3)
        # resampling
        syn_x = resample(syn_x, 1/alpha, 'sinc_best')
        syn_x = syn_x / np.max(abs(syn_x))
        return syn_x
    def shift_freq(self, x: np.array, alpha: float, n_fft=512, hop_length=64) -> np.array:
        S1 = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
        S2 = self.pitch_basics().warp_spectrum(S1, alpha**(1/3) + 0.06)
        return librosa.istft(S2, hop_length=hop_length, win_length=n_fft)
    def pitch_modulation(self, N, pitch, path_tts):
        H_a = int(N*0.33)
        hfilt = np.hanning(N)
        data, sr = librosa.load(path_tts, sr=None)
        out_data = self.synthesize_pitch(data, sr, N, H_a, hfilt, alpha=pitch)
        data = self.shift_freq(data, pitch**(1/3) + 0.06)
        soundfile.write(path_tts, out_data, sr, format='wav')

class Initial_Setting:
    class SoundBoard_Setting:
        filepath = os.path.dirname(os.path.abspath(__file__))
        def __init__(self,guild_id):
            self.guild_id = guild_id
        def soundboard_Setting(self):
            filepath = self.filepath + "/soundboard"
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            soundBoard = os.path.join(filepath, f"({self.guild_id})soundboard.json")
            if not os.path.exists(soundBoard):
                create_soundBoard =  open(soundBoard, 'a')
                create_soundBoard.write('{\n}')
                create_soundBoard.close
    filepath = os.path.dirname(os.path.abspath(__file__))
    def Gemini_Setting(self):
        filepath = self.filepath + "/gemini"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    def tts_Setting(self):
        filepath = self.filepath + "/stts"
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    def guild_Setting(self):
        guild_path = os.path.join(self.filepath, f"guild.json")
        if not os.path.exists(guild_path):
            create_soundBoard =  open(guild_path, 'a')
            create_soundBoard.write('{\n}')
            create_soundBoard.close
    def Setting_run(self):
        self.Gemini_Setting()
        self.tts_Setting()
        self.guild_Setting()

class soundboard_relation:
    def __init__(self,guild_id):
        self.filepath = os.path.dirname(os.path.abspath(__file__)) + "/soundboard"
        self.soundBoard = os.path.join(self.filepath, f"({guild_id})soundboard.json")
        self.guild_id = guild_id
    def dictionary_setting(self):
        with open(self.soundBoard, 'r') as f:
            data = json.load(f)
        Dictionary = data.copy()
        return Dictionary
    def soundboard_del(self, content, Dictionary):
        returnValue = Dictionary[content]
        Dictionary.pop(content)
        with open(self.soundBoard, 'w') as make_file:
            json.dump(Dictionary, make_file, indent='\t')
        os.remove(f"{self.filepath}/({self.guild_id}){content}.wav")
        with open(self.soundBoard, 'r') as f:
            data = json.load(f)
        Dictionary = data.copy()
        return returnValue
    def mp3json(self, content, url):
        Dictionary = self.dictionary_setting()
        Dictionary[content] = {'url': url,'expiration': 7}
        with open(self.soundBoard, 'w') as make_file:
            json.dump(Dictionary, make_file, indent='\t')
        with open(f"{self.filepath}/({self.guild_id}){content}.wav", "wb") as f:
            response = get(Dictionary[content]['url'])
            f.write(response.content)
        with open(self.soundBoard, 'r') as f:
            data = json.load(f)
        Dictionary = data.copy()

class thread_relation:
    def tokendelete(self,guild_id):
        global tokens, history_messages
        tokens[guild_id] = 0
        if current_model[guild_id] == GPT_model_name:
            history_messages[guild_id] = history_messages[guild_id][:1]
        elif current_model[guild_id] == Gemini_model_name:
            Gemini_model = genAI.GenerativeModel(
                current_model[guild_id],
                system_instruction="This is a scripted prompt, so answer in Korean as best you can. " + role[guild_id][0])
            history_messages[guild_id] = Gemini_model.start_chat(history=[])
        print('token 초기화')
    def tokendelete_thread(self,guild_id):
        global token_thread
        token_thread[guild_id] = False
        print('token 시간 초과')
    def notJoin_thread(self,guild_id):
        global notJoin_thread
        notJoin_thread[guild_id] = False
        print('notJoin 시간 초과')
    def recodedelete_thread(self,wav_path):
        os.remove(wav_path)

class cheack_type:
    @staticmethod
    def is_digit(stri):
        try:
            float(stri)
            return True
        except ValueError:
            return False
    @staticmethod
    def is_url(url):
        url_find = ['https://', 'http://', 'www.']
        for i in url_find:
            if i in url:
                return True
        return False
    @staticmethod
    def is_img(message):
        if len(message.attachments) > 0:
            for file in message.attachments:
                for ext in ['.png', '.jpg', '.jpeg', 'webp', 'gif']:
                    if file.filename.endswith(ext):
                        return [file.url, ext]
        return False

class AIchat_relation:
    def __init__(self, Current_model,guild_id):
        self.Current_model = Current_model
        self.guild_id = guild_id
    def LLM_messages(self, system_content, user_content, temperature_num, messages_type):
        if self.Current_model == GPT_model_name:
            return self.gpt_messages(system_content, user_content, temperature_num, messages_type)
        elif self.Current_model == Gemini_model_name:
            return self.Gemini_messages(system_content, user_content, temperature_num, messages_type)
    def gpt_messages(self, system_content, user_content, temperature_num, messages_type):
        global history_messages
        history_messages[self.guild_id].append({"role": "system", "content": system_content})
        img_token = user_content[1]
        if not user_content[1]:
            history_messages[self.guild_id].append({"role": "user", "content": user_content[0]})
        else:
            history_messages[self.guild_id].append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_content[0]},
                    {"type": "image_url", "image_url": {"url": user_content[1][0]}}
                ]})
            user_content[1] = False

        complention = GPTClient.chat.completions.create(
            model=self.Current_model,
            messages=history_messages[self.guild_id],
            max_tokens=1024,
            temperature=temperature_num
        )
        response = {"role": "assistant", "content": f"{complention.choices[0].message.content}"}
        answer = complention.choices[0].message.content
        if messages_type == "summarize":
            return self.gpt_summarize(complention, response), answer
        else:
            return self.gpt_response(complention, response, img_token), answer

    def Gemini_messages(self, system_content, user_content, temperature_num, messages_type):
        global history_messages
        Gemini_model = self.Gemini_instruction(system_content)
        tokens[self.guild_id] = 0
        if not user_content[1]: content = user_content[0]
        else:
            filepath = os.path.dirname(os.path.abspath(__file__)) + "/gemini"
            with open(f"{filepath}/Gemini.{user_content[1][1]}", "wb") as file:
                response = get(user_content[1][0])
                file.write(response.content)
            image = Image.open(f"{filepath}/Gemini.{user_content[1][1]}")
            content = [user_content[0],image]
            tokens[self.guild_id] += 600
            
        if messages_type == "img_tts":
            complention = Gemini_model.generate_content(
                content,
                generation_config=genAI.types.GenerationConfig(candidate_count=1, temperature=temperature_num),
                safety_settings={
                    'HATE': 'BLOCK_NONE',
                    'HARASSMENT': 'BLOCK_NONE',
                    'SEXUAL': 'BLOCK_NONE',
                    'DANGEROUS': 'BLOCK_NONE'
                }
            )
        else:
            complention = history_messages[self.guild_id].send_message(
                content,
                generation_config=genAI.types.GenerationConfig(candidate_count=1, temperature=temperature_num),
                safety_settings={
                    'HATE': 'BLOCK_NONE',
                    'HARASSMENT': 'BLOCK_NONE',
                    'SEXUAL': 'BLOCK_NONE',
                    'DANGEROUS': 'BLOCK_NONE'
                }
            )
            if messages_type == "summarize":
                Gemini_model = self.Gemini_instruction(role[self.guild_id][0])
                history_messages[self.guild_id] = Gemini_model.start_chat(history=[history_messages[self.guild_id].history.pop(-1)])

            complention_tokens = str(Gemini_model.count_tokens(complention.text))
            complention_tokens = re.sub(r'[^0-9]', '', complention_tokens)
            tokens[self.guild_id] += int(complention_tokens)
        return tokens[self.guild_id], complention.text
    def Gemini_instruction(self, system_content):
        return genAI.GenerativeModel(Gemini_model_name,
                                     system_instruction="This is a scripted prompt, so answer in Korean as best you can." + system_content)
    def gpt_response(self, complention, response, img_token):
        global history_messages
        encoding = tiktoken.encoding_for_model(self.Current_model)
        history_messages[self.guild_id].append(response)
        tokens[self.guild_id] = len(encoding.encode(complention.choices[0].message.content))
        if img_token != False: return tokens[self.guild_id] + 600
        else: return tokens[self.guild_id]
    def gpt_summarize(self, complention, response):
        global history_messages
        encoding = tiktoken.encoding_for_model(self.Current_model)
        history_messages[self.guild_id] = history_messages[self.guild_id][:1]
        history_messages[self.guild_id].append(response)
        tokens[self.guild_id] = len(encoding.encode(complention.choices[0].message.content))
        return tokens[self.guild_id]

class tts_sensing(cheack_type, pitch_conversion):
    def __init__(self,guild_id,Dictionary):
        self.filepath = os.path.dirname(os.path.abspath(__file__))
        self.guild_id = guild_id
        self.Dictionary = Dictionary
        
    def check_bracket(self, split, bracket, lang):
        if self.is_digit(split):
            # Speed change / pitch / length
            if '+' not in split and '-' not in split:
                bracket['split_speed'] = float(split)
            elif '-' not in split:
                if self.is_digit(split[1:]):
                    bracket['pitch'] = float(split)
                if bracket['pitch'] > 3:
                    bracket['pitch'] = 2.99
                elif bracket['pitch'] < 0.1:
                    bracket['pitch'] = 0.1
            else:
                if self.is_digit(split[1:]):
                    bracket['length'] = float(split)*-1
                if bracket['length'] < 0:
                    bracket['length'] = 0
        else:
            # Language change
            if split in lang_list: bracket['split_lang'] = split
            else: bracket['split_lang'] = lang
            # reverse regeneration
            if split.lower() == 're': bracket['reverse'] = True
            # Random
            elif split.lower() == 'rm': bracket['Random'] = True
            elif '!' in split:
                # dB change
                if self.is_digit(split[1:]):
                    bracket['dB'] = split[1:]
            else:
                # Soundboard
                for j in self.Dictionary:
                    if split == j:
                        bracket['Dic'] = [True, j]
        return bracket
    def True_random(self, splittext):
        splitlist = list(splittext)
        random.shuffle(splitlist)
        return "".join(splitlist)
    def True_Dic(self, bracket, path_tts):  
        path_url = self.filepath +f"/soundboard/({self.guild_id})" + bracket[1] + ".wav"
        audioA = AudioSegment.from_file(path_url)
        audioB = AudioSegment.from_file(path_tts)
        mix = audioA.append(audioB, crossfade=150)
        mix.export(path_tts, format="wav")
        
        if self.Dictionary[bracket[1]]['expiration'] < 7:
            self.Dictionary[bracket[1]]['expiration'] = 7
            soundBoard = os.path.join(os.path.dirname(os.path.abspath(__file__)) + "/soundboard", f"({self.guild_id})soundboard.json")
            with open(soundBoard, 'w') as make_file:
                json.dump(self.Dictionary, make_file, indent='\t')
    def True_pitch(self, bracket, path_tts):
        # pitch Processing
        if bracket >= 1:
            if bracket < 1.51:
                K = -4 * bracket + 7.2
                N = int(44100/(bracket*bracket)/K)
            else:
                K = 1.5 * bracket - 1.27
                N = int(44100/(bracket*bracket)*K)
        else:
            N = int(44100*(bracket*bracket)/2.4)
        self.pitch_modulation(N, bracket, path_tts)
        return AudioSegment.from_file(path_tts)

class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        for guild_id in guild_Dict.keys():
            try:
                print(guild_id)
                self.tree.copy_global_to(guild=discord.Object(guild_id))
                await self.tree.sync(guild=discord.Object(guild_id))
            except: pass
intents = discord.Intents.all()
intents.message_content = True
client = MyClient(intents=intents)

# Load environment variables
load_dotenv()
# Discord bot token
discord_TOKEN = os.getenv("DISCORD_TOKEN")
openai_KEY = os.getenv('OPENAI_KEY')
gemini_key = os.getenv('GEMINI_KEY')

# guild related
Initial_Setting().Setting_run()
guild_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"guild.json")
with open(guild_path, 'r') as f:
    guild_data = json.load(f)
guild_Dict = guild_data.copy()
# Chat related
GPT_model_name = "gpt-4o-mini"
Gemini_model_name = "gemini-2.0-flash-exp"
token_Maxsize = 4096
# TTS & Soundboard
parent_folder_id = "1Kgmdjdt37reMbUf350qd2_q0YJmocEvi" # 구글드라이브 조상폴더 위치(id)
gpt_running = False
lang_list = [
    'af','ar','bn','bs','ca','cs','cy','da','de','el',
    'en-au','en-ca','en-gb','en-gh','en-ie','en-in',
    'en-ng','en-nz','en-ph','en-tz','en-uk','en-us',
    'en-za','en','eo','es-es','es-us','es','et','fi',
    'fr-ca','fr-fr','fr','gu','hi','hr','hu','hy','id',
    'is','it','ja','jw','km','kn','ko','la','lv','mk',
    'ml','mr','my','ne','nl','no','pl','pt-br','pt-pt',
    'pt','ro','ru','si','sk','sq','sr','su','sv','sw',
    'ta','te','th','tl','tr','uk','ur','vi','zh-cn','zh-tw']

@client.event
async def on_ready():
    global GPTClient, token_thread, notJoin_thread, notJoin_person, current_model, role, history_messages, tokens, Carry_token
    global former_person, tts_queue, tts_running, people, client
    global guild_Dict, client_id, streamAudio_buffer
    await client.change_presence(status=discord.Status.idle)
    await client.change_presence(activity=discord.Streaming(name="Listen & Speak", url='https://www.google.com/'))
    
    token_thread = {}; notJoin_thread = {}; notJoin_person = {}
    current_model = {}; role = {}; history_messages = {}; tokens= {}; Carry_token = {}
    former_person = {}; tts_queue = {}; tts_running = {}; people = {}
    side_Dict= {}; streamAudio_buffer = {}

    for guild in client.guilds:
        Initial_Setting().SoundBoard_Setting(guild.id).soundboard_Setting()
        side_Dict[guild.id] = []
        
        token_thread[guild.id] = False
        notJoin_thread[guild.id] = False
        notJoin_person[guild.id] = None
        current_model[guild.id] = Gemini_model_name
        role[guild.id] = ['', '']
        history_messages[guild.id] = AIchat_relation(Gemini_model_name,guild.id).Gemini_instruction(role[guild.id][0]).start_chat(history=[])
        tokens[guild.id] = 0
        Carry_token[guild.id] = ['', '']
        former_person[guild.id] = None
        tts_queue[guild.id] = []
        tts_running[guild.id] = False
        people[guild.id] = [['None', 'ko', False]]
        streamAudio_buffer[guild.id] = {}
        
    for guild in guild_Dict:
        int_guild = int(guild)
        if int_guild in side_Dict:
            side_Dict[int_guild] = guild_Dict[guild]
    guild_Dict = side_Dict
        
    with open(guild_path, 'w') as make_file:
        json.dump(side_Dict, make_file, indent='\t')
        
    GPTClient = OpenAI(api_key=openai_KEY)
    genAI.configure(api_key=gemini_key)
    
    print(f'We have logged in as {client.user.name}')
    client_id = client.user.id

    Arrangement.start()
    
@client.event
async def on_message(message):
    global tokens, history_messages, tts_queue, role, gpt_running, Carry_token, notJoin_person, former_person
    global token_thread, notJoin_thread

    temperature = 0.95
    encoding = tiktoken.encoding_for_model(GPT_model_name)
    guild_id = message.author.guild.id
    
    # Ignore bot's own messages
    if message.author.id == client_id:
        return

    # TTS channel logic
    if message.channel.id == guild_Dict[guild_id][0]:
        message.content += "\0"
        # If first char '!' => stop previous
        #! is not said name
        if message.content[0] == '!' and message.author.voice:
            if client.voice_clients != []:
                client.voice_clients[0].pause()
            former_person[guild_id] = message.author
            tts_queue[guild_id] = []
            message.content = '\0' + message.content[1:]

        # queue
        if message.author.voice:
            if client.voice_clients == []:
                await message.author.voice.channel.connect()
                
            voice = client.voice_clients[0]

            # message.content[-1] == '\0'
            if len(message.attachments) == 0 and message.content[-2] != '!':
                tts_queue[guild_id].append(message)
            # mp3 + text
            elif len(message.attachments) > 0 and message.content != '':
                message.content = message.content[0:-1]
                for file in message.attachments:
                    for ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
                        if file.filename.endswith(ext):
                            assist_message = message
                            try:
                                dot = '\0'.join(ext)
                                async with message.channel.typing():
                                    img_tts = [f"40글자 내로 하나의 긴 명사구로 설명(종결어미 빼)(캐릭터,유명인이면 정확한 이름말해)", cheack_type().is_img(assist_message)]
                                    response_token, answer = AIchat_relation(Gemini_model_name,guild_id).Gemini_messages("Obey your orders", img_tts, 0.7, "img_tts")
                                    assist_message.content = f"{answer}쩜{dot}" + assist_message.content
                                    tts_queue[guild_id].append(assist_message)
                                    await assist_message.channel.send(f'{answer}', reference=assist_message)
                            except:
                                await assist_message.channel.send('검열 당함', reference=assist_message)
                    for ext in ['.mp3', '.wav']:
                        if file.filename.endswith(ext):
                            soundboard_relation(guild_id).mp3json(message.content, file.url)
                            await message.channel.send(f'\"**{message.content}**\" 저장')
            
            if not tts_running[guild_id] and len(tts_queue[guild_id]) > 0: await tts_logic(voice,guild_id)
        elif notJoin_person[guild_id] != message.author or type(notJoin_thread[guild_id]) == bool:
            if type(notJoin_thread[guild_id]) != bool:
                notJoin_thread[guild_id].cancel()
            threading_Timer = threading.Timer(60, thread_relation().notJoin_thread,args=(guild_id,))
            threading_Timer.start()
            notJoin_thread[guild_id] = threading_Timer
                
            notJoin_person[guild_id] = message.author
            await message.channel.send("접속중인 음성채널 없음", reference=message)

    # GPT channel logic
    if message.channel.id == guild_Dict[guild_id][1] and not gpt_running:
        gpt_running = True
        Current_model = current_model[guild_id]
        if token_thread[guild_id] == False and tokens[guild_id] > 0:
            Carry_token[guild_id][1] = await message.channel.send("이전 대화를 이어서 하시겠습니까?")
            await Carry_token[guild_id][1].add_reaction('✅')
            await asyncio.sleep(2.5)
            await Carry_token[guild_id][1].clear_reactions()
            if not token_thread[guild_id]:
                thread_relation().tokendelete(guild_id)
            Carry_token[guild_id] = ['', '']

        async with message.channel.typing():
            try:
                system_content = role[guild_id][0]
                # If only image is sent
                if message.content == "":
                    message.content = ("You're Best Expert. \"Accurately\" Analyze and solve the image, "
                                       "answers to Korean. Let's think step-by-step.")
                    system_content += "\n Experts in their field. Its mission is to provide accurate and analytical commentary."
                    temperature = 0.75

                user_content = [f"user_name:{message.author.nick},\"content:{message.content})\"", cheack_type().is_img(message)]
                response_token, answer = AIchat_relation(Current_model,guild_id).LLM_messages(system_content, user_content, temperature, "response")

                # Update tokens
                if Current_model == GPT_model_name:
                    tokens[guild_id] += len(encoding.encode(message.content))
                    tokens[guild_id] += response_token
                elif Current_model == Gemini_model_name:
                    tokens[guild_id] += response_token

                if role[guild_id][1] != '':
                    await message.channel.send(f"Token={tokens[guild_id]}/{int(token_Maxsize)} (**{Current_model}**)\n{answer}\n\nRole : {role[guild_id][1].jump_url}", reference=message)
                else:
                    await message.channel.send(f"Token={tokens[guild_id]}/{int(token_Maxsize)} (**{Current_model}**)\n{answer}", reference=message)
            except:
                try:
                    filepath = os.path.dirname(os.path.abspath(__file__)) + "/gemini"
                    with open(f"{filepath}/LLMtext.txt", "w") as file:
                        file.write(f"({Current_model})\n{answer}")
                    await message.channel.send(f"Token={tokens[guild_id]}/{int(token_Maxsize)}", reference=message)
                    await message.channel.send(file=discord.File(f"{filepath}/LLMtext.txt"))
                    
                except:
                    await message.channel.send("무언가 오류로 인한 미출력")
            try:
                if tokens[guild_id] > int(token_Maxsize):
                    system_content = "You're the ultimate summarizer. Summarize the conversation so far as succinctly as possible without leaving anything out.(1900 characters or less)"
                    user_content = ["대화형식으로 빠짐없이 완벽히 요약해줘.", None]
                    tokens[guild_id], Carry_token[guild_id][0] = AIchat_relation(Current_model,guild_id).LLM_messages(system_content, user_content, 0.55, "summarize")
                    Carry_token[guild_id][1] = await message.channel.send(f"**토큰 초과로 인한 요약 정리**\n내용 확인은 ✅")
                    await Carry_token[guild_id][1].add_reaction('✅')

                # Start token reset timer
                if type(token_thread[guild_id]) != bool:
                    token_thread[guild_id].cancel()
                threading_Timer = threading.Timer(900, thread_relation().tokendelete_thread,args=(guild_id,))
                threading_Timer.start()
                token_thread[guild_id] = threading_Timer
            except:
                await message.channel.send("무언가 오류로 인한 미출력")

            gpt_running = False

@client.tree.command(name='basics_channel', description='토대 채널들을 설정합니다')
@app_commands.describe(tts_channel='tts를 사용할 토대채널', llm_channel='LLM을 사용할 토대채널')
async def basics_channel(interaction: discord.Interaction, tts_channel:discord.TextChannel, llm_channel:discord.TextChannel):
    global guild_Dict
    try:
        guild_id = interaction.guild.id
        guild_Dict[guild_id] = [tts_channel.id,llm_channel.id]
        with open(guild_path, 'w') as make_file:
            json.dump(guild_Dict, make_file, indent='\t')
        await interaction.response.send_message("토대채널을 설정하였습니다.")
    except:
        await interaction.response.send_message('권한 및 기타 오류')

@client.tree.command(name='tokenreset', description='토큰 및 역할을 초기화합니다')
async def tokenreset(interaction: discord.Interaction):
    global tokens, history_messages, role
    try:
        guild_id = interaction.guild.id
        tokens[guild_id] = 0
        role[guild_id][0] = ''
        role[guild_id][1] = ''
        Current_model = current_model[guild_id]
        if Current_model == GPT_model_name:
            history_messages[guild_id] = history_messages[guild_id][:1]
        elif Current_model == Gemini_model_name:
            Gemini_model = AIchat_relation(Current_model,guild_id).Gemini_instruction(role[guild_id][0])
            history_messages[guild_id] = Gemini_model.start_chat(history=[])
        await interaction.response.send_message("토큰 및 역할을 초기화 하였습니다.")
    except:
        await interaction.response.send_message('권한 및 기타 오류')

@client.tree.command(name='tts_setting', description='자신의 기본적인 언어 및 속도를 설정합니다')
@app_commands.describe(lang='언어   (기본)한국어', slow='저속음성 유무(True, (기본)False)')
async def tts_setting(interaction: discord.Interaction, lang: str, slow: str):
    global people
    try:
        guild_id = interaction.guild.id

        if lang not in lang_list:
            lang = 'ko'
        slow_bool = False
        if slow.lower() in ['true', '1']:
            slow_bool = True
        tts_list = [interaction.user, lang, bool(slow_bool)]
        updated = False
        for lst in people[guild_id]:
            if interaction.user in lst:
                lst[1] = lang
                lst[2] = slow_bool
                updated = True
        if not updated:
            people[guild_id].append(tts_list)
        await interaction.response.send_message(f'언어={lang},저속모드={bool(slow_bool)} 완료')
    except:
        await interaction.response.send_message('권한 및 기타 오류')

@client.tree.command(name='assignment_role', description='LLM모델에게 역할을 부여합니다.')
@app_commands.describe(roles='부여할 역할 String')
async def assignment_role(interaction: discord.Interaction, roles: str):
    global role
    try:
        guild_id = interaction.guild.id
        role[guild_id][0] = roles
        role[guild_id][1] = interaction.original_response()
        Current_model = current_model[guild_id]
        if Current_model == GPT_model_name:
            history_messages[guild_id] = history_messages[guild_id][:1]
        elif Current_model == Gemini_model_name:
            Gemini_model = AIchat_relation(Current_model,guild_id).Gemini_instruction(role[guild_id][0])
            history_messages[guild_id] = Gemini_model.start_chat(history=[])
        await interaction.response.send_message(f'"{roles}",역할 부여 완료')
    except:
        await interaction.response.send_message('권한 및 기타 오류')

@client.tree.command(name='soundboard_list', description='사운드보드 리스트를 보여줍니다.')
async def soundboard_list(interaction: discord.Interaction):
    try:
        Dictionary = soundboard_relation(interaction.guild.id).dictionary_setting()
        await interaction.response.send_message("# 사운드 보드 리스트")
        output = '@\n'
        for key,value in Dictionary.items():
            output += f"{key} : {value}\n"
            if len(output) > 1750:
                await interaction.channel.send(output)
                output = '@\n'
        await interaction.channel.send(output)
    except:
        await interaction.response.send_message('권한 및 기타 오류')
        
@client.tree.command(name='soundboard_delete', description='사운드보드 트리거 중 하나를 삭제합니다.')
@app_commands.describe(trigger='삭제할 트리거')
async def soundboard_delete(interaction: discord.Interaction, trigger: str):
    try:
        Dictionary = soundboard_relation(interaction.guild.id).dictionary_setting()
        found = False
        for i in Dictionary:
            if trigger == i:
                soundboard_relation(interaction.guild.id).soundboard_del(i,Dictionary)
                await interaction.response.send_message(f"{trigger}를 삭제하였습니다.")
                found = True
                break
        if not found:
            await interaction.response.send_message(f"{trigger}는 사운드보드 리스트에 없습니다.")
    except:
        await interaction.response.send_message('권한 및 기타 오류')

@client.tree.command(name='model_type', description='LLM 모델을 교체합니다')
@app_commands.describe(model_type=f'0: "{GPT_model_name}", 1: "{Gemini_model_name}"')
async def model_type(interaction: discord.Interaction, model_type: str):
    global current_model, history_messages, tokens
    try:
        guild_id = interaction.guild.id
        if model_type in ['0', GPT_model_name]:
            current_model[guild_id] = GPT_model_name
            history_messages[guild_id] = [{"role": "system", "content": "test"}]
        elif model_type in ['1', Gemini_model_name]:
            current_model[guild_id] = Gemini_model_name
            Gemini_model = AIchat_relation(current_model[guild_id],guild_id).Gemini_instruction(role[guild_id][0])
            history_messages[guild_id] = Gemini_model.start_chat(history=[])
        else:
            await interaction.response.send_message(f"\"{model_type}\"은 해당사항 없습니다.")
            return
        tokens[guild_id] = 0
        await interaction.response.send_message(f"현재 LLM 모델은 **\"{current_model[guild_id]}\"**입니다.")
    except:
        await interaction.response.send_message('권한 및 기타 오류')

@client.tree.command(name='tts_tutorial', description='tts사용의 기본적인 사용법을 보여줍니다.')
async def tts_tutorial(interaction: discord.Interaction):
    try:
        lang = str(lang_list)
        if interaction.user.voice:
            if client.voice_clients == []:
                await interaction.user.voice.channel.connect()
            await interaction.response.send_message("/tts_setting /soundboard_list /soundboard_delete가 있습니다.")
            await interaction.channel.send(f"# !첫 단어가 '!'면 말을 끊을수 마지막 단어면 tts를 실행 안합니다!.\n# 문장 사이사이 명령들을 넣을수 있습니다.")
            await interaction.channel.send(f"1. <ja>언어종류를\n**언어의 종류들**\n{lang}")
            await interaction.channel.send(f"2. <re>역재생을\n3. <rm>애너그램을")
            await interaction.channel.send(f"4. <!2.0>데시벨을\n5. <2.0>속도를 늘릴수도<0.75>줄일수도\n6. <+2.0>진폭을 늘리거나<+0.5>줄이거나")
            await interaction.channel.send(f"7. ms만치<-1000>페이드 인 시킬수 있습니다.\n8. 마지막으로 <트리거>사운드보드 또한 사용 가능합니다\nmp3와 작동시킬 트리거 단어를 작성하여 tts채널에 업로드합니다.")
        else:
            await interaction.response.send_message('접속중인 음성채널 없음')
    except:
        await interaction.response.send_message('권한 및 기타 오류')

@client.event
async def on_reaction_add(react, user):
    global Carry_token, token_thread
    guild_id = user.guild.id
    if user == client.user:
        return
    elif react.emoji == '✅' and Carry_token[guild_id] != ['', '']:
        if gpt_running == True:
            await Carry_token[guild_id][1].edit(content="대화 이어가기✅")
            token_thread[guild_id] = True
        else:
            await Carry_token[guild_id][1].clear_reactions()
            try:
                await Carry_token[guild_id][1].edit(content=f"**<<요약된 내용>>** Token=={tokens[guild_id]}\n{Carry_token[guild_id][0]}")
            except:
                await Carry_token[guild_id][1].edit(content=f"**<<요약된 내용>>** Token=={tokens[guild_id]}\n너무 길어 담아낼수 없음...")
            Carry_token[guild_id] = ['', '']

async def tts_logic(voice,guild_id):
    global former_person, tts_queue, tts_running
    tts_running[guild_id] = True
    message = tts_queue[guild_id].pop(0)

    filepath = os.path.dirname(os.path.abspath(__file__)) + "/stts"
    random_num = str(random.random()).replace('.','')
    path_tts = filepath + f"/{random_num}tts.wav"
    path_pre_mix = filepath + f"/{random_num}pre_mix.wav"
    try:
        lang = 'ko'; slow = False
        for lst in people[guild_id]:
            if message.author in lst:
                lang = lst[1]
                slow = lst[2]

            if cheack_type().is_url(message.content):
                message.content = '링크가 포함된 메시지'
            
            # Handling Brackets
            if message.content.find("<") < message.content.find(">") and "<" in message.content and ">" in message.content:
                bracket = {'split_lang':'ko','dB':0,'pitch':0.0,'length':200,'Random':False,'reverse':False,'Dic':[False,''],'split_speed':1.0}
                if message.content.find("<") == 0:
                    fst= '\0'
                else:
                    fst = message.content[:message.content.find("<")]

                if message.author != former_person[guild_id]:
                    masslist = gTTS(text=f"나{message.author.nick}인데 {fst}", lang='ko', slow=False)
                else:
                    masslist = gTTS(text=fst, lang=lang, slow=slow)

                with open(path_pre_mix, 'wb') as f:
                    masslist.write_to_fp(f)
                    
                Dictionary = soundboard_relation(guild_id).dictionary_setting()
                # Process each bracket
                while "<" in message.content and ">" in message.content:
                    if message.content.find(">") == -1: 
                        break
                    split = message.content[message.content.find("<")+1:message.content.find(">")]
                    bracket = tts_sensing(message.guild.id,Dictionary).check_bracket(split, bracket, lang)
                    message.content = message.content[message.content.find(">")+1:]

                    if message.content[0:1] != '<':
                        splittext = message.content.split('<')[0] if '<' in message.content else message.content
                        if bracket['Random'] == True:
                            splittext = tts_sensing(message.guild.id,Dictionary).True_random(splittext)
                        await asyncio.sleep(0.2)
                        try:
                            masslist = gTTS(text=splittext, lang=bracket['split_lang'], slow=slow)
                            with open(path_tts,'wb') as f:
                                masslist.write_to_fp(f)
                        except:
                            masslist = gTTS(text="\0", lang=bracket['split_lang'], slow=slow)
                            with open(path_tts,'wb') as f:
                                masslist.write_to_fp(f)

                        if bracket['Dic'][0] == True:
                            tts_sensing(message.guild.id,Dictionary).True_Dic(bracket['Dic'],path_tts)
                        audio1 = AudioSegment.from_file(path_tts)
                        audio2 = AudioSegment.from_file(path_pre_mix)

                        if bracket['pitch'] != 0.0 and (bracket['Dic'][0] == True or fst != '\0'):
                            audio1 = tts_sensing(message.guild.id,Dictionary).True_pitch(bracket['pitch'],path_tts)
                        if bracket['reverse'] == True:
                            audio1 = audio1.reverse()
                        if bracket['split_speed'] != 1.0:
                            sound_with_altered_frame_rate = audio1._spawn(
                                audio1.raw_data,
                                overrides={"frame_rate": int(audio1.frame_rate * bracket['split_speed'])})
                            audio1 = sound_with_altered_frame_rate
                        try: mix = audio2.append(audio1 + bracket['dB'], crossfade=bracket['length'])
                        except: mix = audio2.append(audio1 + bracket['dB'], crossfade=0)
                        mix.export(path_pre_mix, format="wav")
                        os.remove(path_tts)

                        # Reset bracket
                        bracket = {'split_lang':'ko','dB':0,'pitch':0.0,'length':200,'Random':False,'reverse':False,'Dic':[False,''],'split_speed':1.0}
            else:
                text = "물음표?" if message.content == '?' else message.content
                tts = gTTS(text=text, lang=lang, slow=slow)
                
                await asyncio.sleep(0.2)
                if message.author != former_person[guild_id]:
                    tts_nick = gTTS(text=f"나{message.author.nick}인데 ", lang='ko', slow=False)
                    with open(path_pre_mix,'wb') as f:
                        tts_nick.write_to_fp(f)
                        tts.write_to_fp(f)
                else:
                    with open(path_pre_mix,'wb') as f:
                        tts.write_to_fp(f)
                    
            former_person[guild_id] = message.author

            voice.play(discord.FFmpegPCMAudio(path_pre_mix))#path_mix
            while voice.is_playing():
                await asyncio.sleep(0.1)
            os.remove(path_pre_mix)
    except:
        tts = gTTS(text="예기치못한 오류", lang = 'ko', slow=False)
        with open(path_pre_mix,'wb') as f:
            tts.write_to_fp(f)
            
    if len(tts_queue[guild_id])==0: tts_running[guild_id] = False
    else: await tts_logic(voice,guild_id)

@tasks.loop(hours=24)
async def Arrangement():
    for guild in client.guilds:
        Dictionary = soundboard_relation(guild.id).dictionary_setting()
        expiration_list = []
        for i in Dictionary:
            if Dictionary[i]['expiration'] > 0: Dictionary[i]['expiration'] -= 1
            else: expiration_list.append(i)
        for i in expiration_list:
            soundboard_relation(guild.id).soundboard_del(i,Dictionary)

        filepath = os.path.dirname(os.path.abspath(__file__)) + "/soundboard"
        soundBoard = os.path.join(filepath, f"({guild.id})soundboard.json")
        with open(soundBoard, 'w') as make_file:
            json.dump(Dictionary, make_file, indent='\t')
    for wav_file in os.scandir(os.path.dirname(os.path.abspath(__file__)) + "/stts"):
        os.remove(wav_file)
            
@client.event
async def on_voice_state_update(member, before, after):
    # 음성채널 인원수 감지
    global former_person, tts_queue
    voice_state = member.guild.voice_client
    guild_id = member.guild.id
    if voice_state is None:
        former_person[guild_id] = None
        tts_queue[guild_id] = []
        return
    if len(voice_state.channel.members) == 1:
        former_person[guild_id] = None
        await voice_state.disconnect()

@client.event
async def on_guild_join(guild):
    global guild_Dict, client, token_thread, notJoin_thread, notJoin_person, current_model, role, history_messages, tokens, Carry_token
    global former_person, tts_queue, tts_running, people, streamAudio_buffer
    guild_Dict[guild.id] = []
    with open(guild_path, 'w') as make_file:
        json.dump(guild_Dict, make_file, indent='\t')
        
    client.tree.copy_global_to(guild=discord.Object(guild.id))
    await client.tree.sync(guild=discord.Object(guild.id))
    
    Initial_Setting(guild.id).Setting_run()

    token_thread[guild.id] = False
    notJoin_thread[guild.id] = False
    notJoin_person[guild.id] = None
    current_model[guild.id] = Gemini_model_name
    role[guild.id] = ['', '']
    history_messages[guild.id] = AIchat_relation(Gemini_model_name,guild.id).Gemini_instruction(role[guild.id][0]).start_chat(history=[])
    tokens[guild.id] = 0
    Carry_token[guild.id] = ['', '']
    former_person[guild.id] = None
    tts_queue[guild.id] = []
    tts_running[guild.id] = False
    people[guild.id] = [['None', 'ko', False]]
    streamAudio_buffer[guild.id] = {}
    
    print(f"{guild.id} join!")

@client.event
async def on_guild_remove(guild):
    global guild_Dict, client, token_thread, notJoin_thread, notJoin_person, current_model, role, history_messages, tokens, Carry_token
    global former_person, tts_queue, tts_running, people, streamAudio_buffer
    try:
        guild_Dict.pop(guild.id)
        with open(guild_path, 'w') as make_file:
            json.dump(guild_Dict, make_file, indent='\t')
            
        token_thread.pop(guild.id)
        notJoin_thread.pop(guild.id)
        notJoin_person.pop(guild.id)
        current_model.pop(guild.id)
        role.pop(guild.id)
        history_messages.pop(guild.id)
        tokens.pop(guild.id)
        Carry_token.pop(guild.id)
        former_person.pop(guild.id)
        tts_queue.pop(guild.id)
        tts_running.pop(guild.id)
        people.pop(guild.id)
        streamAudio_buffer.pop(guild.id)
        
        print(f"{guild.id} remove!")
    except:
        print(f"{guild.id} remove Error")
    
client.run(discord_TOKEN)