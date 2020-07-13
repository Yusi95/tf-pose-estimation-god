import Para
import cv2
import numpy as np
import random
import dlib
from dlib import rectangle
from scipy import optimize
from pydub import AudioSegment
import pypinyin
from pyaudio import PyAudio, paInt16
import wave
import os
import json
import requests
from PIL import Image, ImageDraw, ImageFont
import jieba

def AgeRangeMan(Age):
    if (Age >= 0) and (Age <= 35):
        return "小哥哥"
    elif (Age > 35) and (Age <= 60):
        return "先生"
    else:
        return "叔叔"

def AgeRangeWoman(Age):
    if (Age >= 0) and (Age <= 35):
        return "小姐姐"
    elif (Age > 35) and (Age <= 60):
        return "女士"
    else:
        return "阿姨"
def AddStatement(Emotion):
    EmotionDict = Para.EmoDic
    EmoChoice = random.choice(EmotionDict[Emotion])
    return EmoChoice

def topinyin(word):
    pin = ""
    for i in pypinyin.pinyin(word,style = pypinyin.NORMAL):
        pin  = pin + "".join(i) + " "
    return pin

def Txt2Audio(API,Path,TxT):
    ToAudio = API.synthesis(TxT, 'zh', 1, {'vol': 5, 'per': 4, 'spd': 6})
    if not isinstance(ToAudio, dict):
        with open(Path, 'wb') as f:
            f.write(ToAudio)
    song = AudioSegment.from_mp3(Path)
    return song

def ReadFea(File):
    InfContent = File.readlines()[-1].replace("\n","").split("-")
    AgeFea = InfContent[0].split(":")[1]
    GenderFea = InfContent[1].split(":")[1]
    BeforeId = InfContent[2].split(":")[1]
    EmotionFea = InfContent[3].split(":")[1]
    RotaFea = InfContent[4].split(":")[1]
    NumPeoFea = InfContent[5].split(":")[1]
    NeighorInf = eval(InfContent[6][11:])
    return AgeFea,GenderFea,BeforeId,EmotionFea,RotaFea,NumPeoFea,NeighorInf

class QAudio:
    def __init__(self):
        self.CHUNK = Para.CHUNK
        self.FORMAT = paInt16
        self.CHANNELS = Para.CHANNELS
        self.RATE = Para.RATE
        self.RECORD_SECONDS = Para.RECORD_SECONDS
        self.SavePath = Para.SaveAudioMP3
        self.paudio = PyAudio()
        self.stream = self.paudio.open(format=self.FORMAT,
                                       channels=self.CHANNELS,
                                       rate=self.RATE,
                                       input=True,
                                       frames_per_buffer=self.CHUNK)

    def au_st(self):
        return self.paudio,self.stream
        
        

    def read(self):
        data = self.stream.read(self.CHUNK)
        return data

    # def close(self):
    #     self.stream.close()
    #     self.paudio.terminate()
    def ReFrames(self,Render,w,h):
        Frames = []
        N = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)
        PerAng = 360/N
        for i in range(0,N):
            start = i*PerAng
            end = (i+1)*PerAng
            Render = cv2ImgAddText(Render,Para.Listen,int(w*6/17),int(h*6/21),(0,0,0),100)
            cv2.ellipse(Render,(int(w/2),int(h/3)),(250,250),0,start,end,(255,255,0),-1)
            cv2.imshow("Render",Render)
            cv2.waitKey(1)
            Data = self.stream.read(self.CHUNK)
            Frames.append(Data)
            # 这里需要添加可视化内容，根据语音时长
        # self.stream.stop_stream()
        # self.stream.close()
        # self.paudio.terminate()

        wf = wave.open(self.SavePath,"wb")
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.paudio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(Frames))
        wf.close()
        print("Saved Mp3 audio successfully")

def wav_to_pcm(wav_file):
    # 假设 wav_file = "音频文件.wav"
    # wav_file.split(".") 得到["音频文件","wav"] 拿出第一个结果"音频文件"  与 ".pcm" 拼接 等到结果 "音频文件.pcm"
    pcm_file = "%s.pcm" %(wav_file.split(".")[0])

    # 就是此前我们在cmd窗口中输入命令,这里面就是在让Python帮我们在cmd中执行命令
    os.system("ffmpeg -y  -i %s  -acodec pcm_s16le -f s16le -ac 1 -ar 16000 %s"%(wav_file,pcm_file))

    return pcm_file

def Audio2Txt(mp3file,aip):
    PcmFile = wav_to_pcm(mp3file)
    with open(PcmFile,"rb") as fp:
        FileContext = fp.read()
    Ret = aip.asr(FileContext,"wav",16000,{'dev_pid': 1537, })
    return Ret

def SendFuDan(Text):
    Formata = json.dumps({
                "sessionId":Para.SessionID,
                "question":Text,
              })
    null = None
    respones = requests.post(url = Para.Address,data = Formata,headers = Para.Post_header,timeout = Para.TimeOut)
    ResTxt = eval(respones.text)["answer"]
    return ResTxt
def KeyWords(Content):
    Seg = jieba.cut(Content)
    Content = " ".join(Seg).split( )
    for C in Content:
        if C in Para.KeyWords:
            return True
    return False


if __name__ == "__main__":
    a = LeaveReturn([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    print(a)
    # cap = cv2.VideoCapture(0)
    # predictor_path = "./shape_predictor_68_face_landmarks.dat"
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(predictor_path)
    # mean3DShape, blendshapes, idxs3D, idxs2D = load3DFaceModel("./candide.npz")
    # projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])
    # shapes2D = None
    # while True:
    #     ret, cameraImg = cap.read()
    #     R = GetRotation(cameraImg,projectionModel,detector,predictor,mean3DShape)
    #     print("R:",R)
    #     cv2.imshow("img",cameraImg)
    #     key = cv2.waitKey(1)
    #     if key == ord("q"):
    #         break