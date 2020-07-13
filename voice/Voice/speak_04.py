import speech_recognition as sr
import logging
from aip import AipSpeech
import pyaudio
import numpy as np
from pyaudio import PyAudio, paInt16
import Para
import requests
import json
import pypinyin
import utils
from pydub import AudioSegment 
from pydub.playback import play
import cv2
import random
import time

logging.basicConfig(level=logging.DEBUG)

aip_speech = AipSpeech(Para.APP_ID, Para.API_KEY, Para.SECRET_KEY)

num = 0
placenum = 31
filename = "people_atr.txt"

Is_First = True
ConPin = ["sai si ","san si ","sai shi ", "fan si ","si si ","shang si ","30 ","dan shi ","suan shi ","san shi ","sa si ","shai si ","34 ","4s ","suan shi "]
FirstMp3 = "./FirstMp3.mp3"
QAMp3 = "./QAMp3.mp3"
# with open(filename,"r") as file:
AfterId = None
TempAudio = utils.QAudio()

while True:
	try:
		file = open(filename,"r")
		print("file:",file)
		AgeFea,GenderFea,BeforeId,EmotionFea,RotaFea,NumPeoFea,NeighorInf = utils.ReadFea(file)
		print(AgeFea,GenderFea,BeforeId,EmotionFea,RotaFea,NumPeoFea,NeighorInf)
	except:
		continue
	if GenderFea == "man":
		SixStatus = utils.AgeRangeMan(int(AgeFea))
	else:
		SixStatus = utils.AgeRangeWoman(int(AgeFea))
	time.sleep(random.choice(Para.Sleep))
	StepPlace = random.choice(Para.Step)
	if StepPlace == "A":
		# 称谓 + 人数量的唤醒词
		if NumPeoFea == "1":
			Content = random.sample(Para.SinglePerson,1)[0]
			Content = SixStatus + "," + Content
		else:
			Content = random.sample(Para.ManyPeoples,1)[0]
	elif StepPlace == "B":
		# 称谓 + 表情 + 年龄
		EmoStatus = utils.AddStatement(EmotionFea)
		Content = SixStatus + "," + EmoStatus + "," + random.choice(Para.KeyContents) + AgeFea + "岁左右吧！"
	elif StepPlace == "C":
		AllPeo = NeighorInf[1] + NeighorInf[2]
		LeWoman = NeighorInf[0]["Left"]
		RiWoman = NeighorInf[0]["Right"]
		if AllPeo == 0:
			Content = SixStatus + "," + random.choice(Para.OnlyPeople)
		else:
			if (LeWoman + RiWoman) == 0:
				Content = random.choice(Para.OnlyMan)
			else:
				Content = SixStatus + ","
				if LeWoman != 0:
					Content += "我看见你右边有" + str(LeWoman) + "个小姐姐哦！"
				if RiWoman != 0:
					Content += "我看见你左边有" + str(RiWoman) + "个小姐姐哦！"
	else:
		Content = SixStatus + "," + random.choice(Para.TurnHead[RotaFea])		
	Song = utils.Txt2Audio(aip_speech,FirstMp3,Content)
	play(Song)
	file.close()