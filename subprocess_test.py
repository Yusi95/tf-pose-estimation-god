import subprocess
import time
completed = subprocess.Popen(['python', 'voice_out.py'])

cnt = 0
while True:
    print(cnt)
    cnt += 1
    time.sleep(1)