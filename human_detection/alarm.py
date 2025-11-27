import simpleaudio as sa
import threading

class AlarmManager:
    def __init__(self, sound_file):
        self.sound = sa.WaveObject.from_wave_file(sound_file)
    
    def play_alarm(self):
        threading.Thread(target=self.sound.play).start()
