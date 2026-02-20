import winsound # For windows  only
import threading
import time

class CarSoundEffects:
    """Add sound effects to ca detection (Windows onlty). """

    def __init__(self):
        self.latest_sound_time = 0
        self.sound_cooldown = 2 # seconds between sounds    

        #Car sounds (Windows beep frequencies)
        self.sounds = {
            'car' : (500, 200), # frequency in Hz, duration in ms
            'truck' : (300, 300),
            'bus' : (200, 400),
            'motorcycle' : (800, 150),
            'detected' : (600, 100) # frequency in Hz, duration in ms
        }
    def play_detection_sound(self, car_type= 'detected'):
        """Play a sound when car is detected."""
        current_time = time.time()
        
        
        #Prevent sound spam
        if current_time - self.last_sound_time < self.sound_cooldown:
            return
            
        self.last_sound_time = current_time

        #Play sound in background thread
        def play():
            freq, duration = self.sounds.get(car_type, (500, 200))
            windsound.Beep(freq, duration)

        threading.Thread(target=play, daemon = True).start()

    def play_horn(self):
        """Play a horn sound. """
        windsound.Bepp(400, 500) #Honk!
        winsound.Beep(400, 500)


#Usage example"
sound = Car SoundEffects()
sound.play_detection_sound('car') # Play car detection sound
time.sleep(1) # Wait before playing another sound


            