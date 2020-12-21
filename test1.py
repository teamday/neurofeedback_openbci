import argparse
import time
import brainflow
import numpy as np
import queue
import time

import pandas as pd
import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

import cv2
import threading
import numpy as np
from ffpyplayer.player import MediaPlayer


class Neural_Feedback:

    def Player_Thread(self):
        video=cv2.VideoCapture(self.video_path)
        player = MediaPlayer(self.video_path)
        brightness = 0
        brightness_delta = 5
        old_signal = self.positive_signal
        signal_timestamp = time.time()
        while True:
            grabbed, frame=video.read()
            signal = self.positive_signal
            if old_signal != signal and (self.last_signal_delta > 2.0 or time.time() - signal_timestamp > 3):
                signal_timestamp=time.time()
                if signal:
                    player.set_volume(1.0)
                    brightness_delta = 5
                else:
                    player.set_volume(0.2)
                    brightness_delta = -5
                old_signal = signal
            brightness += brightness_delta
            if brightness > 254:
                brightness = 255
            if brightness < 50:
                brightness = 50
            cv2.normalize(frame, frame, 0, brightness, cv2.NORM_MINMAX) # 0 - 255
            #player.set_volume(1.0) # 0.0 - 1.0
            audio_frame, val = player.get_frame()
            if not grabbed:
                print("End of video")
                break
            if cv2.waitKey(28) & 0xFF == ord("q"):
                break
            cv2.imshow("Video", frame)
            if val != 'eof' and audio_frame is not None:
                #audio
                img, t = audio_frame
        video.release()
        cv2.destroyAllWindows()
           

   
    def __init__(self, video_path):
        self.video_path = video_path
        BoardShim.enable_dev_board_logger ()
        params = BrainFlowInputParams ()
        params.serial_port = "/dev/ttyUSB0"
        self.board_id = BoardIds.CYTON_DAISY_BOARD.value
        self.sampling_rate = BoardShim.get_sampling_rate (self.board_id)
        self.board = BoardShim (self.board_id, params)
        self.positive_signal = True
        self.last_signal_delta = 0
        self.signals = []

    def dispose(self):
        self.board.stop_stream ()
        self.board.release_session ()

    def config_protocol(self, protocol):
        self.protocol = protocol
        print(f'config {len(protocol)} channels')
        for channel in protocol:
            print(f'channel {channel.channel_inx}')
            for band in channel.bands:
                print(f'band {band.band_range_min}-{band.band_range_max} {"inhibit" if band.is_inhibit else "enchance"}')

    def main (self):      
        self.board.prepare_session ()
        self.board.start_stream ()
        try:
            nfft = DataFilter.get_nearest_power_of_two (self.sampling_rate)
            eeg_channels = BoardShim.get_eeg_channels (self.board_id)
            time.sleep(3)
            signals = []
            player_thread = threading.Thread(target=self.Player_Thread)
            player_thread.start()
            for z in range(90):
                signals.append(self.on_next(eeg_channels, nfft))
                if len(signals) > 3:
                    signals.pop(0)
                avg_signal = sum(signals) / len(signals)
                self.positive_signal = avg_signal < signals[-1]
                self.last_signal_delta = abs(avg_signal - signals[-1])
                if signals[-1] > 9: # min positive signals
                    self.positive_signal = True
                print(f'up {self.last_signal_delta}' if avg_signal < signals[-1] else f'down {self.last_signal_delta}') 
            player_thread.join()
        except Exception as e: 
            print(e)
        return

    def on_next(self, eeg_channels, nfft):
        time.sleep (.3)
        data = self.board.get_current_board_data(max(self.sampling_rate, nfft) + 1) #get_board_data () we are taking ~1 sec data ~3 times a sec
        for channel in self.protocol:
            channel_data = data[eeg_channels[channel.channel_inx]]
            DataFilter.detrend (channel_data, DetrendOperations.LINEAR.value)
            psd = DataFilter.get_psd_welch (channel_data, nfft, nfft // 2, self.sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
            for band in channel.bands:
                band.add_band_power_value(psd, 30)
            #print(f'channel: {channel.channel_inx} positive_signals: {channel.get_positive_signals_count()} avg_power: {channel.get_avg_bands()}')
        return sum([i.get_positive_signals_count() for i in self.protocol])

class Channel_Context:
    def __init__(self, channel_inx, bands):
        self.channel_inx = channel_inx
        self.bands = bands

    def get_positive_signals_count(self):
        result = 0
        for band in self.bands:
            if band.is_signal_positive():
                result += 1
        return result

    def get_avg_bands(self):
        return [i.get_avg_power() for i in self.bands]

class Band_Context:
    def __init__(self, band_range_min, band_range_max, is_inhibit = False, signal_avg_diff_threshold = 0.05, signal_diviation_cut = 3):
        self.band_range_max = band_range_max
        self.band_range_min = band_range_min
        self.power_values = []
        self.band_current_power = 0.0
        self.is_inhibit = is_inhibit
        self.signal_avg_diff_threshold = signal_avg_diff_threshold
        self.signal_diviation_cut = signal_diviation_cut
    
    def is_signal_positive(self):
        avg_power = self.get_avg_power()
        if self.is_inhibit and self.band_current_power < avg_power - (avg_power*self.signal_avg_diff_threshold):
            return True
        if not self.is_inhibit and self.band_current_power > avg_power + (avg_power*self.signal_avg_diff_threshold):
            return True
        return False

    def add_band_power_value(self, psd, max_size):
        value = DataFilter.get_band_power(psd, self.band_range_min, self.band_range_max)
        self.band_current_power = value
        self.power_values.append(value)
        if(len(self.power_values) > max_size):
            self.power_values.pop(0)

    def get_avg_power(self):
        if len(self.power_values) > 0:
            return sum(self.power_values)/len(self.power_values)
        return 0.0

# Mapping
# Fp1 - 1 +
# Fp7 - 9 +
# T3  - 13 +
# T5  - 5
# O1  - 7
# F3  - 11
# C3  - 3
# P3  - 15 +

# test 1 
# left hemisphere activation (1 9 13 15)
# inhibith theta 4-7 and high beta 18-22
# enchance alpha (8-10) and gamma 40 - 55  65-100?
def get_protocol1():
    result = []
    result.append(Channel_Context(1, [Band_Context(4.0, 7.0, True), Band_Context(8.0, 10.0, False), Band_Context(18.0, 22.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(9, [Band_Context(4.0, 7.0, True), Band_Context(8.0, 10.0, False), Band_Context(18.0, 22.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(13, [Band_Context(4.0, 7.0, True), Band_Context(8.0, 10.0, False), Band_Context(18.0, 22.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(15, [Band_Context(4.0, 7.0, True), Band_Context(8.0, 10.0, False), Band_Context(18.0, 22.0, True), Band_Context(30.0, 55.0, False)]))
    return result

if __name__ == "__main__":
    nf = Neural_Feedback("/home/romans/Downloads/TradeOptionsImplyVolatility")
    nf.config_protocol(get_protocol1())
    nf.main()
    nf.dispose()
