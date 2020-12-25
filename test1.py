import argparse
import time
import brainflow
import numpy as np
import queue
import time
import collections

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

    def cv2_video_thread(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_GUI_EXPANDED)
        video=cv2.VideoCapture(self.video_path)
        signal_timestamp = time.time()
        old_signal = self.positive_signal
        try:
            brightness = 0
            brightness_delta = 5
            while self.player_is_playing:
                grabbed, frame=video.read()
                video_msec = video.get(cv2.CAP_PROP_POS_MSEC)
                audio_current_time = time.time() - self.audio_start_time_sec
                if  audio_current_time*1000 > video_msec + 200:
                    frame_pos = video.get(cv2.CAP_PROP_POS_FRAMES)
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos + 10)
                    #print(f'audio_current_time_ms {audio_current_time}  video_time {video_msec/1000} frame {frame_pos}')
                elif audio_current_time*1000 < video_msec - 100:
                    t = (video_msec - audio_current_time*1000)/1500
                    time.sleep(t)
                    #print(f'video is too fast {t}')
                if not grabbed:
                    break
                if cv2.waitKey(28) & 0xFF == ord("q"):
                    self.player_is_playing = False
                    break
                signal = self.positive_signal
                if old_signal != signal and (self.is_last_signal_delta_high or time.time() - signal_timestamp > 2):
                    signal_timestamp=time.time()
                    if signal:
                        brightness_delta = 10
                    else:
                        brightness_delta = -5
                    old_signal = signal
                brightness += brightness_delta
                if brightness > 254:
                    brightness = 255
                if brightness < 50:
                    brightness = 50

                cv2.normalize(frame, frame, 0, brightness, cv2.NORM_MINMAX) # 0 - 255
                cv2.imshow(self.windowName, frame)
              
        except Exception as e: 
            print(e)
        finally:
            self.player_is_playing = False
            video.release()
            cv2.destroyAllWindows()

    def audio_thread(self):
        player = MediaPlayer(self.video_path, ff_opts = {'vn':True})
        old_signal = self.positive_signal
        signal_timestamp = time.time()
        try:
            player.set_volume(1.0)
            self.audio_start_time_sec = time.time()
            while self.player_is_playing:
                signal = self.positive_signal
                if old_signal != signal and (self.is_last_signal_delta_high or time.time() - signal_timestamp > 2):
                    signal_timestamp=time.time()
                    if signal:
                        player.set_volume(1.0)
                    else:
                        player.set_volume(0.4)
                    old_signal = signal
                time.sleep(0.1)
        except Exception as e: 
            print(e)
        finally:
            self.player_is_playing = False
            player.close_player()

   
    def __init__(self, video_path):
        self.windowName = "Neurofeedback"
        self.video_path = video_path
        self.audio_start_time_sec = time.time()
        BoardShim.enable_dev_board_logger ()
        params = BrainFlowInputParams ()
        params.serial_port = "/dev/ttyUSB0"
        self.board_id = BoardIds.CYTON_DAISY_BOARD.value
        self.sampling_rate = BoardShim.get_sampling_rate (self.board_id)
        self.board = BoardShim (self.board_id, params)
        self.player_is_playing = False
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
            cv2_thread = threading.Thread(target=self.cv2_video_thread)
            audio_thread = threading.Thread(target=self.audio_thread)
            cv2_thread.start()
            audio_thread.start()
            self.player_is_playing = True
            signal_freq_coeff = .3 # auto adjustable coefficient?
            high_signal_freq_coeff = .15
            data_log_file = open(f'log-{time.time()}.csv', 'a')
            while self.player_is_playing:
                time.sleep (.1)
                bands_signals = self.on_next(eeg_channels, nfft)
                positive_signals_sum = 0.0
                negative_signals_sum = 0.0
                for i in bands_signals.keys():
                    if bands_signals[i] > 0:
                        positive_signals_sum += bands_signals[i]
                    else:
                        negative_signals_sum += bands_signals[i]

                self.positive_signal = abs(negative_signals_sum) < positive_signals_sum * signal_freq_coeff 

                self.is_last_signal_delta_high = False
                if self.positive_signal and positive_signals_sum*high_signal_freq_coeff > abs(negative_signals_sum):
                    self.is_last_signal_delta_high = True
                
                print_bands = []
                for proto in self.protocol:
                    print_bands.append(f'{eeg_channels[proto.channel_inx]},'+ ",".join([str(b.band_current_power) for b in proto.bands]))
                
                log_line = f'\n{time.asctime(time.gmtime(time.time()))},{positive_signals_sum},{negative_signals_sum},{self.positive_signal},{self.is_last_signal_delta_high},{",".join(print_bands)}'

                data_log_file.write(log_line)
                
            data_log_file.close()
            audio_thread.join()
            cv2_thread.join()
        except Exception as e: 
            print(e)
            self.player_is_playing = False
        return

    def on_next(self, eeg_channels, nfft):
        data = self.board.get_current_board_data(max(self.sampling_rate, nfft) + 1) #get_board_data () we are taking ~1 sec data ~10 times a sec
        bands_sum = collections.defaultdict(float)
        for channel in self.protocol:
            channel_data = data[eeg_channels[channel.channel_inx]]
            DataFilter.detrend (channel_data, DetrendOperations.LINEAR.value)
            psd = DataFilter.get_psd_welch (channel_data, nfft, nfft // 2, self.sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
            for band in channel.bands:
                band.add_band_power_value(psd)
                bands_sum[band.name] += band.get_signal()
        return bands_sum

class Channel_Context:
    def __init__(self, channel_inx, bands):
        self.channel_inx = channel_inx - 1
        self.bands = bands

"""
Band config and avg power buffer
"""
class Band_Context:
    def __init__(self, band_range_min, band_range_max, is_inhibit = False, signal_diviation_cut = 25, band_buffer_size = 300): 
        self.band_range_max = band_range_max
        self.band_range_min = band_range_min
        self.band_buffer_size = band_buffer_size # 10 = 1sec
        self.power_values = []
        self.band_current_power = 0.0
        self.is_inhibit = is_inhibit
        self.signal_diviation_cut = signal_diviation_cut
        self.name = f'{band_range_min}-{band_range_max}'

    def _add_band_power_value(self, value):
        self.band_current_power = value
        self.power_values.append(value)
        if(len(self.power_values) > self.band_buffer_size):
            self.power_values.pop(0)

    def add_band_power_value(self, psd):
        value = DataFilter.get_band_power(psd, self.band_range_min, self.band_range_max)
        self._add_band_power_value(value)

    def get_signal(self):
        precentile_power = np.percentile(self.power_values, 85)
        if precentile_power > 0:
            if self.is_inhibit:
                return (precentile_power - self.band_current_power) / precentile_power
            else:
                return (self.band_current_power - precentile_power) / precentile_power
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
# current constraint band array should be same because count of positive/negative signals based on bands aggregation from channels
def get_protocol1():
    result = []
    result.append(Channel_Context(1, [Band_Context(4.0, 7.0, True), Band_Context(8.0, 10.0, False), Band_Context(18.0, 22.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(9, [Band_Context(4.0, 7.0, True), Band_Context(8.0, 10.0, False), Band_Context(18.0, 22.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(13, [Band_Context(4.0, 7.0, True), Band_Context(8.0, 10.0, False), Band_Context(18.0, 22.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(15, [Band_Context(4.0, 7.0, True), Band_Context(8.0, 10.0, False), Band_Context(18.0, 22.0, True), Band_Context(30.0, 55.0, False)]))
    return result

# test 2
# right hemi ()
# P4 - 16 T4 - 14 T6 - 6 C4 - 4 O2 - 8
# enchance (4 - 8, 12 - 18, 30 - 50) inhibit (17 - 30)
# 
def get_protocol2():
    result = []
    result.append(Channel_Context(4, [Band_Context(4.0, 7.0, False), Band_Context(12.0, 18.0, False), Band_Context(18.0, 30.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(6, [Band_Context(4.0, 7.0, False), Band_Context(12.0, 18.0, False), Band_Context(18.0, 30.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(8, [Band_Context(4.0, 7.0, False), Band_Context(12.0, 18.0, False), Band_Context(18.0, 30.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(14, [Band_Context(4.0, 7.0, False), Band_Context(12.0, 18.0, False), Band_Context(18.0, 30.0, True), Band_Context(30.0, 55.0, False)]))
    result.append(Channel_Context(16, [Band_Context(4.0, 7.0, False), Band_Context(12.0, 18.0, False), Band_Context(18.0, 30.0, True), Band_Context(30.0, 55.0, False)]))
    return result

# Data flow
# raw data -> get_psd_welch for channel -> get_band_power for band within channel -> spike cut -> add to avg buffer and update band_current_power ->
# get_signal() -> aggregate + and - separately across channels -> stimulus = aggregate(+) > |aggregate_(-)|
if __name__ == "__main__":
    nf = Neural_Feedback('/home/romans/Downloads/y2mate.com - Negotiation Skills How to harness trust, empathy and the word No by Chris Voss_1080p.mp4')
    nf.config_protocol(get_protocol1())
    nf.main()
    nf.dispose()
