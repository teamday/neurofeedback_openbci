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
                if old_signal != signal:# and (self.is_last_signal_delta_high or time.time() - signal_timestamp > 2):
                    signal_timestamp=time.time()
                    if signal:
                        brightness_delta = 30 if self.is_last_signal_delta_high else 10
                    else:
                        brightness_delta = -2
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
                if old_signal != signal:
                    if signal:
                        player.set_volume(1.0)
                        signal_timestamp=time.time()
                    elif time.time() - signal_timestamp > 2:
                        player.set_volume(0.4)
                        signal_timestamp=time.time()
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
        self.metrics = {}
        self.channels = {}

    def dispose(self):
        self.board.stop_stream ()
        self.board.release_session ()

    def config_protocol(self, protocol):
        (components, self.metrics) = protocol
        for component in components.values():
            if(component.channel not in self.channels):
                self.channels[component.channel] = []
            self.channels[component.channel].append(component)
        print(f'config {len(self.channels)} channels')
        for channel in self.channels.keys():
            print(f'ch {channel}')
            for component in self.channels[channel]:
                print(f'{component.name}')
        for metric in self.metrics:
            print(f'metric {metric.metric_type} {metric.component1.name} {metric.component2.name}')

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
            signal_freq_coeff = 1.1 # auto adjustable coefficient?
            high_signal_freq_coeff = 1.5
            data_log_file = open(f'log2-{time.time()}.csv', 'a')
            print_bands = []
            for channel in self.channels.keys():
                print_bands.append(','.join([b.name for b in self.channels[channel]]))
            data_log_file.write(f'time,metrics_sum,signal,high_signal,{",".join(print_bands)}')
            metrics_hist = []
            while self.player_is_playing:
                time.sleep (.3)
                self.on_next(eeg_channels, nfft)
                metrics_sum = 0.0
                for metric in self.metrics:
                    metrics_sum += metric.get_metric()
                metrics_hist.append(metrics_sum)
                if len(metrics_hist) > 50:
                    metrics_hist.pop(0)

                avg_metrics_hist = sum(metrics_hist)/len(metrics_hist)
                self.positive_signal = avg_metrics_hist < metrics_sum * signal_freq_coeff

                self.is_last_signal_delta_high = False
                if self.positive_signal and avg_metrics_hist < metrics_sum * high_signal_freq_coeff:
                    self.is_last_signal_delta_high = True
                
                print(f'{self.positive_signal} {avg_metrics_hist} < {metrics_sum*signal_freq_coeff}')
                print_bands = []
                for channel in self.channels.keys():
                    print_bands.append(','.join([str(b.band_current_power) for b in self.channels[channel]]))
                
                log_line = f'\n{time.asctime(time.gmtime(time.time()))},{metrics_sum},{self.positive_signal},{self.is_last_signal_delta_high},{",".join(print_bands)}'

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
        for channel in self.channels.keys():
            channel_data = data[channel]
            DataFilter.detrend (channel_data, DetrendOperations.LINEAR.value)
            psd = DataFilter.get_psd_welch (channel_data, nfft, nfft // 2, self.sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
            for component in self.channels[channel]:
                component.add_band_power_value(psd)

class Metric:
    def __init__(self, metric_type, is_inhibit, component1, component2):
        self.metric_type = metric_type
        self.is_inhibit = is_inhibit
        self.component1 = component1
        self.component2 = component2

    def get_metric(self):# > 1.0
        if self.metric_type == 1: # absolute component metric
            c = self.component1
            if self.is_inhibit and c.precentile_power > 0:
                return c.band_current_power / c.precentile_power
            elif not self.is_inhibit and c.band_current_power > 0:
                return c.precentile_power / c.precentile_power
        elif self.metric_type == 2 and self.component2.band_current_power > 0: # ratio metric
            return self.component1.band_current_power / self.component2.band_current_power
        elif self.metric_type == 3 and self.component1.precentile_power > 0 and self.component2.precentile_power > 0: # coherence metric    TODO figure out how to do coherence and phase check correctly
            return (self.component1.band_current_power/self.component2.precentile_power) + (self.component2.band_current_power/self.component2.precentile_power)
        elif self.metric_type == 4 and self.component1.precentile_power > 0 and self.component2.precentile_power > 0: # phase metric
            a = abs(self.component1.band_current_power-self.component1.precentile_power)/self.component1.precentile_power
            b = abs(self.component2.band_current_power-self.component2.precentile_power)/self.component2.precentile_power
            return a + b
        return 0.0
          

"""
Band config and avg power buffer
"""
class MetricComponent:
    def __init__(self, channel, band_range_min, band_range_max, signal_diviation_cut = 20, band_buffer_size = 3000): 
        self.band_range_max = band_range_max
        self.band_range_min = band_range_min
        self.band_buffer_size = band_buffer_size # 10 = 1sec
        self.power_values = []
        self.band_current_power = 0.0
        self.precentile_power = 0.0
        self.signal_diviation_cut = signal_diviation_cut
        self.name = f'ch{channel}:{band_range_min}-{band_range_max}Hz'
        self.channel = channel

    def _add_band_power_value(self, value):
        self.band_current_power = value
        self.power_values.append(value)
        if(len(self.power_values) > self.band_buffer_size):
            self.power_values.pop(0)
        self.precentile_power = np.percentile(self.power_values, 85)

    def add_band_power_value(self, psd):
        value = DataFilter.get_band_power(psd, self.band_range_min, self.band_range_max)
        if self.precentile_power == 0.0 or value < self.precentile_power * self.signal_diviation_cut:
            self._add_band_power_value(value)

def get_protocol1():
    metric_components = {}
    metric_components['ch11-4-8Hz'] = MetricComponent(11, 4.0, 8.0)
    metric_components['ch13-4-8Hz'] = MetricComponent(13, 4.0, 8.0)
    metric_components['ch15-4-8Hz'] = MetricComponent(15, 4.0, 8.0)
    metric_components['ch5-4-8Hz'] = MetricComponent(5, 4.0, 8.0)
    metric_components['ch11-8-10Hz'] = MetricComponent(11, 8.0, 10.0)
    metric_components['ch13-8-10Hz'] = MetricComponent(13, 8.0, 10.0)
    metric_components['ch15-8-10Hz'] = MetricComponent(15, 8.0, 10.0)
    metric_components['ch5-8-10Hz'] = MetricComponent(5, 8.0, 10.0)

    metrics = []
    metrics.append(Metric(2, False, metric_components['ch11-4-8Hz'], metric_components['ch11-8-10Hz']))
    metrics.append(Metric(2, False, metric_components['ch13-4-8Hz'], metric_components['ch13-8-10Hz']))
    metrics.append(Metric(2, False, metric_components['ch15-4-8Hz'], metric_components['ch15-8-10Hz']))
    metrics.append(Metric(2, False, metric_components['ch5-4-8Hz'], metric_components['ch5-8-10Hz']))

    return (metric_components, metrics)


if __name__ == "__main__":
    nf = Neural_Feedback('/home/romans/Downloads/y2mate.com - The basis of statistics; necessity and levels of statistics_1080p.mp4')
    nf.config_protocol(get_protocol1())
    nf.main()
    nf.dispose()
