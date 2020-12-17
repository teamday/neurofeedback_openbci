import argparse
import time
import brainflow
import numpy as np

import pandas as pd
import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

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
# trashhold + point based
# band_power_1 less (1:10, 9:3, 13:2, 15:2)
# band_power_2 less (1:2,  9:1, 13:1, 15:1)
# band_power_3 more (1:2,  9:2,  13:2,  15:2)
# band_power_4 more (1:2,  9:2,  13:2,  15:1)
m = {1:[10,2,2,2],9:[3,1,2,2],13:[2,1,2,2],15:[2,1,2,1]}
def main ():
    BoardShim.enable_dev_board_logger ()
    params = BrainFlowInputParams ()
    params.serial_port = "/dev/ttyUSB0"
    board_id = BoardIds.CYTON_DAISY_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate (board_id)
    board = BoardShim (board_id, params)
    board.prepare_session ()
    board.start_stream ()
    try:
        nfft = DataFilter.get_nearest_power_of_two (sampling_rate)
        eeg_channels = BoardShim.get_eeg_channels (board_id)

        time.sleep(10)
        ch_bands = {}

        for z in range(30):
            time.sleep (.3)
            data = board.get_current_board_data(max(sampling_rate, nfft) + 1) #get_board_data ()
            for ch in (1, 9, 13, 15):
                DataFilter.detrend (data[eeg_channels[ch]], DetrendOperations.LINEAR.value)
                psd = DataFilter.get_psd_welch (data[eeg_channels[ch]], nfft, nfft // 2, sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
                band_power_1 = DataFilter.get_band_power (psd, 4.0, 7.0)
                band_power_2 = DataFilter.get_band_power (psd, 18.0, 22.0)
                band_power_3 = DataFilter.get_band_power (psd, 8.0, 10.0)
                band_power_4 = DataFilter.get_band_power (psd, 40.0, 55.0)
                ch_bands[ch] = [band_power_1, band_power_2, band_power_3, band_power_4]
                #band_power_5 = 0 #DataFilter.get_band_power (psd, 65.0, 100.0)

            print (f'{ch_bands[1]}')

    except:
        pass

    board.stop_stream ()
    board.release_session ()





if __name__ == "__main__":
    main ()
