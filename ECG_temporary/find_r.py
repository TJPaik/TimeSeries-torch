import numpy as np
from scipy.signal import convolve, cwt, peak_widths, ricker, find_peaks, medfilt
from scipy.ndimage import gaussian_filter1d


def find_r(standardized_wave_data):
    # ECG - 500 Hz, mainly Lead 2 data
    def total_variation_ft(wave_data, window_size):
        diff = np.abs(np.diff(wave_data))
        return convolve(
            diff,
            np.linspace(1, 1, window_size),
            mode='same'
        )

    def set_threshold(wave_data):
        len_data = len(wave_data)
        copy_wave_data = wave_data.copy()

        def local_delete(index, param=140):
            copy_wave_data[max(0, index - param): min(len_data, index + param)] = 0

        copy_wave_data[:100] = 0
        copy_wave_data[-100:] = 0
        peak_indices, peak_heights = find_peaks(copy_wave_data, distance=140, height=-1e5)
        peak_heights = peak_heights["peak_heights"]
        sorted_peaks = np.argsort(peak_heights)

        for i in range(3):
            local_delete(peak_indices[sorted_peaks[-i - 1]])

        return copy_wave_data.max()

    debase = standardized_wave_data - medfilt(standardized_wave_data, 75)
    de_noised_debase = gaussian_filter1d(debase, 4)

    total_variation = total_variation_ft(de_noised_debase, 15)
    normalize_number = set_threshold(total_variation)
    total_variation = total_variation / normalize_number

    de_noised = gaussian_filter1d(standardized_wave_data, 4)
    cwt_de_noised = cwt(de_noised, ricker, np.arange(1, 31))

    wavelet_data = np.abs(cwt_de_noised[1, :])
    normalize_number = set_threshold(wavelet_data)
    wavelet_data = wavelet_data / normalize_number

    intersection = np.array([
        min(total_variation[i], wavelet_data[i]) for i in range(len(total_variation))
    ])

    height_criterion = 0.4
    first_try_peaks = find_peaks(intersection, distance=140, height=height_criterion)[0]

    all_peaks, all_prominences = find_peaks(standardized_wave_data, width=0, prominence=0, wlen=70)
    all_prominences = all_prominences["prominences"]
    half_width = peak_widths(standardized_wave_data, all_peaks, rel_height=0.5, wlen=70)

    second_try_peaks = []
    for el in first_try_peaks:
        interesting_start = max(0, el - 50)
        interesting_end = min(len(standardized_wave_data), el + 50)
        peak_height = -1e2
        peak_idx = -1
        for peak_candidate, prominence, width in zip(all_peaks, all_prominences, half_width[0]):
            if interesting_start <= peak_candidate <= interesting_end and \
                    prominence / width > 0.05 and prominence > 0.3:
                if peak_height <= standardized_wave_data[peak_candidate]:
                    peak_idx = peak_candidate
                    peak_height = standardized_wave_data[peak_candidate]
        if peak_idx == -1:
            second_try_peaks.append(
                interesting_start + np.argmax(standardized_wave_data[interesting_start:interesting_end]))
        # print("!!!!!!!!!!!!!!!!")
        else:
            second_try_peaks.append(peak_idx)

    return np.array(second_try_peaks).astype(int)
