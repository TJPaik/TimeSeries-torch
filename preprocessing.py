from scipy.signal import medfilt


def preprocess(wave_data):
    base = medfilt(wave_data, 101)
    base = medfilt(base, 301)
    de_based = wave_data - base
    return de_based
