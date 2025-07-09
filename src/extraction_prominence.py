import sys
sys.path.append("/home/anvithak/quantifying-redundancy")
from src.data.components.feature_extractors import ProsodyFeatureExtractor

def extract():
    lang = 'en'
    LAB_ROOT = '/home/anvithak/quantifying-redundancy/languages/en/aligned'
    #f"/Users/cui/Documents/uzh/PhD/Projects/Prosody/crosslingual-redundancy/languages/{lang}/aligned/{category}"
    WAV_ROOT = '/home/anvithak/quantifying-redundancy/languages/en/wav_files'
    #f"/Users/cui/Documents/uzh/PhD/Projects/Prosody/crosslingual-redundancy/languages/{lang}/wav_files/{category}"
    DATA_CACHE = '/home/anvithak/quantifying-redundancy/languages/en/cache_prominence'
    # f"/Users/cui/Documents/uzh/PhD/Projects/Prosody/crosslingual-redundancy/languages/{lang}/cache_prominence"

    # extractor = ProsodyFeatureExtractor(
    # lab_root=LAB_ROOT,
    # wav_root=WAV_ROOT,
    # phoneme_lab_root=LAB_ROOT,
    # data_cache=DATA_CACHE,
    # language = lang,
    # extract_f0=True,
    # f0_stress_localizer = None,
    # celex_path="",
    # f0_n_coeffs=4,
    # f0_mode='curve',#"dct" for paramarized
    # )

    extractor = ProsodyFeatureExtractor(
    lab_root=LAB_ROOT,
    wav_root=WAV_ROOT,
    phoneme_lab_root=LAB_ROOT,
    data_cache=DATA_CACHE,
    language = lang,
    extract_f0=False,
    extract_prominence=True,
    prominence_mode='curve'
    )

    print('Feature extraction complete.')
    return extractor


def main():
    extractor = extract()
    return extractor


if __name__ == "__main__":
    main()
