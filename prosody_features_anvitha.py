import subprocess
import os
import time
import sys
import glob
import csv
import ast
import pickle
import string
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import logging

"""
This file contains functions to automatically extract pitch and energy contour information from Praat as well as a duration calculation function, and graph the resulting output.

"""

#####################################################################################################################################################################################################################################
## This section has functions for running and processing data from the Montreal Forced Aligner. 
## Before running any of these functions, make sure Montreal Forced Aligner (MFA) is installed on your computer.
## To install MFA in a conda environment, run the following two lines:
## conda config --add channels conda-forge
## conda install montreal-forced-aligner
## or follow the instructions on https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html
## These functions together run the command: "mfa align ~/mfa_data/my_corpus english_us_arpa english_us_arpa ~/mfa_data/my_corpus_aligned"
## my_corpus should consist of audio and txt files with the transcript of the relevant audio in them

def run_mfa_validate(audio_dir, language_model_dir, dictionary, new_mfa_data=False):
    """
    This function validates an audio corpus to make sure there are valid inputs with no issues and that the Montreal Forced Aligner can run smoothly on it. 
    
    Parameters:
    - audio_dir: The path to the folder of audio files and associated txt files to align. Make sure the audio and txt files have the same name before the extension. For example: audio_1.wav and audio_1.txt
    - language_model_dir: The acoustic model to use for alignment. For standard American English, use "english_us_arpa", else see https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
    - dictionary: The pronunciation dictionary for the langauge of your corpus. For standard American English, use "english_us_arpa", else see https://mfa-models.readthedocs.io/en/latest/dictionary/index.html
    
    Output:
    - True if MFA can run successfully, False otherwise. Errors from the process, if any, will be printed.
    """
    # Construct the MFA validate command for the command line
    if new_mfa_data:
        validate_command = [
        "mfa", "validate", audio_dir, language_model_dir, dictionary, "--clean"
    ]
    else:    
        validate_command = [
            "mfa", "validate", audio_dir, language_model_dir, dictionary
        ]
    
    try:
        # Run the validation command using subprocess
        result = subprocess.run(validate_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Print the output and errors (if any) from the validation process
        print("Validation successful!")
        print(result.stdout.decode())
        print(result.stderr.decode())
        
        return True  # Return True if the validation is successful
        
    except subprocess.CalledProcessError as e:
        # Handle errors during validation
        print(f"Validation failed: {e}")
        print(f"Output: {e.output.decode()}")
        print(f"Error: {e.stderr.decode()}")
        return False  # Return False if the validation fails
    
def run_mfa_align(audio_dir, language_model_dir, dictionary, output_dir):
    """
    This function runs the Montreal Forced Aligner on a given audio corpus and generates TextGrid files in the given output directory.
    
    Parameters:
    - audio_dir: The path to the folder of audio files and associated txt files to align. Make sure the audio and txt files have the same name before the extension. For example: audio_1.wav and audio_1.txt
    - language_model_dir: The acoustic model to use for alignment. For standard American English, use "english_us_arpa", else see https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
    - dictionary: The pronunciation dictionary for the langauge of your corpus. For standard American English, use "english_us_arpa", else see https://mfa-models.readthedocs.io/en/latest/dictionary/index.html
    - output_dir: The path to the folder where the output TextGrid files should be generated
    
    Output:
    - Returns nothing, but the output folder will be populated with TextGrid files
    """
    # Construct the MFA align command for the command line
    
    command = [
        "mfa", "align", audio_dir, language_model_dir, dictionary, output_dir
    ]
    
    try:
        # Run the align command and wait for it to complete
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Print the output and errors (if any)
        print(result.stdout.decode())
        print(result.stderr.decode())
        
    except subprocess.CalledProcessError as e:
        # Handle errors during alignment
        print(f"Error occurred: {e}")
        print(f"Output: {e.output.decode()}")
        print(f"Error: {e.stderr.decode()}")

def extract_intervals_from_textgrid(file_path):
    """
    This function extracts the onset and offset times of each phoneme and word from a TextGrid file, line-by-line.
    
    Parameters:
    - file_path: The path to the TextGrid file generated by running the Montreal Forced Aligner
    
    Output:
    - intervals: A list of lists containing information about whether the onsets and offsets are for a word or phoneme, the onset and offset times, and the text that is segmented in that row
    """
    intervals = []
    current_tier = None
    
    # Open the file given in the filepath, and go line-by-line to identify what information is contained within the line
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Identify the current tier (words or phones (phonemes))
            if 'name =' in line:
                if 'words' in line:
                    current_tier = 'words'
                elif 'phones' in line:
                    current_tier = 'phones'

            # Extract tmin, tmax, and text for each interval
            if line.startswith('intervals ['):
                tmin_line = next(f).strip()
                tmax_line = next(f).strip()
                text_line = next(f).strip()

                tmin = tmin_line.split('=')[1].strip()
                tmax = tmax_line.split('=')[1].strip()
                text = text_line.split('=')[1].strip().strip('"')  # Remove quotes

                intervals.append([current_tier, tmin, tmax, text])  # Add this information to a list

    return intervals

def write_intervals_to_csv(intervals, output_file):
    """
    This function writes the onset and offset intervals to a csv file.
    
    Parameters:
    - intervals: The output from extract_intervals_from_textgrid(file_path)
    - output_file: The filepath to the output file to write the csv to
    
    Output:
    - Populates the given output_file with interval information in a csv format
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['tier', 'tmin', 'tmax', 'text'])  # Write header
        writer.writerows(intervals)

def process_text_grids_from_folder(input_folder, output_folder=None):
    """
    This function takes a folder of TextGrid files generated from running the Montreal Forced Aligner pipeline and generates easily readable csv files for each TextGrid in the format: current_tier, tmin, tmax, text, where current_tier indicates the unit of language. This is the function that completes the second half of this MFA pipeline.
    
    Parameters:
    - input_folder: The folder containing all of the TextGrid files. This will likely be the same folder as output_dir in the mfa_pipeline
    - output_folder: The folder to write the csv files to. This is an optional argument, so by default, new csv files are added to the input folder
    
    Output:
    - Adds csv files to the input folder in the format: current_tier, tmin, tmax, text, where current_tier indicates the unit of language, tmin and tmax are the onset and offset times, and text is the text described within that interval
    """
    for filename in os.listdir(input_folder):
        if filename.endswith('.TextGrid'):
            input_file = os.path.join(input_folder, filename)
            intervals = extract_intervals_from_textgrid(input_file)

            # Create output file name (will be the TextGrid file name with "_extracted" appended to the end)
            if output_folder:
                output_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_extracted.csv")
            else:
                output_file = os.path.join(input_folder, f"{os.path.splitext(filename)[0]}_extracted.csv")
            write_intervals_to_csv(intervals, output_file)
            print(f'Data extracted to {output_file}')

def mfa_pipeline(audio_dir, language_model_dir, dictionary, output_dir, new_mfa_data=False):
    """
    This function runs the entire MFA pipeline: running the aligner and then processing the TextGrid files into csv files.
    
    Parameters:
    - audio_dir: The path to the folder of audio files and associated txt files to align. Make sure the audio and txt files have the same name before the extension. For example: audio_1.wav and audio_1.txt
    - language_model_dir: The acoustic model to use for alignment. For standard American English, use "english_us_arpa", else see https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
    - dictionary: The pronunciation dictionary for the langauge of your corpus. For standard American English, use "english_us_arpa", else see https://mfa-models.readthedocs.io/en/latest/dictionary/index.html
    - output_dir: The path to the folder where the output TextGrid files should be generated    
    
    Output:
    - Returns nothing, but the output folder will be populated with TextGrid files and csv files with the alignment information
    """
    if run_mfa_validate(audio_dir, language_model_dir, dictionary, new_mfa_data):
        print("Montreal Forced Aligner was validated! Starting alignment!")
        alignments_folder = os.path.join(output_dir, 'alignments')
        os.makedirs(alignments_folder, exist_ok=True)
        run_mfa_align(audio_dir, language_model_dir, dictionary, alignments_folder)
        print("Your audio and text files were aligned! Processing TextGrid files...")
        process_text_grids_from_folder(alignments_folder)
        
#####################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################
## This section has functions for retrieving pitch, intensity, and duration information from audio files, and directly follows from the MFA step of this pipeline.
## Before running any of these functions, make sure Praat is installed on your computer (https://www.fon.hum.uva.nl/praat/).
## For Windows: https://www.fon.hum.uva.nl/praat/download_win.html
## For Mac: https://www.fon.hum.uva.nl/praat/download_mac.html 
## For Linux: https://www.fon.hum.uva.nl/praat/download_linux.html
## All you need to run these is a corpus of audio files and their respective transcript files

def is_audio_file(file_path):
    """
    This function checks if the file is an audio file based on its extension.
    
    Parameters:
    - file_path: The path to the file to check
    
    Output:
    - True if the file is an audio file, False otherwise
    """
    # Known and acceptable audio file extensions
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
    
    # Check if the input file has a valid audio file extension
    return any(file_path.lower().endswith(ext) for ext in audio_extensions)

def run_praat_pitch(audio_file, output_folder, praat_location):
    """
    This function runs a Praat script to get the pitch contour calculated using the filtered autocorrelation method
    for a given .wav file and saves the result to a text file.
    The output file is named based on the input .wav file name, in the following format: {base_name}_praat_pitch_output.txt

    Parameters:
    - audio_file: The path to the input audio file
    - output_folder: The path to the output folder where the pitch contour data should be saved
    - praat_location: The path to the local Praat executable (On Windows, type "where praat" into the command prompt, and on MacOS or Linux, type "which praat" into the Terminal; for Windows usually: "C:/Program Files/Praat/praat.exe" or for Mac usually: "/Applications/Praat.app/Contents/MacOS/Praat")
    
    Output:
    - A .txt file with two columns: time,pitch
    """
    # First, check if the given file is a valid audio file.
    if not is_audio_file(audio_file):
        print(f"Skipping non-audio file: {audio_file}")
        return

    # Get the base name of the file, without extension, to make the output file name
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    pitch_folder = os.path.join(output_folder, 'praat_pitch')
    exists = os.path.exists(pitch_folder)
    if not exists:
        os.makedirs(pitch_folder, exist_ok=True)

    # Create the output filename based on the input filename, in the given output folder location
    output_filename = f"{pitch_folder}/{base_name}_praat_pitch_output.txt"

    # Path to Praat executable (adjust as needed)
    # On Windows, type "where praat" into the command prompt, and on MacOS or Linux, type "which praat" into the Terminal
    praat_executable = praat_location # Example for Windows: "C:/Program Files/Praat/praat.exe" or for Mac: "/Applications/Praat.app/Contents/MacOS/Praat"
    
    # Create a Praat script that uses the filtered autocorrelation method
    praat_script = f"""
    sound = Read from file: "{audio_file}"
    writeFileLine: "{output_filename}", "time,pitch"
    selectObject: 1
    To Pitch (filtered ac): 0, 50, 800, 15, "off", 0.03, 0.09, 0.5, 0.055, 0.35, 0.14
    no_of_frames = Get number of frames

    for frame from 1 to no_of_frames
        time = Get time from frame number: frame
        pitch = Get value in frame: frame, "Hertz"
        appendFileLine: "{output_filename}", "'time','pitch'"
    endfor
    """
    # Save the script to a temporary file
    script_file = "temp_praat_script.praat"
    with open(script_file, "w") as f:
        f.write(praat_script)

    # Run the Praat script in command line using subprocess
    subprocess.run([praat_executable, "--run", script_file])

    # Clean up the temporary script file by deleting it
    os.remove(script_file)

    print(f"Pitch extraction complete for {audio_file}. Output saved to {output_filename}")
    
    return output_filename

def run_praat_intensity(audio_file, output_folder, praat_location):
    """
    Runs Praat to calculate intensity for a given .wav file and saves the result to a text file.
    The output file is named based on the input .wav file name.

    Parameters:
    - audio_file: The path to the input .wav file
    - output_folder: The path to the output folder where the intensity contour data should be saved
    - praat_location: The path to the local Praat executable (On Windows, type "where praat" into the command prompt, and on MacOS or Linux, type "which praat" into the Terminal; for Windows usually: "C:/Program Files/Praat/praat.exe" or for Mac usually: "/Applications/Praat.app/Contents/MacOS/Praat")
    
    Output:
    - A .txt file with two columns: time,intensity
    """
    if not is_audio_file(audio_file):
        print(f"Skipping non-audio file: {audio_file}")
        return

    # Get the base name of the file (without extension)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    intensity_folder = os.path.join(output_folder, 'praat_intensity')
    exists = os.path.exists(intensity_folder)
    if not exists:
        os.makedirs(intensity_folder, exist_ok=True)
        
    # Create the output filename based on the input filename
    output_filename = f"{intensity_folder}/{base_name}_praat_intensity_output.txt"

    # Path to Praat executable (adjust as needed)
    # On Windows, type "where praat" into the command prompt, and on MacOS or Linux, type "which praat" into the Terminal
    praat_executable = praat_location  # Example for Windows: "C:/Program Files/Praat/praat.exe" or for Mac: "/Applications/Praat.app/Contents/MacOS/Praat"
    
    # Create a Praat script for retrieving intensity values
    praat_script = f"""
    sound = Read from file: "{audio_file}"
        writeFileLine: "{output_filename}", "time,intensity"
        selectObject: 1
        To Intensity: 50, 0, 1
        no_of_frames = Get number of frames

        for frame from 1 to no_of_frames
            time = Get time from frame number: frame
            intensity = Get value in frame: frame
            appendFileLine: "{output_filename}", "'time','intensity'"
        endfor
    """
    # Save the script to a temporary file
    script_file = "temp_praat_script.praat"
    with open(script_file, "w") as f:
        f.write(praat_script)

    # Run the Praat script using subprocess
    subprocess.run([praat_executable, "--run", script_file])

    # Clean up the temporary script file
    os.remove(script_file)

    print(f"Intensity extraction complete for {audio_file}. Output saved to {output_filename}")
    
    return output_filename

def read_praat_script_output(feature, feature_file_path):
    """
    This function reads the text file generated by run_praat_pitch or run_praat_intensity, processes the data for further processing by replacing "undefined" with NaN and dropping any NaN rows, and ensures that all values are numeric.
    
    Parameters:
    - feature: The feature whose file this is, either "pitch" or "intensity"
    - feature_file_path: The file path to the pitch or intensity listing generated by run_praat_pitch or run_praat_intensity
    
    Output:
    - A pandas DataFrame of all of the pitch or intensity data, with keys "time" and "pitch" or "intensity"
    """
    if feature=="pitch":
        # First, load the pitch data from the Praat CSV file, and format it so that the data is more usable (replace "undefined" with NaN, make sure all values are numeric)
        feature_data = pd.read_csv(feature_file_path)
        feature_data['pitch'] = feature_data['pitch'].replace('--undefined--', np.nan)
        feature_data['pitch'] = pd.to_numeric(feature_data['pitch'], errors='coerce')
        feature_data['time'] = pd.to_numeric(feature_data['time'], errors='coerce')
    
    elif feature=="intensity":
        # First, load the intensity data from the Praat CSV file, and format it so that the data is more usable (replace "undefined" with NaN, make sure all values are numeric)
        feature_data = pd.read_csv(feature_file_path)
        feature_data['intensity'] = feature_data['intensity'].replace('--undefined--', np.nan)
        feature_data['intensity'] = pd.to_numeric(feature_data['intensity'], errors='coerce')
        feature_data['time'] = pd.to_numeric(feature_data['time'], errors='coerce')
        
    return feature_data

def signal_mean(input_vector):
    """
    This function calculates and returns the mean of a given input vector.
    
    Parameters:
    - input_vector: The given one-dimensional vector to process

    Output:
    - Returns the mean value of the vector 
    """
    return np.mean(input_vector)

def signal_min(input_vector):
    """
    This function calculates and returns the minimum of a given input vector.
    
    Parameters:
    - input_vector: The given one-dimensional vector to process

    Output:
    - Returns the minimum value of the vector 
    """
    return np.min(input_vector)

def signal_max(input_vector):
    """
    This function calculates and returns the maximum of a given input vector.
    
    Parameters:
    - input_vector: The given one-dimensional vector to process

    Output:
    - Returns the maximum value of the vector 
    """
    return np.max(input_vector)

def signal_range(input_vector):
    """
    This function calculates and returns the range of a given input vector.
    
    Parameters:
    - input_vector: The given one-dimensional vector to process

    Output:
    - Returns the range of the vector 
    """
    return np.max(input_vector) - np.min(input_vector)

def signal_derivative(input_vector, times=None):
    """
    This function calculates and returns the derivative of a given input vector.
    
    Parameters:
    - input_vector: The given one-dimensional vector to process

    Output:
    - Returns the derivative of the vector 
    """
    ###TODO: remove print statements, add logging
    print("TYPE TYPE TYPE", type(input_vector))
    if (times is not None) and isinstance(input_vector, (list, np.ndarray, pd.Series, pd.DataFrame)) and len(input_vector) > 1 and len(times) > 1:
        return np.gradient(input_vector, times)
    elif isinstance(input_vector, (list, np.ndarray, pd.Series, pd.DataFrame)) and len(input_vector) > 1:
        return np.gradient(input_vector)
    else:
        return np.nan

def signal_variance(input_vector):
    """
    This function calculates and returns the variance of a given input vector.
    
    Parameters:
    - input_vector: The given one-dimensional vector to process

    Output:
    - Returns the variance of the vector 
    """
    return np.var(input_vector)

def get_pitch_values_per_word(word_row, pitch_data):
    """
    This function takes in a row from the csv generated after running the MFA pipeline and returns the pitch values associated with each word/phoneme.
    
    Parameters:
    - word_row: A row from a csv generated by running the MFA pipeline, which includes whether the row contains a word or phoneme, the onset and offset times, and the word/phoneme itself
    - pitch_data: A DataFrame containing pitch data

    Output:
    - Returns a DataFrame filtered by the onset and offset times of the given word
    """
    # Extract onset and offset times from the row
    onset_time = word_row['tmin']
    offset_time = word_row['tmax']
    
    # Filter pitch data for the times within the onset and offset range
    pitch_filtered = pitch_data[(pitch_data['time'] >= onset_time) & (pitch_data['time'] < offset_time)]
    if pitch_filtered.empty:
        return pd.DataFrame(columns=pitch_data.columns)
    # Return the filtered DataFrame containing both time and pitch
    return pitch_filtered

def get_intensity_values_per_word(word_row, intensity_data):
    """
    This function takes in a row from the csv generated after running the MFA pipeline and returns the intensity values associated with each word/phoneme.
    
    Parameters:
    - word_row: A row from a csv generated by running the MFA pipeline, which includes whether the row contains a word or phoneme, the onset and offset times, and the word/phoneme itself
    - intensity_data: A DataFrame containing intensity data

    Output:
    - Returns a DataFrame filtered by the onset and offset times of the given word
    """
    # Extract onset and offset times from the row
    onset_time = word_row['tmin']
    offset_time = word_row['tmax']
    
    # Filter intensity data for the times within the onset and offset range
    intensity_filtered = intensity_data[(intensity_data['time'] >= onset_time) & (intensity_data['time'] < offset_time)]
    
    if intensity_filtered.empty:
        return pd.DataFrame(columns=intensity_data.columns)
    # Return the filtered DataFrame containing both time and intensity
    return intensity_filtered

def get_average_duration_per_phoneme(alignment_file):
    ### TO DO: maybe you don't want to read this alignment file every time? Make a dataframe and send it over
    sound_onsets_offsets = pd.read_csv(alignment_file)
    duration_total = 0
    total_phonemes = 0
    for index, row in sound_onsets_offsets.iterrows():
        current_tier = row['tier']
        if current_tier == "phones":
            onset_time = row['tmin']
            offset_time = row['tmax']
            duration = offset_time - onset_time
            duration_total += duration
            total_phonemes += 1
    return duration_total / total_phonemes

def extract_all_statistical_features(pitch_data, intensity_data, audio_file, text):
    """
    This function calculates and generates a dictionary of all auditory features.
    
    Parameters:
    - pitch_file: The pitch listing file outputted from run_praat_pitch
    - intensity_file: The intensity listing file outputted from run_praat_intensity
    - audio_file: The original audio file

    Output:
    - Returns a dictionary of the auditory features from the statistics for pitch and intensity
    """
    # Remove rows with NaN values in the pitch data (if any)
    pitch_data_no_nan = pitch_data.dropna()
    # Remove rows with NaN values in the intensity data (if any)
    intensity_data_no_nan = intensity_data.dropna()
    
    if not pitch_data_no_nan.empty:
        # Calculate statistics for pitch data
        pitch_mean = signal_mean(pitch_data_no_nan["pitch"])
        pitch_min = signal_min(pitch_data_no_nan["pitch"])
        pitch_max = signal_max(pitch_data_no_nan["pitch"])
        pitch_range = signal_range(pitch_data_no_nan["pitch"])
        pitch_var = signal_variance(pitch_data_no_nan["pitch"])
        
        pitch_first_derivative = signal_derivative(pitch_data_no_nan["pitch"], pitch_data_no_nan["time"])
        pitch_first_deriv_min = signal_min(pitch_first_derivative)
        pitch_first_deriv_max = signal_max(pitch_first_derivative)
        pitch_first_deriv_mean = signal_mean(pitch_first_derivative)
        pitch_first_deriv_abs_mean = signal_mean(np.absolute(pitch_first_derivative))
        pitch_first_deriv_var = signal_variance(pitch_first_derivative)
        
        pitch_second_derivative = signal_derivative(pitch_first_derivative, pitch_data_no_nan["time"])
        pitch_second_deriv_min = signal_min(pitch_second_derivative)
        pitch_second_deriv_max = signal_max(pitch_second_derivative)
        pitch_second_deriv_var = signal_variance(pitch_second_derivative)
    
    if pitch_data_no_nan.empty:
        # If the pitch DataFrame is empty, then all of the statistics should be NaN.
        pitch_mean = np.nan
        pitch_min = np.nan
        pitch_max = np.nan
        pitch_range = np.nan
        pitch_var = np.nan
        
        pitch_first_derivative = np.nan
        pitch_first_deriv_min = np.nan
        pitch_first_deriv_max = np.nan
        pitch_first_deriv_mean = np.nan
        pitch_first_deriv_abs_mean = np.nan
        pitch_first_deriv_var = np.nan
        
        pitch_second_derivative = np.nan
        pitch_second_deriv_min = np.nan
        pitch_second_deriv_max = np.nan
        pitch_second_deriv_var = np.nan
        
    if not intensity_data_no_nan.empty:
        # Calculate statistics for intensity data
        intensity_mean = signal_mean(intensity_data_no_nan["intensity"])
        intensity_min = signal_min(intensity_data_no_nan["intensity"])
        intensity_max = signal_max(intensity_data_no_nan["intensity"])
        intensity_range = signal_range(intensity_data_no_nan["intensity"])
        intensity_var = signal_variance(intensity_data_no_nan["intensity"])
        
        intensity_first_derivative = signal_derivative(intensity_data_no_nan["intensity"], intensity_data_no_nan["time"])
        intensity_first_deriv_min = signal_min(intensity_first_derivative)
        intensity_first_deriv_max = signal_max(intensity_first_derivative)
        intensity_first_deriv_mean = signal_mean(intensity_first_derivative)
        intensity_first_deriv_abs_mean = signal_mean(np.absolute(intensity_first_derivative))
        intensity_first_deriv_var = signal_variance(intensity_first_derivative)
        
        intensity_second_derivative = signal_derivative(intensity_first_derivative, intensity_data_no_nan["intensity"])
        intensity_second_deriv_min = signal_min(intensity_second_derivative)
        intensity_second_deriv_max = signal_max(intensity_second_derivative)
        intensity_second_deriv_var = signal_variance(intensity_second_derivative)
    
    if intensity_data_no_nan.empty:
        intensity_mean = np.nan
        intensity_min = np.nan
        intensity_max = np.nan
        intensity_range = np.nan
        intensity_var = np.nan
        
        intensity_first_derivative = np.nan
        intensity_first_deriv_min = np.nan
        intensity_first_deriv_max = np.nan
        intensity_first_deriv_mean = np.nan
        intensity_first_deriv_abs_mean = np.nan
        intensity_first_deriv_var = np.nan
        
        intensity_second_derivative = np.nan
        intensity_second_deriv_min = np.nan
        intensity_second_deriv_max = np.nan
        intensity_second_deriv_var = np.nan
        
    # Naming the row after the audio file name, or a custom string
    if pd.notna(audio_file) and isinstance(audio_file, str) and os.path.isfile(audio_file):
        row_name = os.path.basename(audio_file)
    else:
        row_name = audio_file
        
    features = {
        "audio_file": row_name,
        "text": text,
        "pitch": [np.array(pitch_data["pitch"])],
        "pitch_time": [np.array(pitch_data["time"])],
        "pitch_mean": pitch_mean,
        "pitch_minimum": pitch_min,
        "pitch_maximum": pitch_max,
        "pitch_range": pitch_range,
        "pitch_variance": pitch_var,
        "pitch_first_derivative": [pitch_first_derivative],
        "pitch_first_derivative_time": [np.array(pitch_data_no_nan["time"])],
        "pitch_first_derivative_minimum": pitch_first_deriv_min,
        "pitch_first_derivative_maximum": pitch_first_deriv_max,
        "pitch_first_derivative_mean": pitch_first_deriv_mean,
        "pitch_absolute_value_of_first_derivative_mean": pitch_first_deriv_abs_mean,
        "pitch_first_derivative_variance": pitch_first_deriv_var,
        "pitch_second_derivative": [pitch_second_derivative],
        "pitch_second_derivative_minimum": pitch_second_deriv_min,
        "pitch_second_derivative_maximum": pitch_second_deriv_max,
        "pitch_second_derivative_variance": pitch_second_deriv_var,
        "intensity": [np.array(intensity_data["intensity"])],
        "intensity_time": [np.array(intensity_data["time"])],
        "intensity_mean": intensity_mean,
        "intensity_minimum": intensity_min,
        "intensity_maximum": intensity_max,
        "intensity_range": intensity_range,
        "intensity_variance": intensity_var,
        "intensity_first_derivative": [intensity_first_derivative],
        "intensity_first_derivative_time": [np.array(intensity_data_no_nan["time"])],
        "intensity_first_derivative_minimum": intensity_first_deriv_min,
        "intensity_first_derivative_maximum": intensity_first_deriv_max,
        "intensity_first_derivative_mean": intensity_first_deriv_mean,
        "intensity_absolute_value_of_first_derivative_mean": intensity_first_deriv_abs_mean,
        "intensity_first_derivative_variance": intensity_first_deriv_var,
        "intensity_second_derivative": [intensity_second_derivative],
        "intensity_second_derivative_minimum": intensity_second_deriv_min,
        "intensity_second_derivative_maximum": intensity_second_deriv_max,
        "intensity_second_derivative_variance": intensity_second_deriv_var
    }
    return features

def read_prominence_from_pkl_file(pkl_file):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

def extract_prominence_statistical_features(prominence_data):
    if prominence_data is None:
        prominence_data = np.nan
        prominence_mean = np.nan
        prominence_min = np.nan
        prominence_max = np.nan
        prominence_range = np.nan
        prominence_var = np.nan
        
        prominence_first_derivative = np.nan
        prominence_first_deriv_min = np.nan
        prominence_first_deriv_max = np.nan
        prominence_first_deriv_mean = np.nan
        prominence_first_deriv_abs_mean = np.nan
        prominence_first_deriv_var = np.nan
        
        prominence_second_derivative = np.nan
        prominence_second_deriv_min = np.nan
        prominence_second_deriv_max = np.nan
        prominence_second_deriv_var = np.nan
        
    else:
        prominence_mean = signal_mean(prominence_data)
        prominence_min = signal_min(prominence_data)
        prominence_max = signal_max(prominence_data)
        prominence_range = signal_range(prominence_data)
        prominence_var = signal_variance(prominence_data)
        
        prominence_first_derivative = signal_derivative(prominence_data)
        prominence_first_deriv_min = signal_min(prominence_first_derivative)
        prominence_first_deriv_max = signal_max(prominence_first_derivative)
        prominence_first_deriv_mean = signal_mean(prominence_first_derivative)
        prominence_first_deriv_abs_mean = signal_mean(np.absolute(prominence_first_derivative))
        prominence_first_deriv_var = signal_variance(prominence_first_derivative)
        
        prominence_second_derivative = signal_derivative(prominence_first_derivative)
        prominence_second_deriv_min = signal_min(prominence_second_derivative)
        prominence_second_deriv_max = signal_max(prominence_second_derivative)
        prominence_second_deriv_var = signal_variance(prominence_second_derivative)
    
    features = {
        "prominence": [np.array(prominence_data)],
        "prominence_mean": prominence_mean,
        "prominence_minimum": prominence_min,
        "prominence_maximum": prominence_max,
        "prominence_range": prominence_range,
        "prominence_variance": prominence_var,
        "prominence_first_derivative": [prominence_first_derivative],
        "prominence_first_derivative_minimum": prominence_first_deriv_min,
        "prominence_first_derivative_maximum": prominence_first_deriv_max,
        "prominence_first_derivative_mean": prominence_first_deriv_mean,
        "prominence_absolute_value_of_first_derivative_mean": prominence_first_deriv_abs_mean,
        "prominence_first_derivative_variance": prominence_first_deriv_var,
        "prominence_second_derivative": [prominence_second_derivative],
        "prominence_second_derivative_minimum": prominence_second_deriv_min,
        "prominence_second_derivative_maximum": prominence_second_deriv_max,
        "prominence_second_derivative_variance": prominence_second_deriv_var
    }
    
    return features

def extract_statistical_features_per_word(alignment_file, pitch_data, intensity_data, audio_file, output_folder, text, prominence_data=None):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    text = text.split()

    sound_onsets_offsets = pd.read_csv(alignment_file) 
    features_list = []
    for index, row in sound_onsets_offsets.iterrows():
        pitch_by_word = get_pitch_values_per_word(row, pitch_data)
        intensity_by_word = get_intensity_values_per_word(row, intensity_data)
        onset_time = row['tmin']
        offset_time = row['tmax']
        duration = offset_time - onset_time
        pause = 0
        current_tier = row['tier']
        
        if index < len(sound_onsets_offsets) - 2 and (pd.isna(sound_onsets_offsets.iloc[index + 1]['text']) or sound_onsets_offsets.iloc[index + 1]['text'] == '') and (sound_onsets_offsets.iloc[index]['tier'] == sound_onsets_offsets.iloc[index + 1]['tier']) and (sound_onsets_offsets.iloc[index]['tier'] == sound_onsets_offsets.iloc[index + 2]['tier']):
            pause += 1
        features = {
            "tier": current_tier,
            "onset_time": onset_time,
            "offset_time": offset_time,
            "duration": duration,
            "pause_after": pause
        }
        ###TODO: remove print statements, add logging
        print("ON WORD", row["text"])
        # print("PRINTING PITCH PER WORD", pitch_by_word)
        # print("###########################")
        # print("PRINTING INTENSITY PER WORD", intensity_by_word)
        features.update(extract_all_statistical_features(pitch_by_word, intensity_by_word, audio_file, row['text']))
        
        if prominence_data:
            for_logging = row["text"]
            if current_tier == "words" and row["text"] != '' and not pd.isna(row["text"]):
                print("TEXT IS!!!!:", text)
                print(len(text))
                row_text = row["text"].translate(str.maketrans('', '', string.punctuation)).lower()
                print("ROW TEXT IS!!!", row["text"])
                data_index = text.index(row_text)
                print(data_index)
                prominence_features = extract_prominence_statistical_features(prominence_data[data_index])
                print(f"Extracted prominence features for {for_logging}.")
            else:
                prominence_features = extract_prominence_statistical_features(None)
                print(f"Filled in with NaN values because not a word: {for_logging}")

            features.update(prominence_features)
    
        features_list.append(features)
    
    by_word_folder = os.path.join(output_folder, 'features_per_word_by_stimuli')
    os.makedirs(by_word_folder, exist_ok=True)
    output_features = pd.DataFrame(features_list)
    original_base = os.path.splitext(os.path.basename(audio_file))[0]
    output_filename = f"{by_word_folder}/prosody_features_by_word_for_{original_base}.csv"
    output_features.to_csv(output_filename, header=True, index=False)    
    
    return output_features

##TODO: update the summary here
def process_multiple_files_for_feature(feature, input_folder, output_folder, language_model_dir, dictionary, praat_location, new_mfa_data=False, prominence_file=None):
    """
    This function gets the pitch or intensity contour for all audio files in a given folder by calling run_praat_pitch or run_praat_intensity on each.
    
    Parameters:
    - feature: The feature to generate information about, either "pitch" or "intensity"
    - wav_folder: The folder containing audio files to process
    - output_folder: The path to the output folder where the pitch or intensity contour data should be saved
    - praat_location: The path to the local Praat executable (On Windows, type "where praat" into the command prompt, and on MacOS or Linux, type "which praat" into the Terminal; for Windows usually: "C:/Program Files/Praat/praat.exe" or for Mac usually: "/Applications/Praat.app/Contents/MacOS/Praat")

    Output:
    - .txt files created in the given output folder for each audio file with two columns for time, pitch or time, intensity
    """
    # First, run the MFA pipeline to get the onsets and offsets of each of the words in the audio files in the input_folder
    mfa_pipeline(input_folder, language_model_dir, dictionary, output_folder, new_mfa_data)
    # Use glob to find all audio files (with extensions like .wav, .mp3, etc.)
    audio_files = glob.glob(os.path.join(input_folder, "*"))
    features_list = [] # A list of features to create a DataFrame with pitch and intensity statistics for the "all" features.
    # Process each file
    for audio_file in audio_files:
        if not is_audio_file(audio_file):
            print(f"Skipping non-audio file: {audio_file}")
            continue
        # Depending on the feature input, run either the pitch or intensity function
        if feature=="pitch":
            run_praat_pitch(audio_file, output_folder, praat_location)
        elif feature=="intensity":
            run_praat_intensity(audio_file, output_folder, praat_location)
        elif feature=="all":
            pitch_output = run_praat_pitch(audio_file, output_folder, praat_location)
            intensity_output = run_praat_intensity(audio_file, output_folder, praat_location)
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            directory_path = os.path.dirname(audio_file)
            text_file_path = f"{base_name}.txt"
            with open(os.path.join(directory_path, text_file_path), 'r') as file:
                text_transcript = file.read().strip()
            if not pitch_output or not intensity_output:
                print(f"No pitch or intensity output generated for: {audio_file}")
                continue
            pitch_data = read_praat_script_output("pitch", pitch_output)
            intensity_data = read_praat_script_output("intensity", intensity_output)
            prominence_data = None
            if prominence_file:
                data = read_prominence_from_pkl_file(prominence_file)
                data_index = data["texts"].index(text_transcript)
                prominence_data = data["prominence"][data_index]
                flattened_prominence = [item for sublist in prominence_data for item in sublist]
                prominence_features = extract_prominence_statistical_features(flattened_prominence)
            alignments_folder = os.path.join(output_folder, 'alignments')
            alignment_file = os.path.join(alignments_folder, f"{base_name}_extracted.csv")
            mean_duration_of_phonemes = get_average_duration_per_phoneme(alignment_file)
            lower_level_features = extract_statistical_features_per_word(alignment_file, pitch_data, intensity_data, audio_file, output_folder, text_transcript, prominence_data)
            word_level_features = lower_level_features[lower_level_features['tier'] == 'words']
            mean_duration_of_words = word_level_features['duration'].mean()
            pause_rows = word_level_features[word_level_features['pause_after'] == 1]
            pause_durations = []
            for index, row in pause_rows.iterrows():
                next_row = word_level_features.iloc[index + 1]
                pause_durations.append(next_row['duration'])
            pause_durations = [np.array(pause_durations)]
            mean_pause_durations = np.nanmean(pause_durations)
            total_number_of_pauses_after_words = word_level_features['pause_after'].sum()
            
            word_features = {
                "duration_of_words_mean": mean_duration_of_words,
                "duration_of_phonemes_mean": mean_duration_of_phonemes,
                "total_number_of_pauses_after_words": total_number_of_pauses_after_words,
                "duration_of_pauses_after_words": pause_durations,
                "duration_of_pauses_after_words_mean": mean_pause_durations
            }
            
            features = extract_all_statistical_features(pitch_data, intensity_data, audio_file, text_transcript)
            if prominence_file:
                features.update(prominence_features)
            features.update(word_features)
            features_list.append(features)
    if feature=="all":
        output_features = pd.DataFrame(features_list)
        output_filename = f"{output_folder}/prosody_features_all.csv"
        output_features.to_csv(output_filename, header=True, index=False)
        plot_histograms_for_nonvector_prosody_features(output_filename, output_folder)
        plot_all_prosody_features(input_folder, output_folder)
       
def plot_praat_script_output(feature, feature_file_path, audio_file_path, text_file_path):
    """
    This function plots the pitch or intensity contour data from the .txt file generated by run_praat_pitch or run_praat_intensity as well as a spectrogram of the audio file.
    
    Parameters:
    - feature: The feature to generate a plot for, either "pitch" or "intensity"
    - feature_file_path: The file path to the pitch or intensity listing generated by run_praat_pitch or run_praat_intensity
    - audio_file_path: The file path to the original input audio file
    - text_file_path: The file path to the .txt file containing the transcript of the original input audio file
    
    Output:
    - Two subplots showing the pitch or intensity contour and the spectrogram of the audio file
    """
    if feature=="pitch":
        feature_data = read_praat_script_output("pitch", feature_file_path)
    
    elif feature=="intensity":
        feature_data = read_praat_script_output("intensity", feature_file_path)

    # Load the audio file using librosa
    y, sr = librosa.load(audio_file_path, sr=None)  # sr=None to preserve the original sample rate
    
    # Read the transcript of the audio file, to use as the title of the graph
    if text_file_path:
        with open(text_file_path, 'r') as file:
            title_text = file.read().strip()
    
    # Create a figure with two subplots: one for the pitch contour, one for the spectrogram
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # The title is the transcript of the text
    fig.suptitle(title_text)

    # Plot the pitch data in the first subplot
    if feature=="pitch":
        data_column = feature_data['pitch']
    elif feature=="intensity":
        data_column = feature_data["intensity"]
    axes[0].scatter(feature_data['time'], data_column, color='b', marker='o')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_ylim(50, 500)

    # Plot the spectrogram in the second subplot
    # Use librosa to compute the spectrogram
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)

    # Use librosa's display module to show the spectrogram
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Spectrogram')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_histograms_for_nonvector_prosody_features(prosody_features_file, output_folder):
    data = pd.read_csv(prosody_features_file)

    # Get the feature names (columns), excluding the first column (which is assumed to be an identifier)
    features = data.columns[1:]

    # Create a directory for the figures if it doesn't exist
    figures_folder = os.path.join(output_folder, 'histograms')
    os.makedirs(figures_folder, exist_ok=True)

    # Loop over each feature
    for feature in features:
        # Check if the feature is not a cell (i.e., it's not a vector-like feature)
        if not data[feature].apply(lambda x: isinstance(x, str)).any():  # If it's not a string (i.e., numeric)
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)
            # Plot a histogram for this feature
            plt.figure()
            data[feature].hist(bins=10)
            
            # Set the title and labels
            plt.title(feature.replace('_', ' '), fontsize=16)
            plt.xlabel(feature.replace('_', ' '), fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            
            # Save the plot as PDF and PNG
            plt.savefig(os.path.join(figures_folder, f'{feature.replace("_", " ")}.pdf'))
            plt.savefig(os.path.join(figures_folder, f'{feature.replace("_", " ")}.png'))
            
            plt.close()

def plot_all_prosody_features(input_folder, output_folder):
    audio_files = glob.glob(os.path.join(input_folder, "*"))
    pitch_folder = os.path.join(output_folder, 'praat_pitch')
    intensity_folder = os.path.join(output_folder, 'praat_intensity')
    alignments_folder = os.path.join(output_folder, 'alignments')
    all_prosody_features_csv_path = os.path.join(output_folder, 'prosody_features_all.csv')
    all_prosody_features_csv = pd.read_csv(all_prosody_features_csv_path)
    
    # Create a directory for the figures if it doesn't exist
    figures_folder = os.path.join(output_folder, 'feature_plots')
    os.makedirs(figures_folder, exist_ok=True)
    
    for audio_file in audio_files:
        if not is_audio_file(audio_file):
            print(f"Skipping non-audio file: {audio_file}")
            continue
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        pitch_feature_file = f"{pitch_folder}/{base_name}_praat_pitch_output.txt"
        intensity_feature_file = f"{intensity_folder}/{base_name}_praat_intensity_output.txt"
        transcript_alignment_file = f"{alignments_folder}/{base_name}_extracted.csv"
        filtered_prosody_feature = all_prosody_features_csv[all_prosody_features_csv['audio_file'] == os.path.basename(audio_file)]
        if 'prominence' in filtered_prosody_feature.columns:
            prominence_string = filtered_prosody_feature['prominence'].iloc[0]
            cleaned_string = prominence_string.replace('array(', '').replace(')', '')
            print(cleaned_string)
            prominence_value = np.array(ast.literal_eval(cleaned_string)) 
            prominence_value = prominence_value.ravel()
            print("TYPE TYPE TYPE", type(prominence_value))
            print(prominence_value)

        pitch_data = read_praat_script_output("pitch", pitch_feature_file)
        intensity_data = read_praat_script_output("intensity", intensity_feature_file)
        
        y, sr = librosa.load(audio_file, sr=None)  # sr=None to preserve the original sample rate
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        axes[0, 0].plot(np.linspace(0, len(y) / sr, len(y)), y)
        axes[0, 0].set_title('Audio Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        
        D = librosa.stft(y)  # Compute the Short-Time Fourier Transform
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to dB
        librosa.display.specshow(D_db, x_axis='time', y_axis='log', sr=sr, ax=axes[0, 1])  # Display spectrogram
        axes[0, 1].set_title('Spectrogram (STFT)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Frequency (Hz)')
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)  # Convert to dB scale for better visualization
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=axes[0, 2])  # Display Mel spectrogram
        axes[0, 2].set_title('Mel Spectrogram')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Mel Frequency')
        
        axes[1, 0].scatter(pitch_data['time'], pitch_data['pitch'], color='b', marker='o')
        axes[1, 0].plot(pitch_data['time'], pitch_data['pitch'], color='black', linewidth=1)
        axes[1, 0].set_title('Pitch')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        
        axes[1, 1].scatter(intensity_data['time'], intensity_data['intensity'], color='b', marker='o')
        axes[1, 1].plot(intensity_data['time'], intensity_data['intensity'], color='black', linewidth=1)
        axes[1, 1].set_title('Intensity')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Decibels (dB)')
        
        if 'prominence' in filtered_prosody_feature.columns:
            print(prominence_value.shape)
            prominence_time = np.linspace(0, len(y) / sr, len(prominence_value), endpoint=False)
            axes[1, 2].scatter(prominence_time, prominence_value, color='b', marker='o')
            axes[1, 2].plot(prominence_time, prominence_value, color='black', linewidth=1)
            axes[1, 2].set_title('Prominence')
            axes[1, 2].set_xlabel('Time (s)')
            axes[1, 2].set_ylabel('Prominence')
        
        if transcript_alignment_file is not None:
            word_df = pd.read_csv(transcript_alignment_file)
            
            # Filter rows where the type is 'word' (ignoring phonemes)
            word_df = word_df[word_df['tier'] == 'words']

            # Loop through each row in the filtered word DataFrame and add vertical lines and text
            for _, row in word_df.iterrows():
                tmin = row['tmin']
                tmax = row['tmax']
                text = row['text']
                
                # Add vertical lines at the word boundaries
                axes[0, 0].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
                axes[0, 0].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
                axes[0, 1].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
                axes[0, 1].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
                axes[0, 2].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
                axes[0, 2].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
                axes[1, 0].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
                axes[1, 0].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
                axes[1, 1].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
                axes[1, 1].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
                axes[1, 2].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
                axes[1, 2].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)

                # Add the word text between the vertical lines
                mid_point = (tmin + tmax) / 2  # Position text in the middle of the word duration
                axes[0, 0].text(mid_point, -0.25, text, color='black', ha='center', va='center', fontsize=8)

                # Print the word text and its start/end times to the console
                print(f"Word: '{text}', Start: {tmin}s, End: {tmax}s")
        
        if 'prominence' not in filtered_prosody_feature.columns:
            axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_folder, f'{base_name}_plots.pdf'))
        plt.savefig(os.path.join(figures_folder, f'{base_name}_plots.png'))

    return

def plot_all_prosody_features_for_file(input_folder, output_folder, audio_file):
    pitch_folder = os.path.join(output_folder, 'praat_pitch')
    intensity_folder = os.path.join(output_folder, 'praat_intensity')
    alignments_folder = os.path.join(output_folder, 'alignments')
    all_prosody_features_csv_path = os.path.join(output_folder, 'prosody_features_all.csv')
    all_prosody_features_csv = pd.read_csv(all_prosody_features_csv_path)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    pitch_feature_file = f"{pitch_folder}/{base_name}_praat_pitch_output.txt"
    intensity_feature_file = f"{intensity_folder}/{base_name}_praat_intensity_output.txt"
    transcript_alignment_file = f"{alignments_folder}/{base_name}_extracted.csv"
    filtered_prosody_feature = all_prosody_features_csv[all_prosody_features_csv['audio_file'] == os.path.basename(audio_file)]
    prominence_string = filtered_prosody_feature['prominence'].iloc[0]
    cleaned_string = prominence_string.replace('array(', '').replace(')', '')
    prominence_value = np.array(ast.literal_eval(cleaned_string)) 
    prominence_value = prominence_value.ravel()

    pitch_data = read_praat_script_output("pitch", pitch_feature_file)
    intensity_data = read_praat_script_output("intensity", intensity_feature_file)
    
    y, sr = librosa.load(audio_file, sr=None)  # sr=None to preserve the original sample rate
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(np.linspace(0, len(y) / sr, len(y)), y)
    axes[0, 0].set_title('Audio Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    
    D = librosa.stft(y)  # Compute the Short-Time Fourier Transform
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to dB
    librosa.display.specshow(D_db, x_axis='time', y_axis='log', sr=sr, ax=axes[0, 1])  # Display spectrogram
    axes[0, 1].set_title('Spectrogram (STFT)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)  # Convert to dB scale for better visualization
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=axes[0, 2])  # Display Mel spectrogram
    axes[0, 2].set_title('Mel Spectrogram')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Mel Frequency')
    
    axes[1, 0].scatter(pitch_data['time'], pitch_data['pitch'], color='b', marker='o')
    axes[1, 0].plot(pitch_data['time'], pitch_data['pitch'], color='black', linewidth=1)
    axes[1, 0].set_title('Pitch')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    
    axes[1, 1].scatter(intensity_data['time'], intensity_data['intensity'], color='b', marker='o')
    axes[1, 1].plot(intensity_data['time'], intensity_data['intensity'], color='black', linewidth=1)
    axes[1, 1].set_title('Intensity')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Decibels (dB)')
    
    prominence_time = np.linspace(0, len(y) / sr, len(prominence_value), endpoint=False)
    axes[1, 2].scatter(prominence_time, prominence_value, color='b', marker='o')
    axes[1, 2].plot(prominence_time, prominence_value, color='black', linewidth=1)
    axes[1, 2].set_title('Prominence')
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].set_ylabel('Prominence')
    
    if transcript_alignment_file is not None:
        word_df = pd.read_csv(transcript_alignment_file)
        
        # Filter rows where the type is 'word' (ignoring phonemes)
        word_df = word_df[word_df['tier'] == 'words']

        # Loop through each row in the filtered word DataFrame and add vertical lines and text
        for _, row in word_df.iterrows():
            tmin = row['tmin']
            tmax = row['tmax']
            text = row['text']
            
            # Add vertical lines at the word boundaries
            axes[0, 0].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
            axes[0, 0].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
            axes[0, 1].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
            axes[0, 1].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
            axes[0, 2].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
            axes[0, 2].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
            axes[1, 0].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
            axes[1, 0].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
            axes[1, 1].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
            axes[1, 1].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)
            axes[1, 2].axvline(x=tmin, color='y', linestyle='--', alpha=0.5)
            axes[1, 2].axvline(x=tmax, color='y', linestyle='--', alpha=0.5)

            # Add the word text between the vertical lines
            mid_point = (tmin + tmax) / 2  # Position text in the middle of the word duration
            axes[0, 0].text(mid_point, -0.25, text, color='black', ha='center', va='center', fontsize=8)

            # Print the word text and its start/end times to the console
            print(f"Word: '{text}', Start: {tmin}s, End: {tmax}s")
    
    annotations = []
    
    def on_click(event):
        # Loop over axes you want interactivity on
        interactive_axes = [axes[1, 0], axes[1, 1], axes[1, 2]]
        labels = ['Pitch', 'Intensity', 'Prominence']
        data_series = [
            (pitch_data['time'], pitch_data['pitch']),
            (intensity_data['time'], intensity_data['intensity']),
            (prominence_time, prominence_value)
        ]
        for ax, label, (x_data, y_data) in zip(interactive_axes, labels, data_series):
            if event.inaxes == ax:
                x_click, y_click = event.xdata, event.ydata
                x_array = np.array(x_data)
                y_array = np.array(y_data)

                valid = ~np.isnan(y_array)
                if not np.any(valid):
                    return

                x_valid = x_array[valid]
                y_valid = y_array[valid]

                distances = np.hypot(x_valid - x_click, y_valid - y_click)
                idx = distances.argmin()
                clicked_x = x_valid[idx]
                clicked_y = y_valid[idx]

                print(f"{label} → Time: {clicked_x:.2f}s, Value: {clicked_y:.2f}")
                
                # Remove old annotation
                for ann in annotations:
                    ann.remove()
                annotations.clear()

                # Add new annotation
                ann = ax.annotate(f"{clicked_y:.2f}", (clicked_x, clicked_y),
                                textcoords="offset points", xytext=(5, 5),
                                arrowprops=dict(arrowstyle="->"),
                                fontsize=8, color='red')
                annotations.append(ann)

                fig.canvas.draw()
                break

    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.tight_layout()
    plt.show()

    return

def is_leaf_directory(path):
    return all(not os.path.isdir(os.path.join(path, entry)) for entry in os.listdir(path))

def get_conversation_folders(top_level_folder):
    return [
        os.path.join(top_level_folder, directory)
        for directory in os.listdir(top_level_folder)
        if os.path.isdir(os.path.join(top_level_folder, directory))
    ]

def is_valid_speaker_folder(speaker_path, audio_exts={".wav", ".mp3"}, text_exts={".txt", ".TextGrid"}):
    if any(os.path.isdir(os.path.join(speaker_path, f)) for f in os.listdir(speaker_path)):
        return False  # Not a leaf folder

    files = os.listdir(speaker_path)
    has_audio = any(os.path.splitext(f)[1].lower() in audio_exts for f in files)
    has_text = any(os.path.splitext(f)[1].lower() in text_exts for f in files)
    return has_audio and has_text

#####################################################################################################################################################################################################################################
#####################################################################################################################################################################################################################################
## This section runs the functions in this script and contains example usage of them.
## After having installed Python, Montreal Forced Aligner, and Praat locally, following the instructions in the comments at the beginning of each section above, edit this section to include the desired functionality and the folder locations before typing "python prosody_features_anvitha.py" in the command line.

if __name__ == "__main__":
    input_folder = "/Users/anvithak/Desktop/CANDOR_SEG_FINISHED" # "/Users/anvithak/mfa_data/my_corpus"
    language_model_dir = "english_us_arpa"
    dictionary = "english_us_arpa"
    output_folder = "/Users/anvithak/Desktop/finished_outputs"
    praat_location = "/Applications/Praat.app/Contents/MacOS/Praat"
    new_mfa_data = True
    prominence_file = None
    
    mfa_pipeline(input_folder, language_model_dir, dictionary, output_folder, new_mfa_data)
    process_multiple_files_for_feature("all", input_folder, output_folder, language_model_dir, dictionary, praat_location, new_mfa_data)
    
    
    # # How to run the Praat pitch extraction:
    # # Process all audio files in a given folder after typing "python pitch_from_praat.py" in the command line.
    # # Replace the folder names with the path to the audio corpus and the path to the desired output folder.
    # input_folder = "/nese/mit/group/evlab/u/tamarr/CANDOR_SEG/00ae2f18-9599-4df6-8e3a-6936c86b97f0/" # "/Users/anvithak/mfa_data/my_corpus"
    # language_model_dir = "english_us_arpa"
    # dictionary = "english_us_arpa"
    # output_folder = "/Users/anvithak/Desktop/0020a0c5-1658-4747-99c1-2839e736b481/candor_outputs" # "/Users/anvithak/mfa_data/great_outputs_organized_numpy_change"
    # #output_folder = "/Users/anvithak/mfa_data/great_outputs_organized"
    # praat_location = "/home/anvithak/praat" # "/Applications/Praat.app/Contents/MacOS/Praat"
    # new_mfa_data = True
    # #prominence_file = "/Users/anvithak/Downloads/prosody/f0_dct_4.pkl"
    # mfa_pipeline(input_folder, language_model_dir, dictionary, output_folder, new_mfa_data)
    # #process_multiple_files_for_feature("all", input_folder, output_folder, language_model_dir, dictionary, praat_location, new_mfa_data)
    # plot_all_prosody_features(input_folder, output_folder)
    # plot_all_prosody_features_for_file(input_folder, output_folder, '/Users/anvithak/mfa_data/my_corpus/item_id=8_Studio_O_target_dur=2_epsilon_dur=0.03_actual_dur=1.9500_speed=0.893.wav')
    # #process_multiple_files_for_feature("all", "/Users/anvithak/Downloads/baseline200_auditory_stimset", "/Users/anvithak/Downloads/baseline200_auditory_stimset/features", )
    # # # How to run the graphing function (improvements to come)
    # # pitch_file_path = "/Users/anvithak/mfa_data/my_corpus/praat_pitch/common_voice_en_1_praat_pitch_output.txt"  # Path to Praat pitch .txt file
    # # audio_file_path = "/Users/anvithak/mfa_data/my_corpus/common_voice_en_1.mp3"  # Path to audio file
    # # text_file_path = "/Users/anvithak/mfa_data/my_corpus/common_voice_en_1.txt" # Path to transcript of audio file
    # # plot_praat_script_output("pitch", pitch_file_path, audio_file_path, text_file_path)
    
    # # data = read_prominence_from_pkl_file(prominence_file)
    # # data_index = data["texts"].index("It has a light tangy flavor.")
    # # prominence_data = data["prominence"][data_index]
    
    # # pitch_data = read_praat_script_output("pitch", '/Users/anvithak/mfa_data/greta_outputs/item_id=8_Studio_O_target_dur=2_epsilon_dur=0.03_actual_dur=1.9500_speed=0.893_praat_pitch_output.txt')
    # # intensity_data = read_praat_script_output("intensity", '/Users/anvithak/mfa_data/greta_outputs/item_id=8_Studio_O_target_dur=2_epsilon_dur=0.03_actual_dur=1.9500_speed=0.893_praat_intensity_output.txt')
    
    # # extract_statistical_features_per_word('/Users/anvithak/mfa_data/greta_outputs/item_id=8_Studio_O_target_dur=2_epsilon_dur=0.03_actual_dur=1.9500_speed=0.893_extracted.csv', pitch_data, intensity_data, '/Users/anvithak/mfa_data/my_corpus/item_id=8_Studio_O_target_dur=2_epsilon_dur=0.03_actual_dur=1.9500_speed=0.893.wav', output_folder, "It has a light tangy flavor.", prominence_data)
    
    # # # Example usage
    # # audio_dir = "/Users/anvithak/mfa_data/my_corpus"
    # # language_model_dir = "english_us_arpa"
    # # dictionary = "english_us_arpa"
    # # output_dir = "/Users/anvithak/mfa_data/test_aligned2"

    # # mfa_pipeline(audio_dir, language_model_dir, dictionary, output_dir)
    # # pitch_data = read_praat_script_output("pitch", "/Users/anvithak/mfa_data/0305test/common_voice_en_1_praat_pitch_output.txt")
    # # intensity_data = read_praat_script_output("intensity", "/Users/anvithak/mfa_data/0305test/common_voice_en_1_praat_intensity_output.txt")
    # # extract_statistical_features_per_word("/Users/anvithak/Downloads/common_voice_en_1_extracted.csv", pitch_data, intensity_data, "/Users/anvithak/mfa_data/my_corpus/common_voice_en_1.mp3", "/Users/anvithak/mfa_data")
    

    # The below code is for processing CANDOR on OpenMind, and is specific to the folder structure.
    # input_folder = "/Users/anvithak/Desktop/CANDOR_SEG_FINISHED" # "/Users/anvithak/mfa_data/my_corpus"
    # language_model_dir = "english_us_arpa"
    # dictionary = "english_us_arpa"
    # output_folder = "/Users/anvithak/Desktop/CANDOR_finished_outputs" # "/Users/anvithak/mfa_data/great_outputs_organized_numpy_change"
    # praat_location = "/Applications/Praat.app/Contents/MacOS/Praat"
    # new_mfa_data = True
    # prominence_file = None
    # def clean_filenames(root_folder):
    #     for dirpath, dirnames, filenames in os.walk(root_folder):
    #         for filename in filenames:
    #             # Remove 'audio' and 'transcript' from the filename
    #             new_name = filename.replace('_audio', '').replace('_transcript', '')
    #             if new_name != filename:
    #                 old_path = os.path.join(dirpath, filename)
    #                 new_path = os.path.join(dirpath, new_name)
    #                 print(f'Renaming: {old_path} -> {new_path}')
    #                 os.rename(old_path, new_path)

    # folder_path = '/home/anvithak/00ae2f18-9599-4df6-8e3a-6936c86b97f0'
    # clean_filenames(folder_path)
    
    # Parallel setup
    #my_task_id = int(sys.argv[1])
    #num_tasks = int(sys.argv[2])
    
    #all_conversations = get_conversation_folders(input_folder)
    #my_conversations = all_conversations[my_task_id::num_tasks]
    # for conv_path in all_conversations:
    #     for speaker_name in os.listdir(conv_path):
    #         speaker_path = os.path.join(conv_path, speaker_name)
    #         if not os.path.isdir(speaker_path):
    #             continue
    #         if not is_valid_speaker_folder(speaker_path):
    #             continue

    #         output_folder = os.path.join(conv_path, f"{speaker_name}_features")
    #         os.makedirs(output_folder, exist_ok=True)

    #         print(f"[Task {my_task_id}] Processing {speaker_path}")
    #         process_multiple_files_for_feature(
    #             "all",
    #             speaker_path,
    #             output_folder,
    #             language_model_dir,
    #             dictionary,
    #             praat_location,
    #             new_mfa_data=new_mfa_data,
    #             prominence_file=prominence_file
    #         )
    # log_file_path = os.path.join(input_folder, "feature_extraction.log")

    # for conversation_name in os.listdir(input_folder):
    #     conversation_path = os.path.join(input_folder, conversation_name)
    #     if not os.path.isdir(conversation_path):
    #         continue
    
    #     for speaker_name in os.listdir(conversation_path):
    #             speaker_path = os.path.join(conversation_path, speaker_name)
    #             if not os.path.isdir(speaker_path):
    #                 continue
    #             if not is_valid_speaker_folder(speaker_path):
    #                 continue

    #             output_folder = os.path.join(conversation_path, f"{speaker_name}_features")
    #             os.makedirs(output_folder, exist_ok=True)

    #             print(f"[Processing {speaker_path}")
    #             start_time = time.time()
    #             process_multiple_files_for_feature(
    #                 "all",
    #                 speaker_path,
    #                 output_folder,
    #                 language_model_dir,
    #                 dictionary,
    #                 praat_location,
    #                 new_mfa_data=new_mfa_data,
    #                 prominence_file=prominence_file
    #             )
    #             end_time = time.time()
    #             duration = end_time - start_time
                
    #             with open(log_file_path, "a") as log_file:
    #                 log_file.write(f"Processed {speaker_path} in {duration:.2f} seconds\n")