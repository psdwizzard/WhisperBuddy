# Whisper Buddy

A simple desktop application for real-time speech-to-text transcription using OpenAI's Whisper models.

![Whisper Buddy Screenshot](https://raw.githubusercontent.com/psdwizzard/WhisperBuddy/refs/heads/main/Screenshot.png)

## Features

- **Push-to-Talk Transcription:**  
  Hold down the "Listen" button to capture and transcribe your voice in real time.

- **Live Transcription Mode:**  
  Continuous, real-time transcription with timestamps.

- **Model Selection:**  
  Choose from all available Whisper models (Tiny, Base, Small, Medium, Large, Turbo) for the best balance of speed and accuracy.

- **Dark Mode Interface:**  
  Clean, modern dark theme for comfortable use in any environment.

- **Performance Options:**  
  - Keep Model Loaded: Toggle to keep the Whisper model in memory for faster transcriptions.
  - Use CUDA: Enable GPU acceleration (if a CUDA-compatible NVIDIA GPU is available).

- **Persistent Settings:**  
  Your preferences (model selection, theme, etc.) are saved automatically and restored on next launch.

- **Copy to Clipboard:**  
  Easily copy your transcribed text for use in other applications.

## Installation

### Clone the Repository
```bash
git clone https://github.com/psdwizzard/MeetingBuddy.git
cd MeetingBuddy
```

### Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Install Required Packages
```bash
pip install customtkinter
pip install pillow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U openai-whisper
pip install sounddevice
pip install numpy
pip install pyperclip
pip install python-docx
pip install fpdf
pip install requests
```

### Run the Application
```bash
start.bat
```

## Usage

- **Push-to-Talk:**  
  Hold down the "Listen" button while speaking; release to transcribe.

- **Live Transcription:**  
  Click "Live Transcribe" for continuous, real-time speech-to-text.

- **Settings:**  
  - Choose your preferred Whisper model.
  - Toggle dark mode and accent color.
  - Enable or disable GPU acceleration.

- **Copy:**  
  Use the "Copy" button to copy your transcribed text to the clipboard.

## System Requirements

- **Operating System:** Windows
- **Python Version:** 3.8 or higher
- **GPU (Optional):** NVIDIA GPU with CUDA support for acceleration 
