@echo off
cd /d "Z:\drive\Tools\audio_stem_separator"

:: --- SETUP VENV ---
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

:: --- CHECK INSTALLATION FLAG ---
:: If the flag exists, jump straight to launching the app
if exist ".venv\installed.flag" goto :launch

echo ========================================
echo Installing dependencies...
echo ========================================
python -m pip install --upgrade pip setuptools wheel

echo.
echo [1/13] Core dependencies...
pip install "numpy==1.23.5"
pip install "pydantic==1.10.13"

echo.
echo [2/13] UI and utilities...
pip install PySide6
pip install scipy
pip install soundfile
pip install "librosa==0.9.2"
pip install tqdm

echo.
echo [3/13] Huggingface ecosystem...
pip install "huggingface_hub==0.21.0"
pip install "transformers==4.30.2"
pip install "accelerate==0.25.0"
pip install "diffusers==0.25.0"

echo.
echo [4/13] PyTorch CUDA...
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [5/13] ONNX first to set correct protobuf...
pip install onnx
pip install onnxruntime-gpu

echo.
echo [6/13] Audio separation...
pip install "demucs==4.0.1"
pip install "audio-separator==0.17.5"

echo.
echo [7/13] AudioSR...
pip install audiosr
pip install progressbar2
pip install chardet pandas unidecode timm torchlibrosa
pip install einops
pip install "pytorch_lightning==2.1.0"
pip install omegaconf
pip install ftfy

echo.
echo [8/13] Audio codecs...
pip install "descript-audio-codec==1.0.0"
pip install encodec

echo.
echo [9/13] Hydra for inpainting config...
pip install hydra-core

echo.
echo [10/13] Cloning audio-inpainting-diffusion repo...
if not exist "audio-inpainting-diffusion" (
    git clone https://github.com/eloimoliner/audio-inpainting-diffusion
) else (
    echo    Repo already exists, pulling latest...
    cd audio-inpainting-diffusion
    git pull
    cd ..
)

echo.
echo [11/13] Downloading pretrained inpainting models...
if not exist "models\inpainting" mkdir "models\inpainting"

python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Eloimoliner/audio-inpainting-diffusion', 'musicnet_44k_4s-560000.pt', local_dir='models/inpainting')"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Eloimoliner/audio-inpainting-diffusion', 'maestro_22k_8s-750000.pt', local_dir='models/inpainting')"

echo.
echo [12/13] FINAL FIX - Reinstall ONNX, protobuf, and PyTorch CUDA...
pip uninstall protobuf onnx -y
pip install onnx
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [13/13] ADDING MUSICGEN (Manual Steps)...

REM 1. Fix AV (FFmpeg) - Manual Download
if not exist "av-16.0.1-cp310-cp310-win_amd64.whl" (
    echo Downloading av-16.0.1...
    curl -L -o av-16.0.1-cp310-cp310-win_amd64.whl "https://files.pythonhosted.org/packages/08/9d/281699f074d271954314c1d76378e9b0687158752156829141029c733330/av-16.0.1-cp310-cp310-win_amd64.whl"
)
pip install av-16.0.1-cp310-cp310-win_amd64.whl

REM 2. Install Xformers (Required for MusicGen)
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121

REM 3. Upgrade Transformers (4.30.2 is too old for MusicGen, 4.35.2 is safe middle ground)
pip install "transformers==4.35.2"

REM 4. Install Audiocraft Dependencies
pip install flashy hydra-colorlog num2words spacy "torchtext==0.16.0" sentencepiece julius

REM 5. Install Audiocraft (No Deps to skip AV check)
pip install audiocraft --no-deps

REM 6. Fix Librosa (Audio-Separator needs >0.10)
pip install "librosa==0.10.1"

REM 7. Final Protobuf Fix (for ONNX/AudioTools compatibility)
pip install "protobuf==3.20.3"

echo.
echo Verifying installations...
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "from google.protobuf.internal import builder; print('Protobuf: OK')"
python -c "import onnx; print('ONNX: OK')"
python -c "from audio_separator.separator import Separator; print('Audio-Separator: OK')"
python -c "import audiosr; print('AudioSR: OK')"
python -c "import hydra; print('Hydra: OK')"
python -c "import audiocraft; print('Audiocraft: OK')"

echo.
type nul > ".venv\installed.flag"
echo ========================================
echo Installation complete!
echo ========================================

:launch
echo.
echo Starting application...
python __init__.py
pause