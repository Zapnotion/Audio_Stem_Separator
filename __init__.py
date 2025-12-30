import sys
import os
import gc
import numpy as np
import math
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import torch
import torchaudio
import tempfile
import shutil
import soundfile as sf
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QLineEdit, QProgressBar, QMessageBox, QComboBox,
    QHBoxLayout, QGroupBox, QCheckBox, QFrame, QSpinBox, QRadioButton,
    QButtonGroup, QSlider, QScrollArea, QSizePolicy, QTextEdit  
)
from PySide6.QtCore import QThread, Signal, Slot, Qt
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import save_audio
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPAINTING_REPO_PATH = os.path.join(SCRIPT_DIR, "audio-inpainting-diffusion")
INPAINTING_MODELS_DIR = os.path.join(SCRIPT_DIR, "models", "inpainting")







DEMUCS_QUALITY_PRESETS = {
    "preview": {
        "name": "⚡ Preview",
        "shifts": 1,
        "overlap": 0.10,
        "description": "Quick preview - fastest, lower quality",
        "time_estimate": "~30 seconds per song"
    },
    "fast": {
        "name": "🚀 Fast",
        "shifts": 2,
        "overlap": 0.25,
        "description": "Good quality, fast processing",
        "time_estimate": "~1-2 minutes per song"
    },
    "balanced": {
        "name": "⚖️ Balanced",
        "shifts": 3,
        "overlap": 0.35,
        "description": "Better quality, moderate speed",
        "time_estimate": "~2-4 minutes per song"
    },
    "high_quality": {
        "name": "✨ High Quality",
        "shifts": 5,
        "overlap": 0.50,
        "description": "Excellent quality, slower",
        "time_estimate": "~5-8 minutes per song"
    },
    "maximum": {
        "name": "🏆 Maximum",
        "shifts": 10,
        "overlap": 0.75,
        "description": "Absolute best quality, very slow",
        "time_estimate": "~15-25 minutes per song"
    },
}


FREQUENCY_PRESETS_GENRE = {
    "pop_rock": {
        "name": "🎸 Pop / Rock",
        "low_freq": 250,
        "high_freq": 6000,
        "description": "Standard pop/rock mix - guitars, synths in mid, cymbals in high",
        "details": "Low: Bass synths, low guitars | Mid: Main instruments | High: Presence, air"
    },
    "electronic_edm": {
        "name": "🎹 Electronic / EDM",
        "low_freq": 200,
        "high_freq": 8000,
        "description": "Electronic music - extended bass, bright highs",
        "details": "Low: Sub bass, 808s | Mid: Synths, leads | High: Hi-hats, sparkle"
    },
    "hip_hop_trap": {
        "name": "🎤 Hip-Hop / Trap",
        "low_freq": 180,
        "high_freq": 7000,
        "description": "Hip-hop focused - deep bass, crisp highs",
        "details": "Low: 808s, sub bass | Mid: Samples, keys | High: Hi-hats, ad-libs"
    },
    "classical_orchestral": {
        "name": "🎻 Classical / Orchestral",
        "low_freq": 300,
        "high_freq": 5000,
        "description": "Orchestral music - cello/bass separation, string detail",
        "details": "Low: Double bass, cello | Mid: Violins, winds, brass | High: Overtones, room"
    },
    "jazz": {
        "name": "🎷 Jazz",
        "low_freq": 280,
        "high_freq": 5500,
        "description": "Jazz ensemble - upright bass, piano, horns",
        "details": "Low: Upright bass | Mid: Piano, horns, guitar | High: Cymbal wash, breath"
    },
    "metal_heavy": {
        "name": "🤘 Metal / Heavy Rock",
        "low_freq": 220,
        "high_freq": 6500,
        "description": "Heavy music - tight low end, aggressive mids",
        "details": "Low: Bass guitar, kick weight | Mid: Guitars, vocals | High: Cymbals, pick attack"
    },
    "acoustic_folk": {
        "name": "🪕 Acoustic / Folk",
        "low_freq": 300,
        "high_freq": 5000,
        "description": "Acoustic instruments - natural separation",
        "details": "Low: Acoustic bass, low guitar | Mid: Guitar body, vocals | High: String detail, air"
    },
    "rnb_soul": {
        "name": "🎙️ R&B / Soul",
        "low_freq": 200,
        "high_freq": 7000,
        "description": "R&B/Soul - warm bass, silky highs",
        "details": "Low: Bass, keys | Mid: Rhodes, guitars, vocals | High: Breathiness, sheen"
    },
}


FREQUENCY_PRESETS_TECHNICAL = {
    "bass_focus": {
        "name": "🔊 Bass Focus (Sub-heavy)",
        "low_freq": 150,
        "high_freq": 6000,
        "description": "Extended low end - captures more sub bass content",
        "details": "Low: Everything below 150Hz (sub bass, kick fundamentals)"
    },
    "mid_focus": {
        "name": "🎯 Mid Focus (Vocal range)",
        "low_freq": 300,
        "high_freq": 4000,
        "description": "Focused mid range - isolates vocal/instrument fundamentals",
        "details": "Mid: 300Hz-4kHz covers most melodic content"
    },
    "bright_focus": {
        "name": "✨ Bright Focus (Presence)",
        "low_freq": 250,
        "high_freq": 4500,
        "description": "More content in high band - detailed highs",
        "details": "High: Extended range captures more presence and detail"
    },
    "wide_low": {
        "name": "📊 Wide Low (200Hz)",
        "low_freq": 200,
        "high_freq": 6000,
        "description": "Standard low split at 200Hz",
        "details": "Common mixing crossover point"
    },
    "narrow_bands": {
        "name": "📏 Narrow Bands",
        "low_freq": 350,
        "high_freq": 3500,
        "description": "Narrow mid focus - more content in low/high",
        "details": "Low: Extended | Mid: Focused | High: Extended"
    },
    "wide_bands": {
        "name": "📐 Wide Mid Band",
        "low_freq": 150,
        "high_freq": 8000,
        "description": "Very wide mid - minimal low/high content",
        "details": "Most content stays in mid band"
    },
}


FREQUENCY_PRESETS_INSTRUMENT = {
    "piano_keys": {
        "name": "🎹 Piano / Keys Focus",
        "low_freq": 260,  # Below middle C
        "high_freq": 5200,
        "description": "Optimized for piano and keyboard instruments",
        "details": "Low: Bass notes | Mid: Main playing range | High: Brilliance"
    },
    "guitar_focus": {
        "name": "🎸 Guitar Focus",
        "low_freq": 250,
        "high_freq": 5000,
        "description": "Optimized for acoustic and electric guitar",
        "details": "Low: Low E string area | Mid: Body and tone | High: Pick attack, fret noise"
    },
    "strings_orchestral": {
        "name": "🎻 Strings Focus",
        "low_freq": 200,
        "high_freq": 4500,
        "description": "Optimized for violin, viola, cello",
        "details": "Low: Cello, viola low | Mid: Main string tone | High: Bow detail, harmonics"
    },
    "synth_electronic": {
        "name": "🎛️ Synth Focus",
        "low_freq": 180,
        "high_freq": 8000,
        "description": "Optimized for synthesizers and electronic sounds",
        "details": "Low: Sub bass, bass synths | Mid: Leads, pads | High: Shimmer, effects"
    },
    "brass_winds": {
        "name": "🎺 Brass / Winds Focus",
        "low_freq": 280,
        "high_freq": 5000,
        "description": "Optimized for brass and woodwind instruments",
        "details": "Low: Tuba, trombone low | Mid: Main horn range | High: Breath, brightness"
    },
}














def check_available_models():
    """Check which models are available."""
    available = {
        "demucs": False,
        "audio_separator": False,
        "spectral": True,
        "enhanced_spectral": True,
        "diffusion_inpaint": False,
        "audiosr": False,
        "musicgen": False,
    }
    
    try:
        from demucs.pretrained import get_model
        available["demucs"] = True
        print("✅ Demucs available")
    except ImportError as e:
        print(f"❌ Demucs not available: {e}")
    
    try:
        from audio_separator.separator import Separator
        available["audio_separator"] = True
        print("✅ Audio-Separator (UVR5/MDX) available")
    except ImportError as e:
        print(f"❌ Audio-Separator not available: {e}")
    
    print("✅ Spectral Restoration available (basic)")
    print("✅ Enhanced Spectral Restoration available (advanced)")
    
    # Check diffusion inpainting
    inpaint_model = os.path.join(INPAINTING_MODELS_DIR, "musicnet_44k_4s-560000.pt")
    
    if os.path.exists(INPAINTING_REPO_PATH) and os.path.exists(inpaint_model):
        available["diffusion_inpaint"] = True
        print("✅ Diffusion Inpainting available")
    else:
        if not os.path.exists(INPAINTING_REPO_PATH):
            print(f"❌ Diffusion Inpainting: repo not found at {INPAINTING_REPO_PATH}")
        elif not os.path.exists(inpaint_model):
            print(f"❌ Diffusion Inpainting: model not found at {inpaint_model}")
    
    try:
        import audiosr
        available["audiosr"] = True
        print("✅ AudioSR available")
    except ImportError as e:
        print(f"❌ AudioSR not available: {e}")
    
    try:
        import audiocraft
        available["musicgen"] = True
        print("✅ MusicGen (Audiocraft) available")
    except ImportError as e:
        print(f"❌ MusicGen not available: {e}")
    except Exception as e:
        # Audiocraft sometimes throws warnings on import, but still works
        print(f"⚠️ MusicGen loaded with warnings: {e}")
        available["musicgen"] = True 

    return available


AVAILABLE_MODELS = check_available_models()



class DropZone(QFrame):
    file_dropped = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(2)
        self.setStyleSheet("""
            DropZone {
                background-color: #f0f0f0;
                border: 2px dashed #999;
                border-radius: 8px;
                padding: 20px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self.drop_label = QLabel("🎵 Drag & Drop Audio File Here")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("color: #666; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.drop_label)
        self.setMinimumHeight(100)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if any(url.toLocalFile().lower().endswith(ext) 
                       for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac']):
                    event.acceptProposedAction()
                    self.setStyleSheet("DropZone { background-color: #d4edda; border: 2px dashed #28a745; border-radius: 8px; }")
                    return
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("DropZone { background-color: #f0f0f0; border: 2px dashed #999; border-radius: 8px; }")
    
    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if any(path.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac']):
                self.file_dropped.emit(path)
                break
        self.dragLeaveEvent(None)




class UVRSeparator:
    """High-quality stem separation using UVR5/MDX models."""
    
    INSTRUMENTAL_MODELS = [
        ("UVR-MDX-NET-Inst_HQ_3.onnx", "MDX-Net Inst HQ3 (Best)"),
        ("UVR_MDXNET_KARA_2.onnx", "MDX-Net Karaoke 2"),
        ("Kim_Vocal_2.onnx", "Kim Vocal 2 (Good)"),
        ("UVR-MDX-NET-Inst_Main.onnx", "MDX-Net Inst Main"),
        ("UVR_MDXNET_Main.onnx", "MDX-Net Main"),
    ]
    
    VOCAL_MODELS = [
        ("UVR-MDX-NET-Voc_FT.onnx", "MDX-Net Vocal FT (Best)"),
        ("Kim_Vocal_2.onnx", "Kim Vocal 2"),
        ("UVR_MDXNET_Main.onnx", "MDX-Net Main"),
    ]
    
    def __init__(self, output_dir, device=None):
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.separator = None
        self.temp_dir = None
        
    def _init_separator(self, model_name, progress_callback=None):
        """Initialize the separator with specified model."""
        from audio_separator.separator import Separator
        
        if progress_callback:
            progress_callback(f"Loading model: {model_name}...")
        
        self.temp_dir = tempfile.mkdtemp(prefix="uvr_sep_")
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        self.separator = Separator(
            output_dir=self.temp_dir,
            output_format="wav",
        )
        
        self.separator.load_model(model_name)
        
        if progress_callback:
            progress_callback(f"✅ Model loaded: {model_name}")
    
    def separate_single(self, audio_path, model_name, output_stem="instrumental",
                        progress_callback=None):
        """Separate using a single model."""
        self._init_separator(model_name, progress_callback)
        
        if progress_callback:
            progress_callback(f"Separating with {model_name}...")
        
        try:
            output_files = self.separator.separate(audio_path)

            logger.info(f"Output files from separator: {output_files}")
            logger.info(f"Temp dir: {self.temp_dir}")
            logger.info(f"Temp dir contents: {os.listdir(self.temp_dir) if os.path.exists(self.temp_dir) else 'NOT FOUND'}")
            
            vocals_path = None
            instrumental_path = None
            
            for f in output_files:
                if not os.path.isabs(f):
                    temp_path = os.path.join(self.temp_dir, f)
                    if os.path.exists(temp_path):
                        f = temp_path
                    elif os.path.exists(f):
                        f = os.path.abspath(f)
                    else:
                        input_dir = os.path.dirname(audio_path)
                        input_dir_path = os.path.join(input_dir, f)
                        if os.path.exists(input_dir_path):
                            f = input_dir_path
                
                f_lower = os.path.basename(f).lower()
                
                if "(vocals)" in f_lower or "vocals" in f_lower:
                    vocals_path = f
                    logger.info(f"Found vocals: {f}")
                elif "(instrumental)" in f_lower or "instrumental" in f_lower or "(other)" in f_lower:
                    instrumental_path = f
                    logger.info(f"Found instrumental: {f}")
            
            if len(output_files) >= 2:
                for f in output_files:
                    if not os.path.isabs(f):
                        for check_dir in [self.temp_dir, os.getcwd(), os.path.dirname(audio_path)]:
                            check_path = os.path.join(check_dir, f)
                            if os.path.exists(check_path):
                                f = check_path
                                break
                    
                    if os.path.exists(f):
                        f_lower = os.path.basename(f).lower()
                        if vocals_path is None and "(vocals)" in f_lower:
                            vocals_path = f
                        elif instrumental_path is None and "(instrumental)" not in f_lower and "(vocals)" not in f_lower:
                            if instrumental_path is None:
                                instrumental_path = f
            
            if progress_callback:
                progress_callback("✅ Separation complete")
            
            logger.info(f"Final vocals_path: {vocals_path}")
            logger.info(f"Final instrumental_path: {instrumental_path}")
            
            return vocals_path, instrumental_path
            
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            raise

    def separate_ensemble(self, audio_path, models=None, progress_callback=None):
        """Separate using multiple models and ensemble the results."""
        if models is None:
            models = ["UVR-MDX-NET-Inst_HQ_3.onnx", "Kim_Vocal_2.onnx"]
        
        if progress_callback:
            progress_callback(f"Ensemble separation with {len(models)} models...")
        
        vocal_results = []
        instrumental_results = []
        sample_rate = None
        
        for i, model_name in enumerate(models):
            if progress_callback:
                progress_callback(f"Model {i+1}/{len(models)}: {model_name}")
            
            try:
                vocals_path, inst_path = self.separate_single(
                    audio_path, model_name, progress_callback=None
                )
                
                if vocals_path and os.path.exists(vocals_path):
                    audio, sr = sf.read(vocals_path)
                    vocal_results.append(audio)
                    sample_rate = sr
                
                if inst_path and os.path.exists(inst_path):
                    audio, sr = sf.read(inst_path)
                    instrumental_results.append(audio)
                    sample_rate = sr
                    
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        if not instrumental_results and not vocal_results:
            raise Exception("All ensemble models failed!")
        
        final_vocals = None
        final_instrumental = None
        
        if vocal_results:
            if progress_callback:
                progress_callback("Computing vocal ensemble...")
            min_len = min(len(r) for r in vocal_results)
            vocal_results = [r[:min_len] if len(r.shape) == 1 else r[:min_len, :] for r in vocal_results]
            final_vocals = np.median(np.stack(vocal_results), axis=0)
        
        if instrumental_results:
            if progress_callback:
                progress_callback("Computing instrumental ensemble...")
            min_len = min(len(r) for r in instrumental_results)
            instrumental_results = [r[:min_len] if len(r.shape) == 1 else r[:min_len, :] for r in instrumental_results]
            final_instrumental = np.median(np.stack(instrumental_results), axis=0)
        
        vocals_out = None
        instrumental_out = None
        
        if final_vocals is not None:
            vocals_out = os.path.join(self.output_dir, "vocals.wav")
            sf.write(vocals_out, final_vocals, sample_rate)
        
        if final_instrumental is not None:
            instrumental_out = os.path.join(self.output_dir, "instrumental.wav")
            sf.write(instrumental_out, final_instrumental, sample_rate)
        
        if progress_callback:
            progress_callback("✅ Ensemble separation complete")
        
        return vocals_out, instrumental_out
    
    def cleanup(self):
        """Clean up temporary files."""
        if hasattr(self, 'original_cwd') and self.original_cwd:
            os.chdir(self.original_cwd)
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")







class MusicGenRegenerator:
    """
    Regenerates audio using MusicGen.
    Supports 'melody' model (structure preserving) and 'small/medium/large' (style preserving).
    """
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_name = "" # Track which model is loaded
        self.sample_rate = 32000 

    def load_model(self, model_size='melody'):
        from audiocraft.models import MusicGen
        if self.model is None or self.model_name != model_size:
            logger.info(f"Loading MusicGen: facebook/musicgen-{model_size}")
            self.model = MusicGen.get_pretrained(f'facebook/musicgen-{model_size}', device=self.device)
            self.model_name = model_size

    def regenerate(self, input_path, output_path, prompt, overlap_sec=5, temperature=1.0, progress_callback=None):
        import torchaudio
        
        # 1. Load & Preprocess
        wav, sr = torchaudio.load(input_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)
        
        # Mix to mono for conditioning
        if wav.shape[0] > 1: 
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        wav = wav.to(self.device)
        
        duration = wav.shape[1] / self.sample_rate
        chunk_len = 30 
        step = chunk_len - overlap_sec
        total_steps = math.ceil(duration / step)
        if total_steps == 0: total_steps = 1
        
        generated_chunks = []
        current_pos = 0
        
        logger.info(f"Processing: {duration:.2f}s | Temp: {temperature} | Model: {self.model_name}")
        
        for i in range(total_steps):
            if progress_callback: 
                progress_callback(f"Generating segment {i+1}/{total_steps}...")
            
            # Slice Audio (Used for Duration calc and Melody conditioning)
            end = min(current_pos + (chunk_len * self.sample_rate), wav.shape[1])
            chunk = wav[:, int(current_pos):int(end)]
            
            # Pad if too short
            if chunk.shape[1] < self.sample_rate: break 
            
            chunk = chunk.unsqueeze(0) # [1, 1, T]

            # Set Generation Params
            current_duration = chunk.shape[2] / self.sample_rate
            
            # Adaptive Top-K for stability
            top_k = 250
            if temperature < 0.8: top_k = 50
            if temperature < 0.5: top_k = 10 
            
            self.model.set_generation_params(
                duration=current_duration,
                temperature=temperature,
                top_k=top_k
            )
            
            with torch.no_grad():
                # BRANCH LOGIC: Check model type
                if "melody" in self.model_name:
                    # 'Melody' model can HEAR the input structure
                    res = self.model.generate_with_chroma(
                        descriptions=[prompt],
                        melody_wavs=chunk,
                        melody_sample_rate=self.sample_rate,
                        progress=False
                    )
                else:
                    # 'Medium/Large' models are deaf to input structure.
                    # They generate purely based on Text Prompt + Duration.
                    res = self.model.generate(
                        descriptions=[prompt],
                        progress=False
                    )
            
            generated_chunks.append(res[0].cpu())
            current_pos += (step * self.sample_rate)
            
            gc.collect()
            torch.cuda.empty_cache()

        if progress_callback: progress_callback("Stitching audio...")
        final_audio = self._crossfade_stitch(generated_chunks, overlap_sec)
        
        sf.write(output_path, final_audio.numpy().T, self.sample_rate)
        return output_path

    def _crossfade_stitch(self, chunks, overlap_sec):
        overlap_samples = int(overlap_sec * self.sample_rate)
        if not chunks: return torch.zeros(1, 1)
        
        full = chunks[0]
        
        for i in range(1, len(chunks)):
            nxt = chunks[i]
            
            prev_tail = full[:, -overlap_samples:]
            next_head = nxt[:, :overlap_samples]
            
            size = min(prev_tail.shape[1], next_head.shape[1])
            if size == 0:
                full = torch.cat([full, nxt], dim=1)
                continue
            
            # Linear Crossfade
            fade_out = torch.linspace(1, 0, size)
            fade_in = torch.linspace(0, 1, size)
            
            cross = (prev_tail[:, :size] * fade_out) + (next_head[:, :size] * fade_in)
            
            full = full[:, :-size]
            full = torch.cat([full, cross, nxt[:, size:]], dim=1)
            
        return full





class LinkwitzRileyCrossover:
    """
    Phase-aligned frequency band splitter using Linkwitz-Riley crossover filters.
    
    When bands are summed, they reconstruct the original signal perfectly.
    This is critical for preventing artifacts at crossover frequencies.
    """
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def split_bands(self, audio, low_freq=200, high_freq=4000, order=4):
        """
        Split audio into low, mid, high frequency bands.
        
        Args:
            audio: numpy array (samples,) or (samples, 2)
            low_freq: crossover frequency between low and mid (Hz)
            high_freq: crossover frequency between mid and high (Hz)
            order: filter order (higher = steeper rolloff)
            
        Returns:
            low, mid, high: numpy arrays of same shape as input
        """
        from scipy.signal import butter, sosfiltfilt
        
        is_stereo = len(audio.shape) > 1 and audio.shape[1] == 2
        
        if is_stereo:
            low_l, mid_l, high_l = self._split_mono(audio[:, 0], low_freq, high_freq, order)
            low_r, mid_r, high_r = self._split_mono(audio[:, 1], low_freq, high_freq, order)
            
            low = np.stack([low_l, low_r], axis=1)
            mid = np.stack([mid_l, mid_r], axis=1)
            high = np.stack([high_l, high_r], axis=1)
        else:
            audio_mono = audio.flatten() if len(audio.shape) > 1 else audio
            low, mid, high = self._split_mono(audio_mono, low_freq, high_freq, order)
        
        return low.astype(np.float32), mid.astype(np.float32), high.astype(np.float32)
    
    def _split_mono(self, audio, low_freq, high_freq, order):
        """Split mono audio into three bands."""
        from scipy.signal import butter, sosfiltfilt
        
        nyquist = self.sample_rate / 2
        
        # Clamp frequencies to valid range
        low_freq = min(low_freq, nyquist * 0.95)
        high_freq = min(high_freq, nyquist * 0.95)
        high_freq = max(high_freq, low_freq * 1.1)  # Ensure high > low
        
        # Design Linkwitz-Riley filters (cascaded Butterworth)
        # LR filters are created by cascading two Butterworth filters
        
        # Low-pass at low_freq
        sos_lp_low = butter(order // 2, low_freq, btype='low', fs=self.sample_rate, output='sos')
        
        # High-pass at low_freq (for mid and high)
        sos_hp_low = butter(order // 2, low_freq, btype='high', fs=self.sample_rate, output='sos')
        
        # Low-pass at high_freq (for mid)
        sos_lp_high = butter(order // 2, high_freq, btype='low', fs=self.sample_rate, output='sos')
        
        # High-pass at high_freq (for high)
        sos_hp_high = butter(order // 2, high_freq, btype='high', fs=self.sample_rate, output='sos')
        
        # Apply filters with zero-phase (forward-backward)
        # Low band: cascade two low-pass filters
        low = sosfiltfilt(sos_lp_low, audio)
        low = sosfiltfilt(sos_lp_low, low)  # Second pass for LR4
        
        # High band: cascade two high-pass filters
        high = sosfiltfilt(sos_hp_high, audio)
        high = sosfiltfilt(sos_hp_high, high)  # Second pass for LR4
        
        # Mid band: high-pass at low_freq, then low-pass at high_freq
        mid = sosfiltfilt(sos_hp_low, audio)
        mid = sosfiltfilt(sos_hp_low, mid)  # LR4 high-pass
        mid = sosfiltfilt(sos_lp_high, mid)
        mid = sosfiltfilt(sos_lp_high, mid)  # LR4 low-pass
        
        # Verify reconstruction (for debugging)
        # reconstruction = low + mid + high
        # error = np.max(np.abs(audio - reconstruction))
        # Should be very small (< 1e-6)
        
        return low, mid, high
    
    def validate_reconstruction(self, original, low, mid, high):
        """Verify that bands sum back to original."""
        reconstruction = low + mid + high
        error = np.max(np.abs(original - reconstruction))
        energy_error = np.sum((original - reconstruction) ** 2) / (np.sum(original ** 2) + 1e-10)
        
        return {
            'max_error': error,
            'energy_error': energy_error,
            'is_valid': error < 0.01  # Should be essentially zero
        }


class ExtractionWorker(QThread):
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, input_file, output_dir, mode, demucs_model, 
                 separation_model, separation_method, enable_restoration,
                 restoration_model, restoration_preset, threshold, max_regions,
                 shifts, overlap, use_float32, device,
                 # MusicGen params
                 musicgen_prompt="", musicgen_model="melody", 
                 musicgen_preprocess=False, musicgen_temp=1.0,
                 # Honest separation params
                 stem_grouping="extended_6",
                 low_freq=250, high_freq=6000,
                 # Preset params
                 quality_preset=None,
                 frequency_preset=None):
        
        super().__init__()
        self.input_file = input_file
        self.output_dir = output_dir
        self.mode = mode
        self.demucs_model = demucs_model
        self.separation_model = separation_model
        self.separation_method = separation_method
        self.enable_restoration = enable_restoration
        self.restoration_model = restoration_model
        self.restoration_preset = restoration_preset
        self.threshold = threshold
        self.max_regions = max_regions
        self.use_float32 = use_float32
        self.device = device
        
        self.musicgen_prompt = musicgen_prompt
        self.musicgen_model = musicgen_model
        self.musicgen_preprocess = musicgen_preprocess
        self.musicgen_temp = musicgen_temp
        
        # Honest separation params
        self.stem_grouping = stem_grouping
        
        # Handle quality preset - OVERRIDE manual values if preset is selected
        if quality_preset and quality_preset in DEMUCS_QUALITY_PRESETS:
            preset = DEMUCS_QUALITY_PRESETS[quality_preset]
            self.shifts = preset["shifts"]
            self.overlap = preset["overlap"]
            self.quality_preset = quality_preset
        else:
            self.shifts = shifts
            self.overlap = overlap
            self.quality_preset = None
        
        # Handle frequency preset - OVERRIDE manual values if preset is selected
        if frequency_preset:
            freq_preset = None
            for preset_dict in [FREQUENCY_PRESETS_GENRE, FREQUENCY_PRESETS_TECHNICAL, FREQUENCY_PRESETS_INSTRUMENT]:
                if frequency_preset in preset_dict:
                    freq_preset = preset_dict[frequency_preset]
                    break
            
            if freq_preset:
                self.low_freq = freq_preset["low_freq"]
                self.high_freq = freq_preset["high_freq"]
                self.frequency_preset = frequency_preset
            else:
                self.low_freq = low_freq
                self.high_freq = high_freq
                self.frequency_preset = None
        else:
            self.low_freq = low_freq
            self.high_freq = high_freq
            self.frequency_preset = None

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # =====================================================================
            # HONEST STEMS MODE
            # =====================================================================
            if self.mode == "honest_stems":
                self.progress.emit("Initializing honest stem separator...")
                
                # Log the settings being used
                if self.quality_preset:
                    preset = DEMUCS_QUALITY_PRESETS[self.quality_preset]
                    self.progress.emit(f"Quality preset: {preset['name']} (shifts={self.shifts}, overlap={self.overlap:.0%})")
                else:
                    self.progress.emit(f"Custom quality: shifts={self.shifts}, overlap={self.overlap:.0%}")
                
                if self.frequency_preset:
                    self.progress.emit(f"Frequency preset: {self.frequency_preset} (low={self.low_freq}Hz, high={self.high_freq}Hz)")
                else:
                    self.progress.emit(f"Custom frequencies: low={self.low_freq}Hz, high={self.high_freq}Hz")
                
                separator = HonestStemSeparator(self.output_dir, self.device)
                
                try:
                    # Pass the resolved values directly (presets already applied in __init__)
                    stems, sr = separator.separate(
                        self.input_file,
                        grouping=self.stem_grouping,
                        low_freq=self.low_freq,
                        high_freq=self.high_freq,
                        demucs_model=self.demucs_model,
                        shifts=self.shifts,
                        overlap=self.overlap,
                        # Don't pass presets here - already resolved above
                        quality_preset=None,
                        frequency_preset=None,
                        progress_callback=self.progress.emit
                    )
                    
                    # Save stems
                    for stem_name, stem_audio in stems.items():
                        output_path = os.path.join(self.output_dir, f"{stem_name}.wav")
                        sf.write(output_path, stem_audio, sr)
                        self.progress.emit(f"✅ Saved: {stem_name}.wav")
                    
                    self.progress.emit("✅ All stems saved!")
                    
                finally:
                    separator.cleanup()
                
                self.finished.emit(self.output_dir)
                return
            
            # =====================================================================
            # MUSICGEN MODE
            # =====================================================================
            if self.mode == "musicgen_regen":
                if not AVAILABLE_MODELS.get("musicgen", False):
                    raise Exception("MusicGen not installed.")
                
                target_file = self.input_file
                
                if self.musicgen_preprocess:
                    self.progress.emit("Pre-processing: Removing vocals with Demucs...")
                    
                    model = get_model("htdemucs")
                    model.to(self.device)
                    wav, sr = torchaudio.load(self.input_file)
                    
                    ref = wav.mean(0)
                    wav = (wav - ref.mean()) / ref.std()
                    sources = apply_model(model, wav[None], shifts=0, overlap=0.25)[0]
                    sources = sources * ref.std() + ref.mean()
                    
                    no_vocals = sources[0] + sources[1] + sources[2]
                    
                    temp_inst_path = os.path.join(self.output_dir, "temp_instrumental_input.wav")
                    torchaudio.save(temp_inst_path, no_vocals.cpu(), sr)
                    
                    target_file = temp_inst_path
                    self.progress.emit("Vocals removed. Initializing MusicGen...")
                    
                    del model
                    torch.cuda.empty_cache()

                mg = MusicGenRegenerator(device=self.device)
                self.progress.emit(f"Loading MusicGen model: {self.musicgen_model}...")
                mg.load_model(self.musicgen_model)
                
                out_file = os.path.join(self.output_dir, "regenerated_song.wav")
                prompt = self.musicgen_prompt if self.musicgen_prompt.strip() else "high quality instrumental song"
                
                mg.regenerate(
                    target_file, 
                    out_file, 
                    prompt=prompt,
                    temperature=self.musicgen_temp,
                    progress_callback=self.progress.emit
                )
                
                if self.musicgen_preprocess and os.path.exists(target_file) and target_file != self.input_file:
                    try:
                        os.remove(target_file)
                    except:
                        pass
                
                self.finished.emit(self.output_dir)
                return

            # =====================================================================
            # FULL STEMS MODE (Demucs only)
            # =====================================================================
            if self.mode == "full_stems":
                self.progress.emit(f"Starting Demucs (shifts={self.shifts}, overlap={self.overlap:.0%})...")
                
                model = get_model(self.demucs_model)
                model.to(self.device)
                wav, sr = torchaudio.load(self.input_file)
                
                ref = wav.mean(0)
                wav = (wav - ref.mean()) / ref.std()
                sources = apply_model(model, wav[None], shifts=self.shifts, overlap=self.overlap)[0]
                sources = sources * ref.std() + ref.mean()
                
                for source, name in zip(sources, model.sources):
                    output_path = os.path.join(self.output_dir, f"{name}.wav")
                    torchaudio.save(output_path, source.cpu(), sr)
                    self.progress.emit(f"✅ Saved: {name}.wav")
                
                del model
                torch.cuda.empty_cache()
                
                self.finished.emit(self.output_dir)
                return

            # =====================================================================
            # VOCALS ONLY / INSTRUMENT HQ MODE (UVR)
            # =====================================================================
            if self.mode in ["vocals_only", "instrument_hq"]:
                self.progress.emit("Starting UVR Separation...")
                sep = UVRSeparator(self.output_dir, self.device)
                
                vocals_path, instrumental_path = sep.separate_single(
                    self.input_file, 
                    self.separation_model, 
                    progress_callback=self.progress.emit
                )
                
                if vocals_path and os.path.exists(vocals_path):
                    final_vocals = os.path.join(self.output_dir, "vocals.wav")
                    shutil.copy2(vocals_path, final_vocals)
                    self.progress.emit(f"✅ Saved: vocals.wav")
                
                if instrumental_path and os.path.exists(instrumental_path):
                    final_inst = os.path.join(self.output_dir, "instrumental.wav")
                    shutil.copy2(instrumental_path, final_inst)
                    self.progress.emit(f"✅ Saved: instrumental.wav")
                
                sep.cleanup()
                
                self.finished.emit(self.output_dir)
                return

            self.error.emit(f"Unknown mode: {self.mode}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class HonestStemSeparator:
    """
    Honest stem separation that uses the right tool for each job.
    Now with preset support for quality and frequency settings.
    """
    
    GROUPINGS = {
        "demucs_4": {
            "description": "Standard 4 stems from Demucs",
            "stems": ["vocals", "drums", "bass", "other"],
        },
        "extended_6": {
            "description": "4 Demucs stems + other split by frequency",
            "stems": ["vocals", "drums", "bass", "other_low", "other_mid", "other_high"],
        },
        "music_focus": {
            "description": "Vocals, rhythm (drums+bass), and melodic content by frequency",
            "stems": ["vocals", "rhythm", "melody_low", "melody_mid", "melody_high"],
        },
    }
    
    def __init__(self, output_dir, device=None):
        self.output_dir = output_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.demucs_model = None
        self.demucs_model_name = None
        self.crossover = LinkwitzRileyCrossover()
    
    def separate(self, audio_path, grouping="extended_6",
                 low_freq=None, high_freq=None,
                 demucs_model="htdemucs", shifts=2, overlap=0.25,
                 quality_preset=None, frequency_preset=None,
                 progress_callback=None):
        """
        Perform honest stem separation.
        
        Args:
            audio_path: path to input audio
            grouping: output grouping preset
            low_freq: low/mid crossover (overridden by frequency_preset)
            high_freq: mid/high crossover (overridden by frequency_preset)
            demucs_model: which Demucs model to use
            shifts: Demucs shifts (overridden by quality_preset)
            overlap: Demucs overlap (overridden by quality_preset)
            quality_preset: key from DEMUCS_QUALITY_PRESETS
            frequency_preset: key from FREQUENCY_PRESETS_* dicts
            
        Returns:
            dict of stem_name -> audio numpy array, sample_rate
        """
        
        # Apply quality preset if specified
        if quality_preset and quality_preset in DEMUCS_QUALITY_PRESETS:
            preset = DEMUCS_QUALITY_PRESETS[quality_preset]
            shifts = preset["shifts"]
            overlap = preset["overlap"]
            if progress_callback:
                progress_callback(f"Using quality preset: {preset['name']}")
        
        # Apply frequency preset if specified
        if frequency_preset:
            # Check all frequency preset dictionaries
            freq_preset = None
            for preset_dict in [FREQUENCY_PRESETS_GENRE, FREQUENCY_PRESETS_TECHNICAL, FREQUENCY_PRESETS_INSTRUMENT]:
                if frequency_preset in preset_dict:
                    freq_preset = preset_dict[frequency_preset]
                    break
            
            if freq_preset:
                low_freq = freq_preset["low_freq"]
                high_freq = freq_preset["high_freq"]
                if progress_callback:
                    progress_callback(f"Using frequency preset: {freq_preset['name']}")
        
        # Default frequencies if not set
        low_freq = low_freq or 250
        high_freq = high_freq or 6000
        
        if progress_callback:
            progress_callback(f"Settings: shifts={shifts}, overlap={overlap:.0%}, low={low_freq}Hz, high={high_freq}Hz")
        
        # Validate
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # =====================================================================
        # Stage 1: Demucs separation
        # =====================================================================
        if progress_callback:
            progress_callback("Stage 1: Running Demucs separation...")
        
        demucs_stems, sr = self._run_demucs(
            audio_path, demucs_model, shifts, overlap, progress_callback
        )
        
        if progress_callback:
            progress_callback(f"Demucs complete. Got {len(demucs_stems)} stems.")
        
        # =====================================================================
        # Stage 2: Frequency split on "other" (if requested)
        # =====================================================================
        other_low = None
        other_mid = None
        other_high = None
        
        if grouping in ["extended_6", "music_focus"]:
            if progress_callback:
                progress_callback(f"Stage 2: Splitting 'other' at {low_freq}Hz and {high_freq}Hz...")
            
            self.crossover.sample_rate = sr
            
            other_low, other_mid, other_high = self.crossover.split_bands(
                demucs_stems["other"],
                low_freq=low_freq,
                high_freq=high_freq
            )
            
            # Validate split
            validation = self.crossover.validate_reconstruction(
                demucs_stems["other"], other_low, other_mid, other_high
            )
            
            if validation['is_valid']:
                if progress_callback:
                    progress_callback(f"✅ Frequency split verified (error: {validation['energy_error']:.6f})")
            else:
                logger.warning(f"Crossover validation: {validation}")
        
        # =====================================================================
        # Stage 3: Build output based on grouping
        # =====================================================================
        if progress_callback:
            progress_callback("Stage 3: Building final stems...")
        
        if grouping == "demucs_4":
            final_stems = {
                "vocals": demucs_stems["vocals"],
                "drums": demucs_stems["drums"],
                "bass": demucs_stems["bass"],
                "other": demucs_stems["other"],
            }
        
        elif grouping == "extended_6":
            final_stems = {
                "vocals": demucs_stems["vocals"],
                "drums": demucs_stems["drums"],
                "bass": demucs_stems["bass"],
                "other_low": other_low,
                "other_mid": other_mid,
                "other_high": other_high,
            }
        
        elif grouping == "music_focus":
            rhythm = self._safe_add(demucs_stems["drums"], demucs_stems["bass"])
            
            final_stems = {
                "vocals": demucs_stems["vocals"],
                "rhythm": rhythm,
                "melody_low": other_low,
                "melody_mid": other_mid,
                "melody_high": other_high,
            }
        
        else:
            final_stems = demucs_stems
        
        # =====================================================================
        # Verification
        # =====================================================================
        if progress_callback:
            progress_callback("Verifying separation...")
        
        original, _ = sf.read(audio_path)
        self._verify_stems(original, final_stems, demucs_stems, grouping, progress_callback)
        
        if progress_callback:
            progress_callback("✅ Separation complete!")
        
        return final_stems, sr
    
    def _run_demucs(self, audio_path, model_name, shifts, overlap, progress_callback):
        """Run Demucs separation."""
        # Load model (cache if same model)
        if self.demucs_model is None or self.demucs_model_name != model_name:
            if progress_callback:
                progress_callback(f"Loading Demucs model: {model_name}...")
            
            self.demucs_model = get_model(model_name)
            self.demucs_model.to(self.device)
            self.demucs_model_name = model_name
        
        # Load audio
        if progress_callback:
            progress_callback("Loading audio file...")
        
        wav, sr = torchaudio.load(audio_path)
        
        logger.info(f"Loaded audio: shape={wav.shape}, sr={sr}")
        
        # Normalize
        ref = wav.mean(0)
        wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)
        
        # Run separation
        if progress_callback:
            progress_callback(f"Separating (shifts={shifts}, overlap={overlap:.0%})...")
        
        with torch.no_grad():
            sources = apply_model(
                self.demucs_model,
                wav_norm[None].to(self.device),
                shifts=shifts,
                overlap=overlap
            )[0]
        
        # Denormalize
        sources = sources * ref.std() + ref.mean()
        
        # Convert to numpy: (num_sources, channels, samples) -> (samples, channels)
        stems = {}
        source_names = self.demucs_model.sources
        
        for i, name in enumerate(source_names):
            source = sources[i].cpu().numpy()
            source = source.T  # (C, N) -> (N, C)
            stems[name] = source.astype(np.float32)
            logger.debug(f"Stem '{name}': shape={stems[name].shape}")
        
        return stems, sr
    
    def _safe_add(self, audio1, audio2):
        """Safely add two audio arrays."""
        min_len = min(len(audio1), len(audio2))
        
        is_stereo_1 = len(audio1.shape) > 1 and audio1.shape[1] == 2
        is_stereo_2 = len(audio2.shape) > 1 and audio2.shape[1] == 2
        
        if is_stereo_1 and is_stereo_2:
            return (audio1[:min_len, :] + audio2[:min_len, :]).astype(np.float32)
        elif is_stereo_1:
            a2 = np.stack([audio2[:min_len], audio2[:min_len]], axis=1) if len(audio2.shape) == 1 else audio2[:min_len]
            return (audio1[:min_len, :] + a2).astype(np.float32)
        elif is_stereo_2:
            a1 = np.stack([audio1[:min_len], audio1[:min_len]], axis=1) if len(audio1.shape) == 1 else audio1[:min_len]
            return (a1 + audio2[:min_len, :]).astype(np.float32)
        else:
            return (audio1[:min_len] + audio2[:min_len]).astype(np.float32)
    
    def _verify_stems(self, original, final_stems, demucs_stems, grouping, progress_callback):
        """Verify separation quality."""
        demucs_sum = None
        for name in ["vocals", "drums", "bass", "other"]:
            if name in demucs_stems:
                if demucs_sum is None:
                    demucs_sum = demucs_stems[name].copy()
                else:
                    demucs_sum = self._safe_add(demucs_sum, demucs_stems[name])
        
        if demucs_sum is not None:
            min_len = min(len(original), len(demucs_sum))
            
            if len(original.shape) > 1:
                orig_energy = np.sum(original[:min_len] ** 2)
                diff_energy = np.sum((original[:min_len] - demucs_sum[:min_len]) ** 2)
            else:
                orig_energy = np.sum(original[:min_len] ** 2)
                demucs_mono = np.mean(demucs_sum[:min_len], axis=1) if len(demucs_sum.shape) > 1 else demucs_sum[:min_len]
                diff_energy = np.sum((original[:min_len] - demucs_mono) ** 2)
            
            error_ratio = diff_energy / (orig_energy + 1e-10)
            
            if progress_callback:
                if error_ratio < 0.01:
                    progress_callback(f"✅ Reconstruction verified (error: {error_ratio:.4%})")
                else:
                    progress_callback(f"⚠️ Reconstruction error: {error_ratio:.4%}")
    
    def cleanup(self):
        """Release resources."""
        if self.demucs_model is not None:
            del self.demucs_model
            self.demucs_model = None
            self.demucs_model_name = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()




class StemSeparatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Stem Separator - Honest Edition")
        self.setGeometry(100, 100, 600, 950)
        self.setMinimumWidth(500)
        
        self.worker_thread = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        
        widget = QWidget()
        scroll.setWidget(widget)
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # =================================================================
        # STATUS LABEL
        # =================================================================
        self.status_label = QLabel("Drag & drop or browse for an audio file.")
        layout.addWidget(self.status_label)
        
        # Model availability warnings
        self._add_availability_warnings(layout)
        
        # =================================================================
        # DROP ZONE
        # =================================================================
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.handle_dropped_file)
        layout.addWidget(self.drop_zone)
        
        # =================================================================
        # FILE SELECTION
        # =================================================================
        file_group = QGroupBox("📁 File")
        file_layout = QVBoxLayout(file_group)
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        file_layout.addWidget(self.file_input)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)
        layout.addWidget(file_group)

        # =================================================================
        # MODE SELECTION
        # =================================================================
        mode_group = QGroupBox("🎯 Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_group = QButtonGroup(self)
        
        modes = [
            ("🎤 Vocals Only (UVR)", "vocals_only", False),
            ("🎸 High-Quality Instrumental (UVR)", "instrument_hq", False),
            ("🎹 Full Stems - Demucs 4", "full_stems", False),
            ("🎼 Honest Stems (RECOMMENDED)", "honest_stems", True),
            ("🤖 MusicGen Re-creation", "musicgen_regen", False),
        ]
        
        for text, mode, checked in modes:
            radio = QRadioButton(text)
            radio.mode = mode
            radio.setChecked(checked)
            self.mode_group.addButton(radio)
            mode_layout.addWidget(radio)
        
        layout.addWidget(mode_group)

        # =================================================================
        # HONEST SEPARATION SETTINGS
        # =================================================================
        self.honest_group = QGroupBox("🎼 Honest Separation Settings")
        honest_layout = QVBoxLayout(self.honest_group)
        honest_layout.setSpacing(8)
        
        # Description
        desc_label = QLabel(
            "Uses Demucs for vocals/drums/bass/other, then optionally splits 'other' by frequency."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #666; font-size: 11px;")
        honest_layout.addWidget(desc_label)
        
        # Output grouping
        grouping_layout = QHBoxLayout()
        grouping_label = QLabel("Output:")
        grouping_label.setFixedWidth(80)
        grouping_layout.addWidget(grouping_label)
        
        self.honest_grouping_combo = QComboBox()
        self.honest_grouping_combo.addItem("Demucs 4 (vocals/drums/bass/other)", "demucs_4")
        self.honest_grouping_combo.addItem("Extended 6 (+ other split by freq)", "extended_6")
        self.honest_grouping_combo.addItem("Music Focus (vocals/rhythm/melody×3)", "music_focus")
        self.honest_grouping_combo.setCurrentIndex(1)
        self.honest_grouping_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        grouping_layout.addWidget(self.honest_grouping_combo)
        honest_layout.addLayout(grouping_layout)
        
        # Grouping info
        self.honest_grouping_info = QLabel("")
        self.honest_grouping_info.setWordWrap(True)
        self.honest_grouping_info.setStyleSheet(
            "color: #007bff; font-size: 10px; padding: 5px; "
            "background-color: #f0f8ff; border-radius: 4px;"
        )
        honest_layout.addWidget(self.honest_grouping_info)
        
        # ─────────────────────────────────────────────────────────────────
        # QUALITY PRESET
        # ─────────────────────────────────────────────────────────────────
        quality_group = QGroupBox("⚡ Quality Preset")
        quality_layout = QVBoxLayout(quality_group)
        quality_layout.setSpacing(5)
        
        self.quality_preset_combo = QComboBox()
        self.quality_preset_combo.addItem("✏️ Custom (manual settings)", "custom")
        for preset_id, preset in DEMUCS_QUALITY_PRESETS.items():
            self.quality_preset_combo.addItem(
                f"{preset['name']} - {preset['time_estimate']}", 
                preset_id
            )
        self.quality_preset_combo.setCurrentIndex(2)  # Default to "Fast"
        quality_layout.addWidget(self.quality_preset_combo)
        
        self.quality_preset_info = QLabel("")
        self.quality_preset_info.setWordWrap(True)
        self.quality_preset_info.setStyleSheet("color: #666; font-size: 10px;")
        quality_layout.addWidget(self.quality_preset_info)
        
        honest_layout.addWidget(quality_group)
        
        # ─────────────────────────────────────────────────────────────────
        # FREQUENCY PRESET
        # ─────────────────────────────────────────────────────────────────
        self.freq_preset_group = QGroupBox("🎚️ Frequency Preset (for 'other' splitting)")
        freq_preset_layout = QVBoxLayout(self.freq_preset_group)
        freq_preset_layout.setSpacing(5)
        
        # Category selection
        freq_cat_layout = QHBoxLayout()
        freq_cat_layout.addWidget(QLabel("Category:"))
        self.freq_category_combo = QComboBox()
        self.freq_category_combo.addItem("🎵 By Genre", "genre")
        self.freq_category_combo.addItem("🎹 By Instrument", "instrument")
        self.freq_category_combo.addItem("📊 Technical", "technical")
        self.freq_category_combo.addItem("✏️ Custom", "custom")
        freq_cat_layout.addWidget(self.freq_category_combo)
        freq_preset_layout.addLayout(freq_cat_layout)
        
        # Preset selection
        self.freq_preset_combo = QComboBox()
        freq_preset_layout.addWidget(self.freq_preset_combo)
        
        # Preset info
        self.freq_preset_info = QLabel("")
        self.freq_preset_info.setWordWrap(True)
        self.freq_preset_info.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        freq_preset_layout.addWidget(self.freq_preset_info)
        
        # Manual frequency controls
        self.freq_manual_widget = QWidget()
        freq_manual_layout = QVBoxLayout(self.freq_manual_widget)
        freq_manual_layout.setContentsMargins(0, 5, 0, 0)
        
        low_freq_layout = QHBoxLayout()
        low_freq_layout.addWidget(QLabel("Low/Mid split:"))
        self.honest_low_freq_spin = QSpinBox()
        self.honest_low_freq_spin.setRange(50, 500)
        self.honest_low_freq_spin.setValue(250)
        self.honest_low_freq_spin.setSuffix(" Hz")
        self.honest_low_freq_spin.setFixedWidth(100)
        low_freq_layout.addWidget(self.honest_low_freq_spin)
        low_freq_layout.addStretch()
        freq_manual_layout.addLayout(low_freq_layout)
        
        high_freq_layout = QHBoxLayout()
        high_freq_layout.addWidget(QLabel("Mid/High split:"))
        self.honest_high_freq_spin = QSpinBox()
        self.honest_high_freq_spin.setRange(2000, 12000)
        self.honest_high_freq_spin.setValue(6000)
        self.honest_high_freq_spin.setSuffix(" Hz")
        self.honest_high_freq_spin.setFixedWidth(100)
        high_freq_layout.addWidget(self.honest_high_freq_spin)
        high_freq_layout.addStretch()
        freq_manual_layout.addLayout(high_freq_layout)
        
        freq_preset_layout.addWidget(self.freq_manual_widget)
        honest_layout.addWidget(self.freq_preset_group)
        
        layout.addWidget(self.honest_group)

        # =================================================================
        # UVR SEPARATION SETTINGS
        # =================================================================
        self.sep_group = QGroupBox("🎵 UVR Separation Settings")
        sep_layout = QVBoxLayout(self.sep_group)
        sep_layout.setSpacing(8)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("UVR Single Model (Fast)", "uvr_single")
        self.method_combo.addItem("UVR Ensemble (Best)", "uvr_ensemble")
        self.method_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        method_layout.addWidget(self.method_combo)
        sep_layout.addLayout(method_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.sep_model_combo = QComboBox()
        for model_id, model_name in UVRSeparator.INSTRUMENTAL_MODELS:
            self.sep_model_combo.addItem(model_name, model_id)
        self.sep_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.sep_model_combo)
        sep_layout.addLayout(model_layout)

        layout.addWidget(self.sep_group)

        # =================================================================
        # RESTORATION SETTINGS
        # =================================================================
        self.restore_group = QGroupBox("🔧 AI Restoration")
        self.restore_group.setCheckable(True)
        self.restore_group.setChecked(False)
        restore_layout = QVBoxLayout(self.restore_group)
        restore_layout.setSpacing(8)

        restore_model_layout = QHBoxLayout()
        restore_model_layout.addWidget(QLabel("Model:"))
        self.restore_model_combo = QComboBox()
        self.restore_model_combo.addItem("⚡ Spectral Restore (Fast)", "spectral")
        self.restore_model_combo.addItem("🎵 Enhanced Spectral (Medium)", "enhanced_spectral")
        self.restore_model_combo.addItem("🧠 Diffusion Inpaint (Best)", "diffusion_inpaint")
        self.restore_model_combo.addItem("🔊 AudioSR (Legacy)", "audiosr")
        self.restore_model_combo.setCurrentIndex(1)
        self.restore_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        restore_model_layout.addWidget(self.restore_model_combo)
        restore_layout.addLayout(restore_model_layout)

        self.restore_info_label = QLabel("🎵 Harmonic-aware restoration")
        self.restore_info_label.setStyleSheet("color: #666; font-size: 10px;")
        restore_layout.addWidget(self.restore_info_label)

        # Presets
        preset_layout = QHBoxLayout()
        self.preset_group = QButtonGroup(self)
        for text, preset_id in [("Fast", "fast"), ("Balanced", "balanced"), ("Quality", "high_quality")]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.preset_id = preset_id
            btn.setStyleSheet("QPushButton { padding: 5px; } QPushButton:checked { background-color: #007bff; color: white; }")
            if preset_id == "balanced":
                btn.setChecked(True)
            self.preset_group.addButton(btn)
            preset_layout.addWidget(btn)
        restore_layout.addLayout(preset_layout)

        # Sensitivity
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Sensitivity:"))
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(20, 80)
        self.sens_slider.setValue(50)
        sens_layout.addWidget(self.sens_slider)
        self.sens_value = QLabel("0.50")
        sens_layout.addWidget(self.sens_value)
        self.sens_slider.valueChanged.connect(lambda v: self.sens_value.setText(f"{v/100:.2f}"))
        restore_layout.addLayout(sens_layout)

        # Max regions
        regions_layout = QHBoxLayout()
        regions_layout.addWidget(QLabel("Max regions:"))
        self.max_regions_spin = QSpinBox()
        self.max_regions_spin.setRange(1, 50)
        self.max_regions_spin.setValue(10)
        regions_layout.addWidget(self.max_regions_spin)
        regions_layout.addStretch()
        restore_layout.addLayout(regions_layout)

        layout.addWidget(self.restore_group)

        # =================================================================
        # MUSICGEN SETTINGS
        # =================================================================
        self.musicgen_group = QGroupBox("🎹 MusicGen Settings")
        mg_layout = QVBoxLayout(self.musicgen_group)
        
        mg_model_layout = QHBoxLayout()
        mg_model_layout.addWidget(QLabel("Model:"))
        self.mg_model_combo = QComboBox()
        self.mg_model_combo.addItem("Melody (Best Structure)", "melody")
        self.mg_model_combo.addItem("Medium (Better Quality)", "medium")
        self.mg_model_combo.addItem("Large (Best / Slow)", "large")
        self.mg_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        mg_model_layout.addWidget(self.mg_model_combo)
        mg_layout.addLayout(mg_model_layout)
        
        self.mg_preprocess_check = QCheckBox("⚡ Pre-separate Vocals")
        self.mg_preprocess_check.setChecked(True)
        mg_layout.addWidget(self.mg_preprocess_check)

        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Creativity:"))
        self.mg_temp_slider = QSlider(Qt.Horizontal)
        self.mg_temp_slider.setRange(1, 15)
        self.mg_temp_slider.setValue(8)
        temp_layout.addWidget(self.mg_temp_slider)
        self.mg_temp_label = QLabel("0.8")
        temp_layout.addWidget(self.mg_temp_label)
        self.mg_temp_slider.valueChanged.connect(lambda v: self.mg_temp_label.setText(f"{v/10:.1f}"))
        mg_layout.addLayout(temp_layout)

        mg_layout.addWidget(QLabel("Style Prompt:"))
        self.mg_prompt_input = QTextEdit()
        self.mg_prompt_input.setPlaceholderText("e.g. Acoustic guitar version...")
        self.mg_prompt_input.setText("High quality instrumental song")
        self.mg_prompt_input.setMaximumHeight(50)
        mg_layout.addWidget(self.mg_prompt_input)
        
        layout.addWidget(self.musicgen_group)

        # =================================================================
        # DEMUCS SETTINGS
        # =================================================================
        self.demucs_group = QGroupBox("⚙️ Demucs Settings")
        demucs_layout = QVBoxLayout(self.demucs_group)
        demucs_layout.setSpacing(8)
        
        # Model
        dm_layout = QHBoxLayout()
        dm_layout.addWidget(QLabel("Model:"))
        self.demucs_model_combo = QComboBox()
        self.demucs_model_combo.addItem("HTDemucs (Best)", "htdemucs")
        self.demucs_model_combo.addItem("HTDemucs FT", "htdemucs_ft")
        self.demucs_model_combo.addItem("HTDemucs 6s (Experimental)", "htdemucs_6s")
        self.demucs_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dm_layout.addWidget(self.demucs_model_combo)
        demucs_layout.addLayout(dm_layout)
        
        # Preset note
        self.demucs_preset_note = QLabel("💡 Shifts & Overlap controlled by Quality Preset")
        self.demucs_preset_note.setStyleSheet("color: #007bff; font-size: 10px; font-style: italic;")
        demucs_layout.addWidget(self.demucs_preset_note)
        
        # Shifts
        shifts_layout = QHBoxLayout()
        shifts_layout.addWidget(QLabel("Shifts:"))
        self.shifts_spin = QSpinBox()
        self.shifts_spin.setRange(1, 10)
        self.shifts_spin.setValue(2)
        self.shifts_spin.setFixedWidth(60)
        shifts_layout.addWidget(self.shifts_spin)
        shifts_layout.addStretch()
        demucs_layout.addLayout(shifts_layout)
        
        # Overlap
        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Overlap:"))
        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(10, 90)
        self.overlap_spin.setValue(25)
        self.overlap_spin.setSuffix("%")
        self.overlap_spin.setFixedWidth(70)
        overlap_layout.addWidget(self.overlap_spin)
        overlap_layout.addStretch()
        demucs_layout.addLayout(overlap_layout)
        
        # Float32
        self.float32_check = QCheckBox("Float32 Precision")
        self.float32_check.setChecked(True)
        demucs_layout.addWidget(self.float32_check)
        
        # Device
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.device_combo.addItem(f"🚀 GPU ({gpu_name})", "cuda")
            self.device_combo.addItem("🐢 CPU", "cpu")
        else:
            self.device_combo.addItem("🐢 CPU", "cpu")
        self.device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        device_layout.addWidget(self.device_combo)
        demucs_layout.addLayout(device_layout)
        
        if not torch.cuda.is_available():
            warn = QLabel("⚠️ No GPU - processing will be slow")
            warn.setStyleSheet("color: orange;")
            demucs_layout.addWidget(warn)
        
        layout.addWidget(self.demucs_group)

        # =================================================================
        # PROGRESS
        # =================================================================
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setMinimumHeight(50)
        progress_layout.addWidget(self.progress_label)
        
        self.start_btn = QPushButton("🚀 Start")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_separation)
        self.start_btn.setStyleSheet("""
            QPushButton { 
                background-color: #28a745; color: white; 
                font-weight: bold; padding: 12px; 
                border-radius: 5px; font-size: 14px;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        progress_layout.addWidget(self.start_btn)
        
        layout.addWidget(progress_group)
        layout.addStretch()
        
        # =================================================================
        # CONNECT SIGNALS (after ALL widgets exist)
        # =================================================================
        for btn in self.mode_group.buttons():
            btn.toggled.connect(self._on_mode_changed)
        
        self.honest_grouping_combo.currentIndexChanged.connect(self._on_honest_grouping_changed)
        self.quality_preset_combo.currentIndexChanged.connect(self._on_quality_preset_changed)
        self.freq_category_combo.currentIndexChanged.connect(self._on_freq_category_changed)
        self.freq_preset_combo.currentIndexChanged.connect(self._on_freq_preset_changed)
        
        # =================================================================
        # INITIALIZE STATE (after signals connected)
        # =================================================================
        self._populate_freq_presets()
        self._on_quality_preset_changed()
        self._on_honest_grouping_changed()
        self._on_mode_changed()
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _add_availability_warnings(self, layout):
        """Add warnings for unavailable features."""
        warnings = []
        if not AVAILABLE_MODELS.get("audio_separator"):
            warnings.append("UVR")
        if not AVAILABLE_MODELS.get("musicgen"):
            warnings.append("MusicGen")
        
        if warnings:
            label = QLabel(f"⚠️ Not installed: {', '.join(warnings)}")
            label.setStyleSheet(
                "color: #856404; background-color: #fff3cd; "
                "padding: 8px; border-radius: 4px; font-size: 11px;"
            )
            layout.addWidget(label)
    
    def _populate_freq_presets(self):
        """Populate frequency preset combo based on category."""
        category = self.freq_category_combo.currentData()
        self.freq_preset_combo.clear()
        
        if category == "genre":
            for pid, p in FREQUENCY_PRESETS_GENRE.items():
                self.freq_preset_combo.addItem(p["name"], pid)
        elif category == "instrument":
            for pid, p in FREQUENCY_PRESETS_INSTRUMENT.items():
                self.freq_preset_combo.addItem(p["name"], pid)
        elif category == "technical":
            for pid, p in FREQUENCY_PRESETS_TECHNICAL.items():
                self.freq_preset_combo.addItem(p["name"], pid)
        else:  # custom
            self.freq_preset_combo.addItem("Manual Settings", "custom")
        
        self._on_freq_preset_changed()
    
    def _on_mode_changed(self):
        """Update UI based on selected mode."""
        mode = self.get_mode()
        
        self.honest_group.setVisible(mode == "honest_stems")
        self.sep_group.setVisible(mode in ["vocals_only", "instrument_hq"])
        self.restore_group.setVisible(mode == "instrument_hq")
        self.musicgen_group.setVisible(mode == "musicgen_regen")
        self.demucs_group.setVisible(mode in ["full_stems", "honest_stems"])
    
    def _on_honest_grouping_changed(self):
        """Update grouping info and frequency settings visibility."""
        grouping = self.honest_grouping_combo.currentData()
        
        info = {
            "demucs_4": "Output: vocals, drums, bass, other",
            "extended_6": "Output: vocals, drums, bass, other_low, other_mid, other_high",
            "music_focus": "Output: vocals, rhythm, melody_low, melody_mid, melody_high",
        }
        self.honest_grouping_info.setText(info.get(grouping, ""))
        
        # Show frequency settings only when splitting "other"
        show_freq = grouping in ["extended_6", "music_focus"]
        self.freq_preset_group.setVisible(show_freq)
    
    def _on_quality_preset_changed(self):
        """Update quality preset info and Demucs controls."""
        preset_id = self.quality_preset_combo.currentData()
        is_custom = (preset_id == "custom")
        
        # Update note visibility
        self.demucs_preset_note.setVisible(not is_custom)
        
        # Enable/disable manual controls
        self.shifts_spin.setEnabled(is_custom)
        self.overlap_spin.setEnabled(is_custom)
        
        if is_custom:
            self.quality_preset_info.setText("Using manual Shifts and Overlap settings below.")
        elif preset_id in DEMUCS_QUALITY_PRESETS:
            p = DEMUCS_QUALITY_PRESETS[preset_id]
            self.quality_preset_info.setText(
                f"{p['description']}\n"
                f"Shifts: {p['shifts']} | Overlap: {p['overlap']:.0%} | {p['time_estimate']}"
            )
            # Update spinboxes to show values (disabled, just for display)
            self.shifts_spin.setValue(p['shifts'])
            self.overlap_spin.setValue(int(p['overlap'] * 100))
    
    def _on_freq_category_changed(self):
        """Handle frequency category change."""
        self._populate_freq_presets()
        
        category = self.freq_category_combo.currentData()
        self.freq_manual_widget.setVisible(category == "custom")
    
    def _on_freq_preset_changed(self):
        """Handle frequency preset change."""
        preset_id = self.freq_preset_combo.currentData()
        category = self.freq_category_combo.currentData()
        
        if category == "custom" or preset_id == "custom":
            self.freq_preset_info.setText("Set crossover frequencies manually below.")
            self.freq_manual_widget.setVisible(True)
            return
        
        # Find and display preset info
        preset = None
        for d in [FREQUENCY_PRESETS_GENRE, FREQUENCY_PRESETS_TECHNICAL, FREQUENCY_PRESETS_INSTRUMENT]:
            if preset_id in d:
                preset = d[preset_id]
                break
        
        if preset:
            self.freq_preset_info.setText(
                f"{preset['description']}\n{preset['details']}\n"
                f"Low: <{preset['low_freq']}Hz | Mid: {preset['low_freq']}-{preset['high_freq']}Hz | High: >{preset['high_freq']}Hz"
            )
            self.honest_low_freq_spin.setValue(preset['low_freq'])
            self.honest_high_freq_spin.setValue(preset['high_freq'])
        
        self.freq_manual_widget.setVisible(False)
    
    def get_mode(self):
        """Get currently selected mode."""
        for btn in self.mode_group.buttons():
            if btn.isChecked():
                return btn.mode
        return "honest_stems"
    
    def get_preset(self):
        """Get restoration preset."""
        for btn in self.preset_group.buttons():
            if btn.isChecked():
                return btn.preset_id
        return "balanced"
    
    def get_quality_preset(self):
        """Get quality preset or None for custom."""
        pid = self.quality_preset_combo.currentData()
        return None if pid == "custom" else pid
    
    def get_frequency_preset(self):
        """Get frequency preset or None for custom."""
        if self.freq_category_combo.currentData() == "custom":
            return None
        return self.freq_preset_combo.currentData()
    
    @Slot(str)
    def handle_dropped_file(self, path):
        """Handle dropped file."""
        if os.path.exists(path):
            self.file_input.setText(path)
            self.start_btn.setEnabled(True)
            self.status_label.setText(f"✓ Loaded: {os.path.basename(path)}")
    
    @Slot()
    def browse_file(self):
        """Browse for file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio", "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.ogg *.aac);;All (*.*)"
        )
        if path:
            self.file_input.setText(path)
            self.start_btn.setEnabled(True)
            self.status_label.setText(f"✓ Selected: {os.path.basename(path)}")
    
    @Slot()
    def start_separation(self):
        """Start processing."""
        path = self.file_input.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Please select a valid audio file.")
            return
        
        mode = self.get_mode()
        
        # CPU warning
        if self.device_combo.currentData() == "cpu" and mode in ["honest_stems", "full_stems"]:
            reply = QMessageBox.question(
                self, "CPU Warning",
                "Processing on CPU will be slow.\nContinue?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Output directory
        base = os.path.dirname(path)
        name = os.path.splitext(os.path.basename(path))[0]
        
        if mode == "honest_stems":
            grouping = self.honest_grouping_combo.currentData()
            out_dir = os.path.join(base, f"{name}_honest_{grouping}")
        elif mode == "full_stems":
            out_dir = os.path.join(base, f"{name}_demucs")
        elif mode == "musicgen_regen":
            out_dir = os.path.join(base, f"{name}_musicgen")
        else:
            out_dir = os.path.join(base, f"{name}_{mode}")
        
        self.status_label.setText("Processing...")
        self.set_ui_enabled(False)
        self.progress_bar.show()
        self.progress_label.setText("Initializing...")
        
        # Gather parameters
        self.worker_thread = ExtractionWorker(
            input_file=path,
            output_dir=out_dir,
            mode=mode,
            demucs_model=self.demucs_model_combo.currentData(),
            separation_model=self.sep_model_combo.currentData(),
            separation_method=self.method_combo.currentData(),
            enable_restoration=self.restore_group.isChecked() if mode == "instrument_hq" else False,
            restoration_model=self.restore_model_combo.currentData(),
            restoration_preset=self.get_preset(),
            threshold=self.sens_slider.value() / 100.0,
            max_regions=self.max_regions_spin.value(),
            shifts=self.shifts_spin.value(),
            overlap=self.overlap_spin.value() / 100.0,
            use_float32=self.float32_check.isChecked(),
            device=self.device_combo.currentData(),
            # MusicGen
            musicgen_prompt=self.mg_prompt_input.toPlainText(),
            musicgen_model=self.mg_model_combo.currentData(),
            musicgen_preprocess=self.mg_preprocess_check.isChecked(),
            musicgen_temp=self.mg_temp_slider.value() / 10.0,
            # Honest
            stem_grouping=self.honest_grouping_combo.currentData(),
            low_freq=self.honest_low_freq_spin.value(),
            high_freq=self.honest_high_freq_spin.value(),
            # Presets
            quality_preset=self.get_quality_preset(),
            frequency_preset=self.get_frequency_preset(),
        )
        
        self.worker_thread.finished.connect(self.on_finished)
        self.worker_thread.error.connect(self.on_error)
        self.worker_thread.progress.connect(self.on_progress)
        
        try:
            self.worker_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.set_ui_enabled(True)
            self.progress_bar.hide()
    
    def set_ui_enabled(self, enabled):
        """Enable/disable UI during processing."""
        for w in [
            self.browse_btn, self.demucs_model_combo, self.sep_model_combo,
            self.method_combo, self.shifts_spin, self.overlap_spin,
            self.float32_check, self.restore_model_combo, self.sens_slider,
            self.max_regions_spin, self.device_combo, self.restore_group,
            self.honest_grouping_combo, self.quality_preset_combo,
            self.freq_category_combo, self.freq_preset_combo,
            self.honest_low_freq_spin, self.honest_high_freq_spin,
            self.mg_model_combo, self.mg_preprocess_check,
            self.mg_temp_slider, self.mg_prompt_input,
        ]:
            w.setEnabled(enabled)
        
        for btn in self.mode_group.buttons():
            btn.setEnabled(enabled)
        for btn in self.preset_group.buttons():
            btn.setEnabled(enabled)
        
        self.start_btn.setEnabled(enabled and bool(self.file_input.text()))
    
    @Slot(str)
    def on_progress(self, msg):
        """Update progress."""
        self.progress_label.setText(msg)
        logger.info(f"Progress: {msg}")
    
    @Slot(str)
    def on_finished(self, out_dir):
        """Handle completion."""
        self.progress_bar.hide()
        self.progress_label.setText("")
        self.set_ui_enabled(True)
        self.status_label.setText("✅ Complete!")
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Success")
        msg.setText(f"Files saved to:\n{out_dir}")
        msg.addButton("Open Folder", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        
        msg.exec()
        if msg.clickedButton().text() == "Open Folder":
            if sys.platform == 'win32':
                os.startfile(out_dir)
            elif sys.platform == 'darwin':
                os.system(f'open "{out_dir}"')
            else:
                os.system(f'xdg-open "{out_dir}"')
    
    @Slot(str)
    def on_error(self, error_msg):
        """Handle error."""
        self.progress_bar.hide()
        self.progress_label.setText("")
        self.set_ui_enabled(True)
        self.status_label.setText("❌ Error")
        QMessageBox.critical(self, "Error", error_msg)


def main():
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = StemSeparatorApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()