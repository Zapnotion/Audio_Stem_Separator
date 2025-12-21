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




class SpectralHoleDetector:
    """
    Detects frequency-domain damage in instrumental audio.
    """
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 512
    
    def compute_spectrogram(self, audio):
        """Compute magnitude spectrogram."""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        pad_length = self.n_fft - (len(audio) % self.hop_length)
        audio_padded = np.pad(audio, (0, pad_length))
        
        num_frames = (len(audio_padded) - self.n_fft) // self.hop_length + 1
        window = np.hanning(self.n_fft)
        
        spec = np.zeros((self.n_fft // 2 + 1, num_frames))
        
        for i in range(num_frames):
            start = i * self.hop_length
            frame = audio_padded[start:start + self.n_fft] * window
            spec[:, i] = np.abs(np.fft.rfft(frame))
        
        return spec
    
    def detect_spectral_damage(self, instrumental, original, 
                                threshold=0.5, min_damage_frames=10,
                                max_duration_sec=30.0):
        """Detect regions with spectral damage."""
        logger.info("Computing spectrograms for damage detection...")
        
        if len(instrumental.shape) > 1:
            inst_mono = np.mean(instrumental, axis=1) if instrumental.shape[1] == 2 else instrumental[:, 0]
        else:
            inst_mono = instrumental
            
        if len(original.shape) > 1:
            orig_mono = np.mean(original, axis=1) if original.shape[1] == 2 else original[:, 0]
        else:
            orig_mono = original
        
        min_len = min(len(inst_mono), len(orig_mono))
        inst_mono = inst_mono[:min_len]
        orig_mono = orig_mono[:min_len]
        
        spec_inst = self.compute_spectrogram(inst_mono)
        spec_orig = self.compute_spectrogram(orig_mono)
        
        min_frames = min(spec_inst.shape[1], spec_orig.shape[1])
        spec_inst = spec_inst[:, :min_frames]
        spec_orig = spec_orig[:, :min_frames]
        
        logger.info(f"Spectrogram shape: {spec_inst.shape}")
        
        epsilon = 1e-10
        ratio = (spec_inst + epsilon) / (spec_orig + epsilon)
        
        min_energy = np.percentile(spec_orig, 10)
        damage_mask = (ratio < threshold) & (spec_orig > min_energy)
        
        damage_per_frame = np.sum(damage_mask, axis=0) / damage_mask.shape[0]
        damage_per_frame = gaussian_filter1d(damage_per_frame, sigma=3)
        
        logger.info(f"Damage per frame - min: {damage_per_frame.min():.3f}, "
                   f"max: {damage_per_frame.max():.3f}, mean: {damage_per_frame.mean():.3f}")
        
        damage_threshold = 0.05
        damaged_frames = damage_per_frame > damage_threshold
        
        regions = []
        in_region = False
        region_start = 0
        
        for i, is_damaged in enumerate(damaged_frames):
            if is_damaged and not in_region:
                region_start = i
                in_region = True
            elif not is_damaged and in_region:
                if i - region_start >= min_damage_frames:
                    freq_mask = np.mean(damage_mask[:, region_start:i], axis=1) > 0.3
                    severity = np.mean(damage_per_frame[region_start:i])
                    
                    start_sample = region_start * self.hop_length
                    end_sample = i * self.hop_length
                    duration = (end_sample - start_sample) / self.sample_rate
                    
                    if duration <= max_duration_sec:
                        regions.append((start_sample, end_sample, duration, severity, freq_mask))
                in_region = False
        
        if in_region and len(damaged_frames) - region_start >= min_damage_frames:
            freq_mask = np.mean(damage_mask[:, region_start:], axis=1) > 0.3
            severity = np.mean(damage_per_frame[region_start:])
            start_sample = region_start * self.hop_length
            end_sample = len(damaged_frames) * self.hop_length
            duration = (end_sample - start_sample) / self.sample_rate
            if duration <= max_duration_sec:
                regions.append((start_sample, end_sample, duration, severity, freq_mask))
        
        regions.sort(key=lambda r: r[3], reverse=True)
        
        logger.info(f"Found {len(regions)} damaged regions")
        for i, (start, end, dur, sev, mask) in enumerate(regions[:10]):
            damaged_bands = np.sum(mask)
            logger.info(f"  Region {i+1}: {dur:.2f}s, severity: {sev:.3f}, {damaged_bands} freq bands")
        
        return regions
    
    def get_frequency_ranges(self, freq_mask):
        """Convert frequency mask to Hz ranges."""
        freqs = np.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        damaged_freqs = freqs[freq_mask]
        
        if len(damaged_freqs) == 0:
            return None
        
        return (damaged_freqs.min(), damaged_freqs.max())



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



class VampNetRestorer:
    """Basic spectral interpolation for light damage."""
    
    PRESETS = {
        "ultra_fast": (1, 0.6),
        "fast": (2, 0.7),
        "balanced": (4, 0.8),
        "high_quality": (8, 0.9),
    }
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = 44100
        self._loaded = True
    
    def load_model(self, progress_callback=None):
        if progress_callback:
            progress_callback("✅ Spectral restoration ready")
        self._loaded = True
    
    def restore_region(self, damaged_audio, original_audio, freq_mask,
                       preset="balanced", progress_callback=None):
        iterations, strength = self.PRESETS.get(preset, self.PRESETS["balanced"])
        
        if progress_callback:
            progress_callback(f"Spectral interpolation ({iterations} iterations)...")
        
        return self._restore_with_interpolation(
            damaged_audio, freq_mask, iterations, strength, progress_callback
        )
    
    def _restore_with_interpolation(self, damaged, freq_mask, iterations, strength, progress_callback):
        n_fft = 2048
        hop = 512
        
        def stft(audio):
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            pad_len = n_fft - (len(audio) % hop)
            audio = np.pad(audio, (0, pad_len))
            
            num_frames = (len(audio) - n_fft) // hop + 1
            window = np.hanning(n_fft)
            
            result = np.zeros((n_fft // 2 + 1, num_frames), dtype=complex)
            
            for i in range(num_frames):
                s = i * hop
                frame = audio[s:s + n_fft] * window
                result[:, i] = np.fft.rfft(frame)
            
            return result
        
        def istft(spec, length):
            num_frames = spec.shape[1]
            window = np.hanning(n_fft)
            
            result = np.zeros(length + n_fft)
            window_sum = np.zeros(length + n_fft)
            
            for i in range(num_frames):
                s = i * hop
                frame = np.fft.irfft(spec[:, i])
                result[s:s + n_fft] += frame * window
                window_sum[s:s + n_fft] += window ** 2
            
            window_sum = np.maximum(window_sum, 1e-10)
            return (result / window_sum)[:length]
        
        orig_shape = damaged.shape
        orig_len = len(damaged) if len(damaged.shape) == 1 else damaged.shape[0]
        is_stereo = len(orig_shape) > 1 and orig_shape[1] == 2
        
        if is_stereo:
            channels = [damaged[:, 0], damaged[:, 1]]
        else:
            channels = [damaged.flatten()]
        
        result_channels = []
        
        for ch_idx, damaged_ch in enumerate(channels):
            dam_spec = stft(damaged_ch)
            
            if len(freq_mask) != dam_spec.shape[0]:
                freq_mask_extended = np.interp(
                    np.linspace(0, 1, dam_spec.shape[0]),
                    np.linspace(0, 1, len(freq_mask)),
                    freq_mask.astype(float)
                ) > 0.5
            else:
                freq_mask_extended = freq_mask
            
            mag = np.abs(dam_spec)
            phase = np.angle(dam_spec)
            
            result_mag = mag.copy()
            
            for iteration in range(iterations):
                new_mag = result_mag.copy()
                
                for f_idx in range(len(freq_mask_extended)):
                    if freq_mask_extended[f_idx]:
                        below_idx = None
                        above_idx = None
                        
                        for search_dist in range(1, min(50, dam_spec.shape[0])):
                            if below_idx is None and f_idx - search_dist >= 0:
                                if not freq_mask_extended[f_idx - search_dist]:
                                    below_idx = f_idx - search_dist
                            if above_idx is None and f_idx + search_dist < len(freq_mask_extended):
                                if not freq_mask_extended[f_idx + search_dist]:
                                    above_idx = f_idx + search_dist
                            if below_idx is not None and above_idx is not None:
                                break
                        
                        if below_idx is not None and above_idx is not None:
                            weight = (f_idx - below_idx) / (above_idx - below_idx)
                            interpolated = (1 - weight) * result_mag[below_idx, :] + weight * result_mag[above_idx, :]
                        elif below_idx is not None:
                            interpolated = result_mag[below_idx, :]
                        elif above_idx is not None:
                            interpolated = result_mag[above_idx, :]
                        else:
                            continue
                        
                        for t in range(dam_spec.shape[1]):
                            t_neighbors = []
                            for dt in [-2, -1, 1, 2]:
                                if 0 <= t + dt < dam_spec.shape[1]:
                                    t_neighbors.append(result_mag[f_idx, t + dt])
                            
                            if t_neighbors:
                                temporal_avg = np.mean(t_neighbors)
                                interpolated[t] = 0.7 * interpolated[t] + 0.3 * temporal_avg
                        
                        current_strength = strength * (iteration + 1) / iterations
                        new_mag[f_idx, :] = (1 - current_strength) * result_mag[f_idx, :] + current_strength * interpolated
                
                result_mag = new_mag
                
                if iteration < iterations - 1:
                    result_mag = gaussian_filter(result_mag, sigma=0.3)
            
            result_spec = result_mag * np.exp(1j * phase)
            result_ch = istft(result_spec, orig_len)
            result_channels.append(result_ch)
        
        if is_stereo:
            result = np.stack(result_channels, axis=1)
        else:
            result = result_channels[0]
        
        return result.astype(np.float32)
    
    def cleanup(self):
        pass



class AudioSRRestorer:
    """Uses AudioSR for audio super-resolution with improved blending."""
    
    PRESETS = {
        "ultra_fast": (10, 3.5),
        "fast": (25, 3.5),
        "balanced": (50, 3.5),
        "high_quality": (100, 3.5),
    }
    
    # Conservative limits to prevent freezing/crashes
    MAX_CHUNK_DURATION = 5.0      # seconds - AudioSR's sweet spot
    MIN_CHUNK_DURATION = 1.0      # minimum processable length
    OVERLAP_RATIO = 0.15          # 15% overlap between chunks
    MAX_TOTAL_DURATION = 300.0    # 5 minutes max to prevent memory issues
    
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._loaded = False
        self.output_sr = 48000    # AudioSR outputs 48kHz
        self.input_sr = 44100
    
    def load_model(self, progress_callback=None):
        if self._loaded:
            return
        
        if progress_callback:
            progress_callback("Loading AudioSR model...")
        
        try:
            from audiosr import build_model, super_resolution
            
            self.super_resolution_fn = super_resolution
            self.model = build_model(model_name="basic", device=self.device)
            self._loaded = True
            
            if progress_callback:
                progress_callback("✅ AudioSR model loaded")
                
        except Exception as e:
            logger.error(f"Failed to load AudioSR: {e}")
            raise
    
    def _create_crossfade_weights(self, length, fade_samples):
        """Create smooth crossfade weight array."""
        weights = np.ones(length, dtype=np.float32)
        
        if fade_samples > 0 and fade_samples * 2 < length:
            # Smooth fade using raised cosine (Hann-like)
            fade_in = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
            fade_out = 0.5 * (1 + np.cos(np.pi * np.arange(fade_samples) / fade_samples))
            
            weights[:fade_samples] = fade_in
            weights[-fade_samples:] = fade_out
        
        return weights
    
    def _estimate_safe_chunk_duration(self, total_samples, sample_rate):
        """Estimate safe chunk duration based on available memory."""
        total_duration = total_samples / sample_rate
        
        # Check available GPU memory if using CUDA
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                free_memory = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated(0)
                available_gb = (free_memory - allocated) / (1024**3)
                
                # Rough estimate: AudioSR needs ~2GB per 5 seconds at 48kHz
                if available_gb < 4:
                    return min(3.0, self.MAX_CHUNK_DURATION)
                elif available_gb < 6:
                    return min(4.0, self.MAX_CHUNK_DURATION)
                else:
                    return self.MAX_CHUNK_DURATION
            except:
                pass
        
        return self.MAX_CHUNK_DURATION
    
    def _process_single_chunk(self, chunk, ddim_steps, guidance_scale, chunk_idx, total_chunks, progress_callback):
        """Process a single audio chunk through AudioSR with error handling."""
        temp_input_path = None
        original_len = len(chunk)
        
        try:
            # Ensure minimum length for AudioSR
            min_samples = int(self.MIN_CHUNK_DURATION * self.input_sr)
            
            if len(chunk) < min_samples:
                # Pad with reflection to avoid edge artifacts
                pad_needed = min_samples - len(chunk)
                chunk = np.pad(chunk, (0, pad_needed), mode='reflect')
            
            # Normalize chunk to prevent clipping issues
            chunk_max = np.max(np.abs(chunk))
            if chunk_max > 0:
                chunk_normalized = chunk / chunk_max
            else:
                chunk_normalized = chunk
            
            # Write temporary file
            temp_input_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_input_path, chunk_normalized, self.input_sr)
            
            # Clear GPU memory before processing
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            if progress_callback:
                progress_callback(f"AudioSR processing chunk {chunk_idx + 1}/{total_chunks}...")
            
            # Run AudioSR
            restored = self.super_resolution_fn(
                self.model,
                temp_input_path,
                seed=42,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                latent_t_per_second=12.8
            )
            
            # Convert output to numpy
            if torch.is_tensor(restored):
                restored = restored.cpu().numpy()
            
            # Handle various output shapes from AudioSR
            if len(restored.shape) == 3:
                restored = restored[0, 0, :]  # [batch, channel, samples]
            elif len(restored.shape) == 2:
                if restored.shape[0] == 1:
                    restored = restored[0, :]  # [batch, samples]
                else:
                    restored = restored[0, :]  # [channels, samples] - take first
            
            # Resample from 48kHz back to input sample rate
            if self.output_sr != self.input_sr:
                target_samples = int(len(restored) * self.input_sr / self.output_sr)
                restored = signal.resample(restored, target_samples)
            
            # Restore original amplitude
            if chunk_max > 0:
                restored = restored * chunk_max
            
            # Trim to original length (remove padding if added)
            if len(restored) > original_len:
                restored = restored[:original_len]
            elif len(restored) < original_len:
                # Shouldn't happen often, but handle gracefully
                restored = np.pad(restored, (0, original_len - len(restored)), mode='edge')
            
            return restored, True
            
        except torch.cuda.OutOfMemoryError:
            logger.warning(f"CUDA OOM on chunk {chunk_idx + 1}, returning original")
            torch.cuda.empty_cache()
            return chunk[:original_len], False
            
        except Exception as e:
            logger.warning(f"AudioSR chunk {chunk_idx + 1} failed: {e}")
            return chunk[:original_len], False
            
        finally:
            # Cleanup
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.unlink(temp_input_path)
                except:
                    pass
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
    
    def restore_region(self, damaged_audio, original_audio, freq_mask,
                       preset="balanced", progress_callback=None):
        """Restore damaged region using AudioSR with proper chunking and blending."""
        if not self._loaded:
            self.load_model(progress_callback)
        
        ddim_steps, guidance_scale = self.PRESETS.get(preset, self.PRESETS["balanced"])
        
        # Get audio dimensions
        orig_shape = damaged_audio.shape
        orig_len = damaged_audio.shape[0] if len(damaged_audio.shape) > 1 else len(damaged_audio)
        is_stereo = len(orig_shape) > 1 and orig_shape[1] == 2
        
        # Check duration limit
        duration = orig_len / self.input_sr
        if duration > self.MAX_TOTAL_DURATION:
            if progress_callback:
                progress_callback(f"⚠️ Audio too long ({duration:.1f}s), truncating to {self.MAX_TOTAL_DURATION}s")
            orig_len = int(self.MAX_TOTAL_DURATION * self.input_sr)
            if is_stereo:
                damaged_audio = damaged_audio[:orig_len, :]
            else:
                damaged_audio = damaged_audio[:orig_len]
        
        if progress_callback:
            progress_callback(f"AudioSR restoration ({ddim_steps} steps, {duration:.1f}s audio)...")
        
        # Convert to mono for processing
        if is_stereo:
            # Store original stereo info for later reconstruction
            left_channel = damaged_audio[:, 0].copy()
            right_channel = damaged_audio[:, 1].copy()
            audio_mono = (left_channel + right_channel) / 2
            
            # Calculate stereo difference for reconstruction
            stereo_diff = (left_channel - right_channel) / 2
        else:
            audio_mono = damaged_audio.flatten()
            stereo_diff = None
        
        # Determine chunk parameters
        chunk_duration = self._estimate_safe_chunk_duration(len(audio_mono), self.input_sr)
        chunk_samples = int(chunk_duration * self.input_sr)
        overlap_samples = int(chunk_samples * self.OVERLAP_RATIO)
        hop_samples = chunk_samples - overlap_samples
        
        # Ensure valid parameters
        if hop_samples <= 0:
            hop_samples = chunk_samples // 2
            overlap_samples = chunk_samples - hop_samples
        
        total_len = len(audio_mono)
        
        # Calculate number of chunks
        if total_len <= chunk_samples:
            num_chunks = 1
        else:
            num_chunks = int(np.ceil((total_len - overlap_samples) / hop_samples))
        
        if progress_callback:
            progress_callback(f"Processing {num_chunks} chunks ({chunk_duration:.1f}s each, {overlap_samples/self.input_sr:.2f}s overlap)...")
        
        # Initialize output buffers with overlap-add
        output = np.zeros(total_len, dtype=np.float64)
        weight_sum = np.zeros(total_len, dtype=np.float64)
        
        successful_chunks = 0
        
        for i in range(num_chunks):
            # Calculate chunk boundaries
            start = i * hop_samples
            end = min(start + chunk_samples, total_len)
            
            # For last chunk, adjust to ensure full coverage
            if i == num_chunks - 1 and end < total_len:
                end = total_len
                start = max(0, end - chunk_samples)
            
            chunk = audio_mono[start:end].copy()
            actual_len = len(chunk)
            
            # Process chunk
            restored_chunk, success = self._process_single_chunk(
                chunk, ddim_steps, guidance_scale, 
                i, num_chunks, progress_callback
            )
            
            if success:
                successful_chunks += 1
            
            # Ensure correct length
            if len(restored_chunk) != actual_len:
                if len(restored_chunk) > actual_len:
                    restored_chunk = restored_chunk[:actual_len]
                else:
                    restored_chunk = np.pad(restored_chunk, (0, actual_len - len(restored_chunk)), mode='edge')
            
            # Create weights for this chunk (with crossfade)
            weights = self._create_crossfade_weights(actual_len, overlap_samples)
            
            # First chunk: no fade-in needed
            if i == 0:
                weights[:overlap_samples] = 1.0
            
            # Last chunk: no fade-out needed
            if i == num_chunks - 1:
                weights[-overlap_samples:] = 1.0
            
            # Accumulate with overlap-add
            output[start:start + actual_len] += restored_chunk * weights
            weight_sum[start:start + actual_len] += weights
        
        # Normalize by weights
        weight_sum = np.maximum(weight_sum, 1e-8)
        output = (output / weight_sum).astype(np.float32)
        
        if progress_callback:
            progress_callback(f"AudioSR: {successful_chunks}/{num_chunks} chunks processed successfully")
        
        # Frequency-selective blending with original
        output = self._frequency_selective_blend(
            original=audio_mono,
            restored=output,
            freq_mask=freq_mask,
            blend_strength=0.85,
            progress_callback=progress_callback
        )
        
        # Restore stereo if needed
        if is_stereo:
            output = self._reconstruct_stereo(output, stereo_diff, left_channel, right_channel, orig_len)
        
        # Final length adjustment
        if len(output) > orig_len:
            output = output[:orig_len] if not is_stereo else output[:orig_len, :]
        elif len(output) < orig_len:
            if is_stereo:
                pad_len = orig_len - output.shape[0]
                output = np.pad(output, ((0, pad_len), (0, 0)), mode='edge')
            else:
                output = np.pad(output, (0, orig_len - len(output)), mode='edge')
        
        if progress_callback:
            progress_callback("✅ AudioSR restoration complete")
        
        return output.astype(np.float32)
    
    def _frequency_selective_blend(self, original, restored, freq_mask, blend_strength=0.85, progress_callback=None):
        """
        Blend restored audio with original, only replacing damaged frequencies.
        This prevents AudioSR from changing frequencies that were already good.
        """
        if progress_callback:
            progress_callback("Frequency-selective blending...")
        
        n_fft = 2048
        hop_length = 512
        
        min_len = min(len(original), len(restored))
        original = original[:min_len]
        restored = restored[:min_len]
        
        # Compute STFTs
        orig_spec = self._stft(original, n_fft, hop_length)
        rest_spec = self._stft(restored, n_fft, hop_length)
        
        # Ensure same shape
        min_frames = min(orig_spec.shape[1], rest_spec.shape[1])
        orig_spec = orig_spec[:, :min_frames]
        rest_spec = rest_spec[:, :min_frames]
        
        num_bins = orig_spec.shape[0]
        
        # Resize frequency mask to match spectrogram bins
        if len(freq_mask) != num_bins:
            freq_mask_resized = np.interp(
                np.linspace(0, 1, num_bins),
                np.linspace(0, 1, len(freq_mask)),
                freq_mask.astype(float)
            )
        else:
            freq_mask_resized = freq_mask.astype(float)
        
        # Smooth the mask to avoid harsh transitions between bands
        freq_mask_smooth = gaussian_filter1d(freq_mask_resized, sigma=5)
        
        # Create 2D blend mask: damaged frequencies use restored, others use original
        blend_mask = freq_mask_smooth[:, np.newaxis] * blend_strength
        
        # Add slight temporal smoothing to the mask
        blend_mask = gaussian_filter(blend_mask, sigma=(2, 3))
        
        # Blend magnitude and phase separately for better results
        orig_mag = np.abs(orig_spec)
        rest_mag = np.abs(rest_spec)
        orig_phase = np.angle(orig_spec)
        rest_phase = np.angle(rest_spec)
        
        # Blend magnitudes
        blended_mag = (1 - blend_mask) * orig_mag + blend_mask * rest_mag
        
        # For phase, prefer original in undamaged regions, blend in damaged
        # Phase blending is tricky - use weighted combination
        phase_blend_mask = blend_mask * 0.7  # Less aggressive phase blending
        
        # Simple phase blending (complex domain would be better but more complex)
        blended_phase = orig_phase.copy()
        high_damage_mask = blend_mask > 0.5
        blended_phase[high_damage_mask] = rest_phase[high_damage_mask]
        
        # Reconstruct complex spectrogram
        blended_spec = blended_mag * np.exp(1j * blended_phase)
        
        # ISTFT
        result = self._istft(blended_spec, min_len, n_fft, hop_length)
        
        return result.astype(np.float32)
    
    def _reconstruct_stereo(self, mono_restored, stereo_diff, orig_left, orig_right, target_len):
        """
        Reconstruct stereo from restored mono using mid-side technique.
        Preserves the original stereo field while applying restoration.
        """
        # Ensure all arrays are the right length
        mono_restored = mono_restored[:target_len]
        
        if stereo_diff is not None:
            stereo_diff = stereo_diff[:target_len]
            
            # Scale stereo difference to match restored mid channel energy
            orig_mid = (orig_left[:target_len] + orig_right[:target_len]) / 2
            
            # Calculate RMS ratio for scaling
            orig_rms = np.sqrt(np.mean(orig_mid ** 2) + 1e-8)
            rest_rms = np.sqrt(np.mean(mono_restored ** 2) + 1e-8)
            
            scale = rest_rms / orig_rms if orig_rms > 1e-8 else 1.0
            scale = np.clip(scale, 0.5, 2.0)  # Limit scaling to reasonable range
            
            scaled_diff = stereo_diff * scale
            
            # Reconstruct left and right
            new_left = mono_restored + scaled_diff
            new_right = mono_restored - scaled_diff
        else:
            # No stereo info available, duplicate mono
            new_left = mono_restored
            new_right = mono_restored
        
        # Ensure correct length
        if len(new_left) < target_len:
            new_left = np.pad(new_left, (0, target_len - len(new_left)), mode='edge')
            new_right = np.pad(new_right, (0, target_len - len(new_right)), mode='edge')
        
        return np.stack([new_left[:target_len], new_right[:target_len]], axis=1)
    
    def _stft(self, audio, n_fft, hop_length):
        """Compute STFT with proper padding."""
        # Pad for centered STFT
        pad_len = n_fft // 2
        audio_padded = np.pad(audio, (pad_len, pad_len), mode='reflect')
        
        num_frames = (len(audio_padded) - n_fft) // hop_length + 1
        window = np.hanning(n_fft)
        
        spec = np.zeros((n_fft // 2 + 1, num_frames), dtype=complex)
        
        for i in range(num_frames):
            start = i * hop_length
            frame = audio_padded[start:start + n_fft] * window
            spec[:, i] = np.fft.rfft(frame)
        
        return spec
    
    def _istft(self, spec, target_length, n_fft, hop_length):
        """Compute ISTFT with overlap-add."""
        num_frames = spec.shape[1]
        window = np.hanning(n_fft)
        
        # Output buffer
        pad_len = n_fft // 2
        output_length = (num_frames - 1) * hop_length + n_fft
        output = np.zeros(output_length, dtype=np.float64)
        window_sum = np.zeros(output_length, dtype=np.float64)
        
        for i in range(num_frames):
            start = i * hop_length
            frame = np.fft.irfft(spec[:, i])
            
            # Ensure frame is correct length
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            elif len(frame) > n_fft:
                frame = frame[:n_fft]
            
            output[start:start + n_fft] += frame * window
            window_sum[start:start + n_fft] += window ** 2
        
        # Normalize
        window_sum = np.maximum(window_sum, 1e-10)
        output = output / window_sum
        
        # Remove padding
        output = output[pad_len:]
        
        # Adjust to target length
        if len(output) > target_length:
            output = output[:target_length]
        elif len(output) < target_length:
            output = np.pad(output, (0, target_length - len(output)), mode='edge')
        
        return output.astype(np.float32)
    
    def cleanup(self):
        """Release GPU memory and cleanup."""
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()




class EnhancedSpectralRestorer:
    """Advanced spectral interpolation with harmonic awareness."""
    
    PRESETS = {
        "ultra_fast": {"iterations": 1, "harmonic_search": False, "multi_res": False},
        "fast": {"iterations": 2, "harmonic_search": True, "multi_res": False},
        "balanced": {"iterations": 4, "harmonic_search": True, "multi_res": True},
        "high_quality": {"iterations": 8, "harmonic_search": True, "multi_res": True},
    }
    
    def __init__(self, device=None):
        self.device = device
        self.sample_rate = 44100
        self._loaded = True
    
    def load_model(self, progress_callback=None):
        if progress_callback:
            progress_callback("✅ Enhanced Spectral Restorer ready")
        self._loaded = True
    
    def restore_region(self, damaged_audio, original_audio, freq_mask,
                       preset="balanced", progress_callback=None):
        settings = self.PRESETS.get(preset, self.PRESETS["balanced"])
        
        if progress_callback:
            progress_callback(f"Enhanced spectral restoration ({settings['iterations']} iterations)...")
        
        orig_shape = damaged_audio.shape
        orig_len = damaged_audio.shape[0] if len(damaged_audio.shape) > 1 else len(damaged_audio)
        is_stereo = len(orig_shape) > 1 and orig_shape[1] == 2
        
        if is_stereo:
            left = self._restore_channel(damaged_audio[:, 0], freq_mask, settings, progress_callback, "L")
            right = self._restore_channel(damaged_audio[:, 1], freq_mask, settings, progress_callback, "R")
            result = np.stack([left, right], axis=1)
        else:
            mono = damaged_audio.flatten() if len(damaged_audio.shape) > 1 else damaged_audio
            result = self._restore_channel(mono, freq_mask, settings, progress_callback, "")
        
        if len(result) > orig_len:
            result = result[:orig_len]
        elif len(result) < orig_len:
            result = np.pad(result, ((0, orig_len - len(result)), (0, 0)) if is_stereo else (0, orig_len - len(result)))
        
        return result.astype(np.float32)
    
    def _restore_channel(self, audio, freq_mask, settings, progress_callback, channel_name):
        if settings["multi_res"]:
            results = []
            weights = []
            
            for n_fft in [1024, 2048, 4096]:
                restored = self._process_single_resolution(audio, freq_mask, n_fft, settings)
                results.append(restored)
                weights.append(n_fft / 4096)
            
            min_len = min(len(r) for r in results)
            results = [r[:min_len] for r in results]
            weights = np.array(weights) / sum(weights)
            
            final = sum(w * r for w, r in zip(weights, results))
        else:
            final = self._process_single_resolution(audio, freq_mask, 2048, settings)
        
        return final
    
    def _process_single_resolution(self, audio, freq_mask, n_fft, settings):
        hop_length = n_fft // 4
        
        spec = self._stft(audio, n_fft, hop_length)
        mag = np.abs(spec)
        phase = np.angle(spec)
        
        target_bins = spec.shape[0]
        if len(freq_mask) != target_bins:
            freq_mask_resized = np.interp(
                np.linspace(0, 1, target_bins),
                np.linspace(0, 1, len(freq_mask)),
                freq_mask.astype(float)
            ) > 0.5
        else:
            freq_mask_resized = freq_mask.copy()
        
        damage_mask_2d = np.tile(freq_mask_resized[:, np.newaxis], (1, spec.shape[1]))
        
        restored_mag = mag.copy()
        
        for iteration in range(settings["iterations"]):
            if settings["harmonic_search"]:
                restored_mag = self._reconstruct_harmonics(restored_mag, damage_mask_2d, n_fft)
            
            restored_mag = self._interpolate_frequencies(restored_mag, damage_mask_2d)
            restored_mag = self._temporal_smooth(restored_mag, damage_mask_2d, mag)
        
        restored_spec = restored_mag * np.exp(1j * phase)
        restored_audio = self._istft(restored_spec, len(audio), n_fft, hop_length)
        
        return restored_audio
    
    def _stft(self, audio, n_fft, hop_length):
        pad_len = n_fft - (len(audio) % hop_length)
        audio_padded = np.pad(audio, (n_fft // 2, pad_len + n_fft // 2))
        
        num_frames = (len(audio_padded) - n_fft) // hop_length + 1
        window = np.hanning(n_fft)
        
        spec = np.zeros((n_fft // 2 + 1, num_frames), dtype=complex)
        
        for i in range(num_frames):
            start = i * hop_length
            frame = audio_padded[start:start + n_fft] * window
            spec[:, i] = np.fft.rfft(frame)
        
        return spec
    
    def _istft(self, spec, target_length, n_fft, hop_length):
        num_frames = spec.shape[1]
        window = np.hanning(n_fft)
        
        output_length = (num_frames - 1) * hop_length + n_fft
        output = np.zeros(output_length)
        window_sum = np.zeros(output_length)
        
        for i in range(num_frames):
            start = i * hop_length
            frame = np.fft.irfft(spec[:, i])
            output[start:start + n_fft] += frame * window
            window_sum[start:start + n_fft] += window ** 2
        
        window_sum = np.maximum(window_sum, 1e-10)
        output = output / window_sum
        
        output = output[n_fft // 2:]
        if len(output) > target_length:
            output = output[:target_length]
        elif len(output) < target_length:
            output = np.pad(output, (0, target_length - len(output)))
        
        return output
    
    def _reconstruct_harmonics(self, mag, damage_mask, n_fft):
        result = mag.copy()
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
        
        undamaged_mask = ~damage_mask
        mean_energy = np.mean(mag, axis=1)
        
        threshold = np.percentile(mean_energy[undamaged_mask[:, 0]], 70)
        fundamental_candidates = np.where(
            (undamaged_mask[:, 0]) & (mean_energy > threshold)
        )[0]
        
        for fund_bin in fundamental_candidates:
            fund_freq = freqs[fund_bin]
            if fund_freq < 50:
                continue
            
            for harmonic_num in range(2, 6):
                harmonic_freq = fund_freq * harmonic_num
                harmonic_bin = int(harmonic_freq * n_fft / self.sample_rate)
                
                if harmonic_bin >= len(freqs):
                    break
                
                if damage_mask[harmonic_bin, 0]:
                    rolloff = 1.0 / (harmonic_num ** 0.8)
                    estimated_mag = mag[fund_bin, :] * rolloff
                    
                    blend_factor = 0.7
                    result[harmonic_bin, :] = (
                        blend_factor * estimated_mag +
                        (1 - blend_factor) * result[harmonic_bin, :]
                    )
        
        return result
    
    def _interpolate_frequencies(self, mag, damage_mask):
        result = mag.copy()
        num_freqs, num_frames = mag.shape
        
        for f in range(num_freqs):
            if not damage_mask[f, 0]:
                continue
            
            below = above = None
            
            for dist in range(1, min(100, num_freqs)):
                if below is None and f - dist >= 0 and not damage_mask[f - dist, 0]:
                    below = f - dist
                if above is None and f + dist < num_freqs and not damage_mask[f + dist, 0]:
                    above = f + dist
                if below is not None and above is not None:
                    break
            
            if below is not None and above is not None:
                dist_below = f - below
                dist_above = above - f
                total_dist = dist_below + dist_above
                
                weight_below = 1 - (dist_below / total_dist)
                weight_above = 1 - (dist_above / total_dist)
                
                weight_sum = weight_below + weight_above
                weight_below /= weight_sum
                weight_above /= weight_sum
                
                result[f, :] = weight_below * mag[below, :] + weight_above * mag[above, :]
                
            elif below is not None:
                dist = f - below
                decay = np.exp(-0.1 * dist)
                result[f, :] = mag[below, :] * decay
                
            elif above is not None:
                dist = above - f
                decay = np.exp(-0.1 * dist)
                result[f, :] = mag[above, :] * decay
        
        return result
    
    def _temporal_smooth(self, mag, damage_mask, original_mag):
        result = mag.copy()
        
        smoothed = gaussian_filter(mag, sigma=(1, 2))
        
        for f in range(mag.shape[0]):
            if damage_mask[f, 0]:
                result[f, :] = 0.7 * smoothed[f, :] + 0.3 * result[f, :]
        
        boundary_smooth = gaussian_filter1d(result, sigma=1, axis=1)
        
        for f in range(mag.shape[0]):
            if damage_mask[f, 0]:
                if f > 0 and not damage_mask[f - 1, 0]:
                    result[f, :] = 0.5 * boundary_smooth[f, :] + 0.5 * result[f, :]
                if f < mag.shape[0] - 1 and not damage_mask[f + 1, 0]:
                    result[f, :] = 0.5 * boundary_smooth[f, :] + 0.5 * result[f, :]
        
        return result
    
    def cleanup(self):
        pass



class AudioInpaintingRestorer:
    """
    Diffusion-based audio inpainting using Eloimoliner's model.
    Paper: "Diffusion-Based Audio Inpainting" (JAES, March 2024)
    """
    
    PRESETS = {
        "ultra_fast": {"steps": 25, "noise_level": 0.5},
        "fast": {"steps": 50, "noise_level": 0.5},
        "balanced": {"steps": 100, "noise_level": 0.5},
        "high_quality": {"steps": 200, "noise_level": 0.5},
    }
    
    MODELS = {
        "musicnet_44k": {
            "filename": "musicnet_44k_4s-560000.pt",
            "sample_rate": 44100,
            "segment_length": 4.0,
            "description": "General music (44.1kHz)"
        },
        "maestro_22k": {
            "filename": "maestro_22k_8s-750000.pt",
            "sample_rate": 22050,
            "segment_length": 8.0,
            "description": "Piano/Classical (22kHz)"
        }
    }
    
    def __init__(self, device=None, model_type="musicnet_44k"):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model_config = self.MODELS[model_type]
        
        self.network = None
        self.diff_params = None
        self._loaded = False
        self._fallback_restorer = None
        
        self.repo_path = INPAINTING_REPO_PATH
        self.models_dir = INPAINTING_MODELS_DIR
        self.model_path = os.path.join(self.models_dir, self.model_config["filename"])
    
    def load_model(self, progress_callback=None):
        """Load the diffusion inpainting model."""
        if self._loaded:
            return
        
        # Check if repo and model exist
        if not os.path.exists(self.repo_path):
            if progress_callback:
                progress_callback(f"⚠️ Inpainting repo not found, using fallback")
            self._setup_fallback(progress_callback)
            return
        
        if not os.path.exists(self.model_path):
            if progress_callback:
                progress_callback(f"⚠️ Model not found at {self.model_path}, using fallback")
            self._setup_fallback(progress_callback)
            return
        
        try:
            if progress_callback:
                progress_callback("Loading diffusion inpainting model...")
            
            # Add repo to path
            if self.repo_path not in sys.path:
                sys.path.insert(0, self.repo_path)
            
            # Load checkpoint
            if progress_callback:
                progress_callback(f"Loading checkpoint: {self.model_config['filename']}...")
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Build network from checkpoint
            self.network = self._build_network(checkpoint)
            self.network.to(self.device)
            self.network.eval()
            
            # Setup diffusion parameters
            self.diff_params = self._setup_diff_params()
            
            self._loaded = True
            
            if progress_callback:
                progress_callback("✅ Diffusion inpainting model loaded")
                
        except Exception as e:
            logger.error(f"Failed to load inpainting model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            if progress_callback:
                progress_callback(f"⚠️ Load failed: {e}, using fallback")
            
            self._setup_fallback(progress_callback)
    
    def _build_network(self, checkpoint):
        """Build network from checkpoint."""
        try:
            # Try importing from repo
            from networks.unet_cqt import UNet
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'ema' in checkpoint:
                state_dict = checkpoint['ema']
            else:
                state_dict = checkpoint
            
            # Try to infer architecture from state dict
            # This is simplified - may need adjustment based on actual model
            network = UNet(
                in_channels=1,
                out_channels=1,
            )
            network.load_state_dict(state_dict, strict=False)
            return network
            
        except Exception as e:
            logger.warning(f"Could not build UNet from repo: {e}")
            
            # Try direct load
            if isinstance(checkpoint, torch.nn.Module):
                return checkpoint
            elif 'ema' in checkpoint and isinstance(checkpoint['ema'], torch.nn.Module):
                return checkpoint['ema']
            elif 'model' in checkpoint and isinstance(checkpoint['model'], torch.nn.Module):
                return checkpoint['model']
            else:
                raise Exception(f"Cannot extract model from checkpoint: {type(checkpoint)}")
    
    def _setup_diff_params(self):
        """Setup diffusion parameters."""
        try:
            from diff_params.edm import EDM
            return EDM()
        except ImportError:
            # Minimal diff params
            class SimpleDiffParams:
                def __init__(self):
                    self.sigma_min = 0.002
                    self.sigma_max = 80
                    self.rho = 7
                
                def get_sigmas(self, num_steps):
                    ramp = torch.linspace(0, 1, num_steps)
                    min_inv_rho = self.sigma_min ** (1 / self.rho)
                    max_inv_rho = self.sigma_max ** (1 / self.rho)
                    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
                    return sigmas
            
            return SimpleDiffParams()
    
    def _setup_fallback(self, progress_callback=None):
        """Setup fallback to EnhancedSpectralRestorer."""
        self._fallback_restorer = EnhancedSpectralRestorer(self.device)
        self._fallback_restorer.load_model(progress_callback)
        self._loaded = True
    
    def restore_region(self, damaged_audio, original_audio, freq_mask,
                       preset="balanced", progress_callback=None):
        """Restore damaged region using diffusion inpainting."""
        if not self._loaded:
            self.load_model(progress_callback)
        
        if self._fallback_restorer is not None:
            return self._fallback_restorer.restore_region(
                damaged_audio, original_audio, freq_mask, preset, progress_callback
            )
        
        settings = self.PRESETS.get(preset, self.PRESETS["balanced"])
        
        if progress_callback:
            progress_callback(f"Diffusion inpainting ({settings['steps']} steps)...")
        
        try:
            return self._inpaint(damaged_audio, freq_mask, settings, progress_callback)
        except Exception as e:
            logger.error(f"Diffusion inpainting failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            if progress_callback:
                progress_callback(f"⚠️ Inpainting failed, using spectral fallback...")
            
            if self._fallback_restorer is None:
                self._fallback_restorer = EnhancedSpectralRestorer(self.device)
                self._fallback_restorer.load_model(progress_callback)
            
            return self._fallback_restorer.restore_region(
                damaged_audio, original_audio, freq_mask, preset, progress_callback
            )
    
    def _inpaint(self, damaged_audio, freq_mask, settings, progress_callback):
        """Core diffusion inpainting logic."""
        orig_shape = damaged_audio.shape
        orig_len = damaged_audio.shape[0] if len(damaged_audio.shape) > 1 else len(damaged_audio)
        is_stereo = len(orig_shape) > 1 and orig_shape[1] == 2
        
        model_sr = self.model_config["sample_rate"]
        segment_len = self.model_config["segment_length"]
        segment_samples = int(segment_len * model_sr)
        
        # Convert to mono
        if is_stereo:
            audio_mono = np.mean(damaged_audio, axis=1)
        else:
            audio_mono = damaged_audio.flatten()
        
        # Resample if needed
        input_sr = 44100
        if input_sr != model_sr:
            audio_mono = signal.resample(audio_mono, int(len(audio_mono) * model_sr / input_sr))
        
        # Normalize
        audio_mono = audio_mono.astype(np.float32)
        max_val = np.max(np.abs(audio_mono))
        if max_val > 0:
            audio_mono = audio_mono / max_val
        
        # Create inpainting mask
        inpaint_mask = self._create_inpaint_mask(audio_mono, freq_mask, model_sr)
        
        # Process in segments
        num_segments = int(np.ceil(len(audio_mono) / segment_samples))
        results = []
        
        for i in range(num_segments):
            start = i * segment_samples
            end = min(start + segment_samples, len(audio_mono))
            
            segment = audio_mono[start:end]
            mask_segment = inpaint_mask[start:end]
            
            # Pad if needed
            if len(segment) < segment_samples:
                pad_len = segment_samples - len(segment)
                segment = np.pad(segment, (0, pad_len))
                mask_segment = np.pad(mask_segment, (0, pad_len))
            
            if progress_callback:
                progress_callback(f"Inpainting segment {i+1}/{num_segments}...")
            
            # Run diffusion
            restored_segment = self._inpaint_segment(segment, mask_segment, settings, progress_callback)
            
            # Trim padding
            if end - start < segment_samples:
                restored_segment = restored_segment[:end - start]
            
            results.append(restored_segment)
            torch.cuda.empty_cache()
        
        # Concatenate
        restored = np.concatenate(results)
        
        # Resample back
        if input_sr != model_sr:
            restored = signal.resample(restored, int(len(restored) * input_sr / model_sr))
        
        # Restore amplitude
        if max_val > 0:
            restored = restored * max_val
        
        # Match length
        if len(restored) > orig_len:
            restored = restored[:orig_len]
        elif len(restored) < orig_len:
            restored = np.pad(restored, (0, orig_len - len(restored)))
        
        # Restore stereo
        if is_stereo:
            restored = np.stack([restored, restored], axis=1)
        
        if progress_callback:
            progress_callback("✅ Diffusion inpainting complete")
        
        return restored.astype(np.float32)
    
    def _create_inpaint_mask(self, audio, freq_mask, sample_rate):
        """Create time-domain inpainting mask from frequency mask."""
        n_fft = 2048
        hop_length = 512
        
        # Compute STFT
        audio_tensor = torch.from_numpy(audio).float()
        spec = torch.stft(
            audio_tensor,
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True,
            window=torch.hann_window(n_fft)
        )
        mag = torch.abs(spec)
        
        # Resize freq_mask
        num_bins = mag.shape[0]
        if len(freq_mask) != num_bins:
            freq_mask_resized = np.interp(
                np.linspace(0, 1, num_bins),
                np.linspace(0, 1, len(freq_mask)),
                freq_mask.astype(float)
            )
        else:
            freq_mask_resized = freq_mask.astype(float)
        
        # Create frame-level damage mask
        damage_per_frame = np.zeros(mag.shape[1])
        
        for f in range(num_bins):
            if freq_mask_resized[f] > 0.5:
                damage_per_frame += mag[f, :].numpy() * freq_mask_resized[f]
        
        if np.max(damage_per_frame) > 0:
            damage_per_frame = damage_per_frame / np.max(damage_per_frame)
        
        frame_mask = (damage_per_frame > 0.2).astype(float)
        frame_mask = gaussian_filter1d(frame_mask, sigma=3)
        
        # Expand to sample-level
        sample_mask = np.zeros(len(audio))
        
        for i, val in enumerate(frame_mask):
            start = i * hop_length
            end = min(start + n_fft, len(audio))
            sample_mask[start:end] = np.maximum(sample_mask[start:end], val)
        
        return sample_mask
    
    def _inpaint_segment(self, segment, mask, settings, progress_callback):
        """Run diffusion inpainting on a single segment."""
        num_steps = settings['steps']
        
        # Convert to tensor
        x = torch.from_numpy(segment).float().unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)
        
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_t = mask_t.to(self.device)
        
        # Get sigma schedule
        sigmas = self.diff_params.get_sigmas(num_steps).to(self.device)
        
        # Initialize with noise in masked regions
        noise = torch.randn_like(x)
        x_noisy = x * (1 - mask_t) + noise * sigmas[0] * mask_t
        
        # Diffusion loop
        with torch.no_grad():
            for i in range(num_steps - 1):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                
                # Denoise
                denoised = self.network(x_noisy, sigma.unsqueeze(0))
                
                # Apply mask
                denoised = x * (1 - mask_t) + denoised * mask_t
                
                # Add noise for next step
                if sigma_next > 0:
                    noise = torch.randn_like(x)
                    x_noisy = denoised + noise * sigma_next
                else:
                    x_noisy = denoised
                
                x_noisy = x * (1 - mask_t) + x_noisy * mask_t
        
        result = x_noisy.squeeze().cpu().numpy()
        return result
    
    def cleanup(self):
        """Release GPU memory."""
        if self.network is not None:
            del self.network
            self.network = None
        
        if self.diff_params is not None:
            del self.diff_params
            self.diff_params = None
        
        if self._fallback_restorer is not None:
            self._fallback_restorer.cleanup()
            self._fallback_restorer = None
        
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()



class RestorationProcessor:
    """Unified interface for audio restoration."""
    
    def __init__(self, model_type="spectral", device=None):
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = None
        self.restorer = None
        self._loaded = False
    
    def load(self, progress_callback=None):
        """Load the selected restoration model."""
        if self._loaded:
            return
        
        if progress_callback:
            progress_callback(f"Initializing {self.model_type} restoration...")
        
        self.detector = SpectralHoleDetector()
        
        if self.model_type == "spectral":
            self.restorer = VampNetRestorer(self.device)
        elif self.model_type == "enhanced_spectral":
            self.restorer = EnhancedSpectralRestorer(self.device)
        elif self.model_type == "diffusion_inpaint":
            self.restorer = AudioInpaintingRestorer(self.device)
        elif self.model_type == "audiosr":
            self.restorer = AudioSRRestorer(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.restorer.load_model(progress_callback)
        self._loaded = True
    
    def process(self, instrumental_path, original_path, output_path,
                preset="balanced", threshold=0.5, max_regions=10,
                progress_callback=None):
        """Full restoration pipeline with improved blending."""
        if not self._loaded:
            self.load(progress_callback)
        
        if progress_callback:
            progress_callback("Loading audio files...")
        
        instrumental, sr = sf.read(instrumental_path)
        original, orig_sr = sf.read(original_path)
        
        # Resample if needed
        if orig_sr != sr:
            original = signal.resample(original, int(len(original) * sr / orig_sr))
        
        # Match lengths
        min_len = min(len(instrumental), len(original))
        instrumental = instrumental[:min_len]
        original = original[:min_len]
        
        if progress_callback:
            progress_callback("Detecting damaged regions...")
        
        self.detector.sample_rate = sr
        regions = self.detector.detect_spectral_damage(
            instrumental, original, threshold=threshold
        )
        
        if not regions:
            if progress_callback:
                progress_callback("No significant damage detected - saving as-is")
            sf.write(output_path, instrumental, sr)
            return output_path
        
        # Limit regions
        regions = regions[:max_regions]
        
        # Merge overlapping or adjacent regions for better processing
        regions = self._merge_adjacent_regions(regions, sr, gap_threshold=0.5)
        
        total_damage_time = sum(r[2] for r in regions)
        if progress_callback:
            progress_callback(f"Found {len(regions)} regions ({total_damage_time:.1f}s total)")
        
        result = instrumental.copy()
        
        for i, (start, end, duration, severity, freq_mask) in enumerate(regions):
            if progress_callback:
                progress_callback(f"Restoring region {i+1}/{len(regions)} ({duration:.1f}s, severity: {severity:.2f})")
            
            # Longer context for better blending
            context_seconds = min(1.0, duration * 0.3)  # Up to 1 second or 30% of region
            context_samples = int(context_seconds * sr)
            
            padded_start = max(0, start - context_samples)
            padded_end = min(len(instrumental), end + context_samples)
            
            inst_chunk = instrumental[padded_start:padded_end].copy()
            orig_chunk = original[padded_start:padded_end].copy()
            
            try:
                restored_chunk = self.restorer.restore_region(
                    inst_chunk, orig_chunk, freq_mask,
                    preset=preset,
                    progress_callback=progress_callback
                )
                
                # Extract the actual region from restored chunk
                actual_start = start - padded_start
                actual_end = actual_start + (end - start)
                
                if len(restored_chunk.shape) > 1:
                    restored_region = restored_chunk[actual_start:actual_end, :]
                else:
                    restored_region = restored_chunk[actual_start:actual_end]
                
                # Ensure correct length
                target_len = end - start
                restored_region = self._adjust_length(restored_region, target_len)
                
                # Improved crossfade with longer, smoother transitions
                crossfade_samples = min(int(0.1 * sr), target_len // 3)  # Up to 100ms or 1/3 of region
                
                if crossfade_samples > 10:  # Only crossfade if meaningful
                    orig_region = instrumental[start:end].copy()
                    restored_region = self._apply_crossfade(
                        orig_region, restored_region, crossfade_samples
                    )
                
                result[start:end] = restored_region
                
                logger.info(f"Region {i+1} restored successfully")
                
            except Exception as e:
                logger.error(f"Failed to restore region {i+1}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
            
            # Periodic memory cleanup
            if i % 3 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Final processing
        result = np.clip(result, -1.0, 1.0)
        
        # Optional: Apply subtle global smoothing to reduce any remaining artifacts
        # result = self._final_smoothing(result, sr)
        
        sf.write(output_path, result, sr)
        
        if progress_callback:
            progress_callback("✅ Restoration complete!")
        
        return output_path
    
    def _merge_adjacent_regions(self, regions, sr, gap_threshold=0.5):
        """Merge regions that are close together for more coherent processing."""
        if len(regions) <= 1:
            return regions
        
        # Sort by start position
        regions = sorted(regions, key=lambda r: r[0])
        
        merged = []
        current = list(regions[0])
        
        for next_region in regions[1:]:
            gap_samples = next_region[0] - current[1]
            gap_seconds = gap_samples / sr
            
            if gap_seconds <= gap_threshold:
                # Merge regions
                current[1] = next_region[1]  # Extend end
                current[2] = (current[1] - current[0]) / sr  # Update duration
                current[3] = max(current[3], next_region[3])  # Max severity
                # Combine frequency masks
                current[4] = current[4] | next_region[4]
            else:
                merged.append(tuple(current))
                current = list(next_region)
        
        merged.append(tuple(current))
        
        return merged
    
    def _adjust_length(self, audio, target_len):
        """Adjust audio array to target length."""
        is_stereo = len(audio.shape) > 1 and audio.shape[1] == 2
        current_len = audio.shape[0] if is_stereo else len(audio)
        
        if current_len == target_len:
            return audio
        
        if is_stereo:
            if current_len > target_len:
                return audio[:target_len, :]
            else:
                pad_len = target_len - current_len
                return np.pad(audio, ((0, pad_len), (0, 0)), mode='edge')
        else:
            if current_len > target_len:
                return audio[:target_len]
            else:
                return np.pad(audio, (0, target_len - current_len), mode='edge')
    
    def _apply_crossfade(self, original, restored, crossfade_samples):
        """Apply smooth crossfade between original and restored."""
        is_stereo = len(restored.shape) > 1 and restored.shape[1] == 2
        
        # Create smooth crossfade curves (raised cosine)
        t = np.linspace(0, np.pi, crossfade_samples)
        fade_in = 0.5 * (1 - np.cos(t))   # 0 -> 1
        fade_out = 0.5 * (1 + np.cos(t))  # 1 -> 0
        
        result = restored.copy()
        
        if is_stereo:
            fade_in_2d = fade_in[:, np.newaxis]
            fade_out_2d = fade_out[:, np.newaxis]
            
            # Fade in at start
            result[:crossfade_samples, :] = (
                original[:crossfade_samples, :] * (1 - fade_in_2d) +
                restored[:crossfade_samples, :] * fade_in_2d
            )
            
            # Fade out at end
            result[-crossfade_samples:, :] = (
                restored[-crossfade_samples:, :] * fade_out_2d +
                original[-crossfade_samples:, :] * (1 - fade_out_2d)
            )
        else:
            # Fade in at start
            result[:crossfade_samples] = (
                original[:crossfade_samples] * (1 - fade_in) +
                restored[:crossfade_samples] * fade_in
            )
            
            # Fade out at end
            result[-crossfade_samples:] = (
                restored[-crossfade_samples:] * fade_out +
                original[-crossfade_samples:] * (1 - fade_out)
            )
        
        return result



    def cleanup(self):
        """Release resources."""
        if self.restorer:
            self.restorer.cleanup()
        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()



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
                 musicgen_preprocess=False, musicgen_temp=1.0): # <--- NEW PARAM
        
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
        self.shifts = shifts
        self.overlap = overlap
        self.use_float32 = use_float32
        self.device = device
        
        self.musicgen_prompt = musicgen_prompt
        self.musicgen_model = musicgen_model
        self.musicgen_preprocess = musicgen_preprocess
        self.musicgen_temp = musicgen_temp

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # --- MUSICGEN MODE ---
            if self.mode == "musicgen_regen":
                if not AVAILABLE_MODELS["musicgen"]:
                    raise Exception("MusicGen not installed.")
                
                target_file = self.input_file
                
                # 1. PRE-PROCESS (Remove Vocals) if checked
                if self.musicgen_preprocess:
                    self.progress.emit("Pre-processing: Removing vocals with Demucs...")
                    from demucs.pretrained import get_model
                    from demucs.apply import apply_model
                    import torchaudio
                    
                    model = get_model("htdemucs")
                    model.to(self.device)
                    wav, sr = torchaudio.load(self.input_file)
                    
                    # Separate
                    ref = wav.mean(0)
                    wav = (wav - ref.mean()) / ref.std()
                    sources = apply_model(model, wav[None], shifts=0, overlap=0.25)[0]
                    sources = sources * ref.std() + ref.mean()
                    
                    # Sum everything EXCEPT vocals (index 3)
                    no_vocals = sources[0] + sources[1] + sources[2] 
                    
                    # Save temp instrumental
                    temp_inst_path = os.path.join(self.output_dir, "temp_instrumental_input.wav")
                    torchaudio.save(temp_inst_path, no_vocals.cpu(), sr)
                    
                    target_file = temp_inst_path
                    self.progress.emit("Vocals removed. Initializing MusicGen...")

                # 2. RUN MUSICGEN
                mg = MusicGenRegenerator(device=self.device)
                self.progress.emit(f"Loading MusicGen model: {self.musicgen_model}...")
                mg.load_model(self.musicgen_model)
                
                out_file = os.path.join(self.output_dir, "regenerated_song.wav")
                prompt = self.musicgen_prompt if self.musicgen_prompt.strip() else "high quality instrumental song"
                
                mg.regenerate(
                    target_file, 
                    out_file, 
                    prompt=prompt,
                    temperature=self.musicgen_temp, # <--- Pass the temp
                    progress_callback=self.progress.emit
                )
                
                # Clean up temp file
                if self.musicgen_preprocess and os.path.exists(target_file) and target_file != self.input_file:
                    try:
                        os.remove(target_file)
                    except:
                        pass
                
                self.finished.emit(self.output_dir)
                return

            # --- STANDARD SEPARATION MODES ---
            if self.mode == "full_stems":
                self.progress.emit("Starting Demucs...")
                from demucs.pretrained import get_model
                from demucs.apply import apply_model
                import torchaudio
                
                model = get_model(self.demucs_model)
                model.to(self.device)
                wav, sr = torchaudio.load(self.input_file)
                
                ref = wav.mean(0)
                wav = (wav - ref.mean()) / ref.std()
                sources = apply_model(model, wav[None], shifts=self.shifts, overlap=self.overlap)[0]
                sources = sources * ref.std() + ref.mean()
                
                for source, name in zip(sources, model.sources):
                    torchaudio.save(os.path.join(self.output_dir, f"{name}.wav"), source.cpu(), sr)

            elif self.mode in ["vocals_only", "instrument_hq"]:
                self.progress.emit("Starting UVR Separation...")
                sep = UVRSeparator(self.output_dir, self.device)
                
                # ← FIX: CAPTURE THE RETURN VALUES
                vocals_path, instrumental_path = sep.separate_single(
                    self.input_file, 
                    self.separation_model, 
                    progress_callback=self.progress.emit
                )
                
                # ← FIX: COPY FILES TO OUTPUT DIR BEFORE CLEANUP
                if vocals_path and os.path.exists(vocals_path):
                    final_vocals = os.path.join(self.output_dir, "vocals.wav")
                    shutil.copy2(vocals_path, final_vocals)
                    self.progress.emit(f"✅ Saved: vocals.wav")
                
                if instrumental_path and os.path.exists(instrumental_path):
                    final_inst = os.path.join(self.output_dir, "instrumental.wav")
                    shutil.copy2(instrumental_path, final_inst)
                    self.progress.emit(f"✅ Saved: instrumental.wav")
                
                sep.cleanup()  # Now safe to cleanup temp files

            self.finished.emit(self.output_dir)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))






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
















class StemSeparatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Stem Separator - High Quality")
        self.setGeometry(100, 100, 520, 750)
        self.setMinimumWidth(420)
        
        self.worker_thread = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.setCentralWidget(scroll)
        
        widget = QWidget()
        scroll.setWidget(widget)
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Status
        self.status_label = QLabel("Drag & drop or browse for an audio file.")
        layout.addWidget(self.status_label)
        
        # Model availability warnings
        self._add_availability_warnings(layout)
        
        # Drop Zone
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.handle_dropped_file)
        layout.addWidget(self.drop_zone)
        
        # File
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        file_layout.addWidget(self.file_input)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)
        layout.addWidget(file_group)

        # Mode
        mode_group = QGroupBox("🎯 Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_group = QButtonGroup(self)
        
        modes = [
            ("🎤 Vocals Only", "vocals_only", False),
            ("🎸 High-Quality Instrumental", "instrument_hq", True),
            ("🎹 Full Stems (Demucs)", "full_stems", False),
            ("🤖 MusicGen Re-creation", "musicgen_regen", False) # <--- ADD THIS LINE
        ]
        
        for text, mode, checked in modes:
            radio = QRadioButton(text)
            radio.mode = mode
            radio.setChecked(checked)
            radio.toggled.connect(self._on_mode_changed)
            self.mode_group.addButton(radio)
            mode_layout.addWidget(radio)
        
        layout.addWidget(mode_group)

        # Separation Settings
        sep_group = QGroupBox("🎵 Separation Settings")
        sep_layout = QVBoxLayout(sep_group)
        sep_layout.setSpacing(8)

        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        method_label.setFixedWidth(80)
        method_layout.addWidget(method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItem("UVR Single Model (Fast)", "uvr_single")
        self.method_combo.addItem("UVR Ensemble (Best)", "uvr_ensemble")
        self.method_combo.setCurrentIndex(0)
        self.method_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        method_layout.addWidget(self.method_combo)
        sep_layout.addLayout(method_layout)

        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setFixedWidth(80)
        model_layout.addWidget(model_label)
        self.sep_model_combo = QComboBox()
        for model_id, model_name in UVRSeparator.INSTRUMENTAL_MODELS:
            self.sep_model_combo.addItem(model_name, model_id)
        self.sep_model_combo.setCurrentIndex(0)
        self.sep_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        model_layout.addWidget(self.sep_model_combo)
        sep_layout.addLayout(model_layout)

        layout.addWidget(sep_group)

        # Restoration Settings
        self.restore_group = QGroupBox("🔧 AI Restoration")
        self.restore_group.setCheckable(True)
        self.restore_group.setChecked(True)
        restore_layout = QVBoxLayout(self.restore_group)
        restore_layout.setSpacing(8)

        # Restoration model dropdown
        restore_model_layout = QHBoxLayout()
        restore_model_label = QLabel("Model:")
        restore_model_label.setFixedWidth(80)
        restore_model_layout.addWidget(restore_model_label)
        self.restore_model_combo = QComboBox()
        self.restore_model_combo.addItem("⚡ Spectral Restore (Fast)", "spectral")
        self.restore_model_combo.addItem("🎵 Enhanced Spectral (Medium)", "enhanced_spectral")
        self.restore_model_combo.addItem("🧠 Diffusion Inpaint (Best)", "diffusion_inpaint")
        self.restore_model_combo.addItem("🔊 AudioSR (Legacy)", "audiosr")
        self.restore_model_combo.setCurrentIndex(1)  # Default to Enhanced Spectral
        self.restore_model_combo.currentIndexChanged.connect(self._on_restore_model_changed)
        self.restore_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        restore_model_layout.addWidget(self.restore_model_combo)
        restore_layout.addLayout(restore_model_layout)


        # --- MUSICGEN SETTINGS ---
        self.musicgen_group = QGroupBox("🎹 MusicGen Settings")
        mg_layout = QVBoxLayout(self.musicgen_group)
        
        # Model Selector
        mg_model_layout = QHBoxLayout()
        mg_model_layout.addWidget(QLabel("Model:"))
        self.mg_model_combo = QComboBox()
        self.mg_model_combo.addItem("Melody (Best Structure)", "melody")
        self.mg_model_combo.addItem("Medium (Better Audio Quality)", "medium")
        self.mg_model_combo.addItem("Large (Best Audio / Slow)", "large")
        self.mg_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        mg_model_layout.addWidget(self.mg_model_combo)
        mg_layout.addLayout(mg_model_layout)
        
        # Pre-process Checkbox
        self.mg_preprocess_check = QCheckBox("⚡ Pre-separate Vocals (Recommended)")
        self.mg_preprocess_check.setChecked(True)
        mg_layout.addWidget(self.mg_preprocess_check)

        # NEW: CREATIVITY SLIDER
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Creativity:"))
        
        self.mg_temp_slider = QSlider(Qt.Horizontal)
        self.mg_temp_slider.setRange(1, 15) # 0.1 to 1.5
        self.mg_temp_slider.setValue(8)     # Default 0.8
        self.mg_temp_slider.setToolTip("Lower = Stricter adherence to original. Higher = More improvised.")
        temp_layout.addWidget(self.mg_temp_slider)
        
        self.mg_temp_label = QLabel("0.8")
        self.mg_temp_label.setFixedWidth(30)
        temp_layout.addWidget(self.mg_temp_label)
        
        # Connect label update
        self.mg_temp_slider.valueChanged.connect(lambda v: self.mg_temp_label.setText(f"{v/10.0:.1f}"))
        
        mg_layout.addLayout(temp_layout)

        # Prompt Input
        mg_layout.addWidget(QLabel("Style Prompt:"))
        self.mg_prompt_input = QTextEdit()
        self.mg_prompt_input.setPlaceholderText("e.g. Acoustic guitar version, synthwave style...")
        self.mg_prompt_input.setText("High quality instrumental song, no vocals")
        self.mg_prompt_input.setMaximumHeight(60)
        mg_layout.addWidget(self.mg_prompt_input)
        
        layout.addWidget(self.musicgen_group)



        # Model info
        self.restore_info_label = QLabel("🎵 Harmonic-aware restoration for medium damage")
        self.restore_info_label.setStyleSheet("color: #666; font-size: 10px;")
        restore_layout.addWidget(self.restore_info_label)

        # Preset buttons
        preset_label = QLabel("Quality Preset:")
        restore_layout.addWidget(preset_label)
        
        preset_layout = QHBoxLayout()
        preset_layout.setSpacing(5)
        self.preset_group = QButtonGroup(self)
        
        presets = [
            ("Ultra Fast", "ultra_fast", "⚡ ~2-5s per region"),
            ("Fast", "fast", "🚀 ~5-15s per region"),
            ("Balanced", "balanced", "⚖️ ~15-30s per region"),
            ("Quality", "high_quality", "✨ ~30-60s per region"),
        ]
        
        for text, preset_id, tooltip in presets:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.preset_id = preset_id
            btn.setToolTip(tooltip)
            btn.setStyleSheet("""
                QPushButton { padding: 5px 10px; }
                QPushButton:checked { background-color: #007bff; color: white; }
            """)
            if preset_id == "balanced":
                btn.setChecked(True)
            self.preset_group.addButton(btn)
            preset_layout.addWidget(btn)
        
        restore_layout.addLayout(preset_layout)

        # Sensitivity slider
        sens_layout = QHBoxLayout()
        sens_label = QLabel("Sensitivity:")
        sens_label.setFixedWidth(80)
        sens_layout.addWidget(sens_label)
        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(20, 80)
        self.sens_slider.setValue(50)
        self.sens_slider.setToolTip("Lower = detect more damage, Higher = only severe damage")
        sens_layout.addWidget(self.sens_slider)
        self.sens_value = QLabel("0.50")
        self.sens_value.setFixedWidth(35)
        sens_layout.addWidget(self.sens_value)
        restore_layout.addLayout(sens_layout)
        self.sens_slider.valueChanged.connect(lambda v: self.sens_value.setText(f"{v/100:.2f}"))

        # Max regions
        regions_layout = QHBoxLayout()
        regions_label = QLabel("Max regions:")
        regions_label.setFixedWidth(80)
        regions_layout.addWidget(regions_label)
        self.max_regions_spin = QSpinBox()
        self.max_regions_spin.setRange(1, 50)
        self.max_regions_spin.setValue(10)
        self.max_regions_spin.setToolTip("Maximum number of damaged regions to restore")
        self.max_regions_spin.setFixedWidth(60)
        regions_layout.addWidget(self.max_regions_spin)
        regions_layout.addStretch()
        restore_layout.addLayout(regions_layout)

        layout.addWidget(self.restore_group)

        # Demucs Settings
        self.demucs_group = QGroupBox("⚙️ Demucs Settings")
        demucs_layout = QVBoxLayout(self.demucs_group)
        demucs_layout.setSpacing(8)
        
        demucs_model_layout = QHBoxLayout()
        dm_label = QLabel("Model:")
        dm_label.setFixedWidth(80)
        demucs_model_layout.addWidget(dm_label)
        self.demucs_model_combo = QComboBox()
        self.demucs_model_combo.addItem("HTDemucs 6s (Best)", "htdemucs_6s")
        self.demucs_model_combo.addItem("HTDemucs FT", "htdemucs_ft")
        self.demucs_model_combo.addItem("HTDemucs", "htdemucs")
        self.demucs_model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        demucs_model_layout.addWidget(self.demucs_model_combo)
        demucs_layout.addLayout(demucs_model_layout)
        
        shifts_layout = QHBoxLayout()
        shifts_label = QLabel("Shifts:")
        shifts_label.setFixedWidth(80)
        shifts_layout.addWidget(shifts_label)
        self.shifts_spin = QSpinBox()
        self.shifts_spin.setRange(1, 10)
        self.shifts_spin.setValue(5)
        self.shifts_spin.setFixedWidth(60)
        shifts_layout.addWidget(self.shifts_spin)
        shifts_layout.addStretch()
        demucs_layout.addLayout(shifts_layout)
        
        self.float32_check = QCheckBox("Float32 Precision")
        self.float32_check.setChecked(True)
        demucs_layout.addWidget(self.float32_check)
        
        # Device
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        device_label.setFixedWidth(80)
        device_layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            self.device_combo.addItem(f"🚀 GPU ({gpu_name})", "cuda")
            self.device_combo.addItem("🐢 CPU", "cpu")
        else:
            self.device_combo.addItem("🐢 CPU", "cpu")
        
        self.device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        device_layout.addWidget(self.device_combo)
        demucs_layout.addLayout(device_layout)
        
        if not cuda_available:
            gpu_warning = QLabel("⚠️ No CUDA GPU detected - processing will be slow")
            gpu_warning.setStyleSheet("color: orange;")
            demucs_layout.addWidget(gpu_warning)
        
        layout.addWidget(self.demucs_group)

        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setWordWrap(True)
        self.progress_label.setMinimumHeight(40)
        progress_layout.addWidget(self.progress_label)
        
        self.start_btn = QPushButton("🚀 Start")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_separation)
        self.start_btn.setStyleSheet("""
            QPushButton { 
                background-color: #28a745; 
                color: white; 
                font-weight: bold; 
                padding: 12px; 
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton:disabled { background-color: #ccc; color: #666; }
        """)
        progress_layout.addWidget(self.start_btn)
        
        layout.addWidget(progress_group)
        layout.addStretch()
        
        # Initial UI state
        self._on_mode_changed()
    
    def _add_availability_warnings(self, layout):
        """Add warnings for missing models."""
        warnings = []
        
        if not AVAILABLE_MODELS.get("audio_separator"):
            warnings.append("audio-separator (UVR5/MDX)")
        if not AVAILABLE_MODELS.get("diffusion_inpaint"):
            warnings.append("diffusion-inpaint")
        if not AVAILABLE_MODELS.get("audiosr"):
            warnings.append("audiosr")
        
        if warnings:
            warning_text = "⚠️ Missing: " + ", ".join(warnings)
            warning_label = QLabel(warning_text)
            warning_label.setStyleSheet(
                "color: #856404; background-color: #fff3cd; "
                "padding: 8px; border-radius: 4px; font-size: 11px;"
            )
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)
    
    def _on_mode_changed(self):
        """Update UI based on selected mode."""
        mode = self.get_mode()
        
        # Existing visibility logic
        show_restoration = (mode == "instrument_hq")
        self.restore_group.setVisible(show_restoration)
        
        show_demucs = (mode == "full_stems")
        self.demucs_group.setVisible(show_demucs or mode == "instrument_hq")
        
        # --- NEW: Hide/Show MusicGen Group ---
        self.musicgen_group.setVisible(mode == "musicgen_regen")
        
        # Hide other groups if MusicGen is selected
        if mode == "musicgen_regen":
            self.restore_group.setVisible(False)
            self.demucs_group.setVisible(False)
            # Also hide UVR/Sep group if you have one defined as sep_group
            if hasattr(self, 'sep_group'): self.sep_group.setVisible(False)
 


 
    def _on_restore_model_changed(self, index):
        """Update info label when restoration model changes."""
        model = self.restore_model_combo.currentData()
        
        info_map = {
            "spectral": "⚡ Fast basic interpolation for minor damage",
            "enhanced_spectral": "🎵 Harmonic-aware restoration for medium damage",
            "diffusion_inpaint": "🧠 AI diffusion inpainting - best quality, slower",
            "audiosr": "🔊 Legacy upsampling - not recommended for holes",
        }
        
        self.restore_info_label.setText(info_map.get(model, ""))
    
    def get_mode(self):
        """Get currently selected mode."""
        for btn in self.mode_group.buttons():
            if btn.isChecked():
                return btn.mode
        return "instrument_hq"
    
    def get_preset(self):
        """Get currently selected quality preset."""
        for btn in self.preset_group.buttons():
            if btn.isChecked():
                return btn.preset_id
        return "balanced"
    
    @Slot(str)
    def handle_dropped_file(self, path):
        """Handle file dropped onto drop zone."""
        if os.path.exists(path):
            self.file_input.setText(path)
            self.start_btn.setEnabled(True)
            self.status_label.setText(f"✓ Loaded: {os.path.basename(path)}")
    
    @Slot()
    def browse_file(self):
        """Open file browser dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio", "",
            "Audio Files (*.mp3 *.wav *.flac *.m4a *.ogg *.aac);;All Files (*.*)"
        )
        if path:
            self.file_input.setText(path)
            self.start_btn.setEnabled(True)
            self.status_label.setText(f"✓ Selected: {os.path.basename(path)}")
    
    @Slot()
    def start_separation(self):
        """Start the separation/restoration process."""
        path = self.file_input.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Please select a valid audio file.")
            return
        
        mode = self.get_mode()
        
        if self.device_combo.currentData() == "cpu":
            if mode == "instrument_hq" and self.restore_group.isChecked():
                reply = QMessageBox.question(
                    self, "CPU Warning",
                    "AI restoration on CPU will be very slow.\n\n"
                    "A CUDA GPU is strongly recommended.\n\n"
                    "Continue anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        
        base = os.path.dirname(path)
        name = os.path.splitext(os.path.basename(path))[0]
        
        if mode == "instrument_hq" and self.restore_group.isChecked():
            restore_model = self.restore_model_combo.currentData()
            out_dir = os.path.join(base, f"{name}_instrumental_{restore_model}")
        else:
            out_dir = os.path.join(base, f"{name}_{mode.replace('_', '-')}")
        
        self.status_label.setText("Processing...")
        self.set_ui_enabled(False)
        self.progress_bar.show()
        self.progress_label.setText("Initializing...")
        
        self.worker_thread = ExtractionWorker(
            input_file=path,
            output_dir=out_dir,
            mode=mode,
            demucs_model=self.demucs_model_combo.currentData(),
            separation_model=self.sep_model_combo.currentData(),
            separation_method="uvr_single", # or self.method_combo.currentData() if you kept it
            enable_restoration=False, # logic simplified
            restoration_model="spectral",
            restoration_preset="balanced",
            threshold=0.5,
            max_regions=10,
            shifts=2,
            overlap=0.25,
            use_float32=True,
            device=self.device_combo.currentData(),
            
            # --- UPDATED PARAMS ---
            musicgen_prompt=self.mg_prompt_input.toPlainText(),
            musicgen_model=self.mg_model_combo.currentData(),
            musicgen_preprocess=self.mg_preprocess_check.isChecked(),
            musicgen_temp=self.mg_temp_slider.value() / 10.0 # Convert 8 -> 0.8
        )
        
        self.worker_thread.finished.connect(self.on_finished)
        self.worker_thread.error.connect(self.on_error)
        self.worker_thread.progress.connect(self.on_progress)

        try:
            self.worker_thread.start()
        except Exception as e:
            import traceback
            QMessageBox.critical(
                self, "Startup Error", 
                f"Failed to start processing:\n\n{traceback.format_exc()}"
            )
            self.set_ui_enabled(True)
            self.progress_bar.hide()
    
    def set_ui_enabled(self, enabled):
        """Enable/disable UI elements during processing."""
        widgets = [
            self.browse_btn, 
            self.demucs_model_combo, 
            self.sep_model_combo,
            self.method_combo, 
            self.shifts_spin, 
            self.float32_check,
            self.restore_model_combo,
            self.sens_slider,
            self.max_regions_spin,
            self.device_combo,
            self.restore_group,
        ]
        
        for w in widgets:
            w.setEnabled(enabled)
        
        for btn in self.mode_group.buttons():
            btn.setEnabled(enabled)
        
        for btn in self.preset_group.buttons():
            btn.setEnabled(enabled)
        
        self.start_btn.setEnabled(enabled and bool(self.file_input.text()))
    
    @Slot(str)
    def on_progress(self, message):
        """Update progress label."""
        self.progress_label.setText(message)
        logger.info(f"Progress: {message}")
    
    @Slot(str)
    def on_finished(self, out_dir):
        """Handle successful completion."""
        self.progress_bar.hide()
        self.progress_label.setText("")
        self.set_ui_enabled(True)
        self.status_label.setText("✅ Complete!")
        
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Success")
        msg.setText("Processing complete!")
        msg.setInformativeText(f"Files saved to:\n{out_dir}")
        
        msg.addButton("Open Folder", QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        
        result = msg.exec()
        
        if msg.clickedButton().text() == "Open Folder":
            if sys.platform == 'win32':
                os.startfile(out_dir)
            elif sys.platform == 'darwin':
                os.system(f'open "{out_dir}"')
            else:
                os.system(f'xdg-open "{out_dir}"')
    
    @Slot(str)
    def on_error(self, msg):
        """Handle error during processing."""
        self.progress_bar.hide()
        self.progress_label.setText("")
        self.set_ui_enabled(True)
        self.status_label.setText("❌ Error occurred")
        
        QMessageBox.critical(self, "Error", msg)



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