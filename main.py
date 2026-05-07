import io
import json
import logging
import random
import os
import csv
import tempfile
import pathlib
from typing import Dict, List, Optional, Tuple, Any

import librosa
import noisereduce as nr
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, get_db
import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Try to import heavy ML libraries gracefully (for demonstration without large downloads)
try:
    import torch
    # pyrefly: ignore [missing-import]
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    VADER_AVAILABLE = True
    NLTK_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    NLTK_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emolens Multimodal Emotion Recognition API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard emotion categories
EMOTIONS = ["Happy", "Sad", "Angry", "Neutral"]

# Emotion to emoji mapping
EMOTION_EMOJI_MAP = {
    "Happy": "😄", "Sad": "😢", "Angry": "😡", "Neutral": "😐",
    "joy": "😄", "sadness": "😢", "anger": "😡", "love": "🥰",
    "fear": "😨", "surprise": "😮", "disgust": "🤢"
}

# --- Modality Pipelines ---

class TextModality:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        self.transformer = None
        self.tokenizer = None
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.stop_words = set(stopwords.words("english")) if NLTK_AVAILABLE else set()
        
        # English emotion lexicon
        self.english_emotion_words: Dict[str, str] = {
            "happy": "Happy", "joy": "Happy", "glad": "Happy", "pleased": "Happy",
            "love": "Happy", "loving": "Happy", "sad": "Sad", "sorrow": "Sad",
            "cry": "Sad", "angry": "Angry", "mad": "Angry", "fear": "Sad",
            "afraid": "Sad", "scared": "Sad", "surprise": "Neutral", "surprised": "Neutral",
            "disgust": "Angry", "disgusted": "Angry"
        }
        
        # Multilingual emotion lexicon
        self.multilingual_lexicon: Dict[str, Dict[str, str]] = {
            "es": {"feliz": "Happy", "triste": "Sad", "enojo": "Angry", "amor": "Happy"},
            "hi": {"खुश": "Happy", "दुखी": "Sad", "गुस्सा": "Angry", "प्यार": "Happy"},
            "fr": {"heureux": "Happy", "triste": "Sad", "colère": "Angry", "amour": "Happy"},
            "de": {"glücklich": "Happy", "traurig": "Sad", "wut": "Angry", "liebe": "Happy"},
            "it": {"felice": "Happy", "triste": "Sad", "rabbia": "Angry", "amore": "Happy"},
            "pt": {"feliz": "Happy", "tristeza": "Sad", "raiva": "Angry", "amor": "Happy"},
            "ru": {"радость": "Happy", "грусть": "Sad", "гнев": "Angry", "любовь": "Happy"}
        }
        
        # Load transformer if available
        if TRANSFORMERS_AVAILABLE:
            self._load_transformer()

    def _load_transformer(self):
        """Load sentiment transformer with fallback models"""
        candidates = [
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "distilbert-base-uncased-finetuned-sst-2-english"
        ]
        for model_name in candidates:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.transformer = pipeline("sentiment-analysis", model=model, tokenizer=self.tokenizer)
                logger.info(f"Loaded transformer: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        logger.warning("No transformer model available, will use lexicon fallback")

    def preprocess_text(self, text: str) -> Tuple[List[str], str]:
        """Preprocess text with tokenization and lemmatization"""
        if not text:
            return [], ""
        try:
            text = text.lower()
            if NLTK_AVAILABLE:
                tokens = word_tokenize(text)
            else:
                tokens = text.split()
            
            words = [
                self.lemmatizer.lemmatize(w) if self.lemmatizer else w
                for w in tokens if w.isalpha() and w not in self.stop_words
            ]
            return words, " ".join(words)
        except Exception as e:
            logger.warning(f"Preprocess fallback: {e}")
            words = [w for w in text.lower().split() if w.isalpha() and w not in self.stop_words]
            return words, " ".join(words)

    def analyze(self, text: str, language: str = "en") -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Analyze text emotion using ensemble of methods.
        Returns emotion probabilities and metadata.
        """
        if not text or not text.strip():
            return np.array([0.25, 0.25, 0.25, 0.25]), {
                "method": "None",
                "language": language,
                "confidence": 0.0
            }

        words, cleaned_text = self.preprocess_text(text)
        
        # 1. Lexicon-based approach (baseline)
        lexicon_probs = self._lexicon_analyze(words, text, language)
        
        # 2. VADER sentiment analysis
        vader_probs = self._vader_analyze(cleaned_text)
        
        # 3. Transformer approach
        transformer_probs = self._transformer_analyze(cleaned_text)
        
        # Ensemble: weighted combination
        final_probs = (
            0.3 * lexicon_probs +
            0.3 * vader_probs +
            0.4 * transformer_probs
        )
        
        # Normalize
        final_probs = final_probs / (np.sum(final_probs) + 1e-6)
        
        return final_probs, {
            "method": "Ensemble (Lexicon + VADER + Transformer)",
            "language": language,
            "confidence": float(np.max(final_probs))
        }

    def _lexicon_analyze(self, words: List[str], original_text: str, language: str) -> np.ndarray:
        """Emotion detection using lexicon"""
        emotion_counts = {"Happy": 0, "Sad": 0, "Angry": 0, "Neutral": 0}
        
        # Get language-specific lexicon
        lang_lexicon = self.multilingual_lexicon.get(language, {})
        
        for w in words:
            emotion = lang_lexicon.get(w) or self.english_emotion_words.get(w)
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
        
        total = sum(emotion_counts.values())
        if total > 0:
            probs = np.array([emotion_counts[e] / total for e in EMOTIONS])
        else:
            probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        return probs

    def _vader_analyze(self, text: str) -> np.ndarray:
        """VADER sentiment analysis"""
        if not self.vader:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        try:
            scores = self.vader.polarity_scores(text)
            # Map VADER (pos, neg, neu) to our 4 emotions
            probs = np.array([
                max(0, scores["pos"]),           # Happy
                max(0, scores["neg"]),           # Sad
                max(0, scores["neg"]),           # Angry (use neg)
                max(0, scores["neu"])            # Neutral
            ])
            probs = probs / (np.sum(probs) + 1e-6)
            return probs
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return np.array([0.25, 0.25, 0.25, 0.25])

    def _transformer_analyze(self, text: str) -> np.ndarray:
        """Transformer-based sentiment analysis"""
        if not self.transformer:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        try:
            results = self.transformer(text[:512])  # Limit to 512 tokens
            if isinstance(results, list) and results:
                label = results[0].get("label", "").lower()
                score = float(results[0].get("score", 0.5))
                
                # Map to emotions
                if "positive" in label or "label_2" in label:
                    return np.array([score, 0.1, 0.1, 1.0 - score])
                elif "negative" in label or "label_0" in label:
                    return np.array([0.1, score * 0.5, score * 0.5, 1.0 - score])
                else:
                    return np.array([0.25, 0.25, 0.25, 0.25])
        except Exception as e:
            logger.warning(f"Transformer analysis failed: {e}")
        
        return np.array([0.25, 0.25, 0.25, 0.25])


class AudioModality:
    def __init__(self):
        self.whisper_model = None
        self.sr_recognizer = sr.Recognizer() if SR_AVAILABLE else None
        
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded")
            except Exception as e:
                logger.warning(f"Failed to load Whisper: {e}")

    def transcribe(self, audio_bytes: bytes, method: str = "whisper") -> str:
        """
        Transcribe audio to text using Whisper or Speech Recognition.
        Returns: transcribed text
        """
        if not audio_bytes:
            return ""
        
        temp_path = None
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                tmp.flush()
                temp_path = tmp.name
            
            # Try Whisper first
            if method == "whisper" and self.whisper_model:
                try:
                    result = self.whisper_model.transcribe(temp_path, language="en")
                    return result.get("text", "").strip()
                except Exception as e:
                    logger.warning(f"Whisper transcription failed: {e}")
            
            # Fallback to speech_recognition
            if SR_AVAILABLE and self.sr_recognizer:
                try:
                    with sr.AudioFile(temp_path) as source:
                        self.sr_recognizer.adjust_for_ambient_noise(source)
                        audio_data = self.sr_recognizer.record(source)
                    return self.sr_recognizer.recognize_google(audio_data)
                except Exception as e:
                    logger.warning(f"SpeechRecognition failed: {e}")
            
            return ""
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

    def analyze(self, audio_bytes: bytes) -> Tuple[np.ndarray, float, str]:
        """
        Processes audio data, performs denoising, and returns emotion probabilities, 
        confidence score, and transcription.
        """
        if not audio_bytes:
            return np.array([0.25, 0.25, 0.25, 0.25]), 0.0, ""

        try:
            # Load audio using librosa
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            
            # Calculate noise level (SNR proxy)
            stft = np.abs(librosa.stft(y))
            noise_energy = np.mean(stft[100:]) if stft.shape[0] > 100 else np.mean(stft)
            signal_energy = np.mean(stft[:100]) if stft.shape[0] > 100 else np.mean(stft)
            snr = signal_energy / (noise_energy + 1e-6)
            
            # MANDATORY Denoising Step
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            
            # Feature Extraction (MFCCs)
            mfccs = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=13)
            
            # Determine confidence based on SNR
            confidence = min(1.0, max(0.1, snr / 10.0))
            if snr < 2.0:
                logger.warning(f"Audio is highly noisy (SNR: {snr:.2f}). Reducing audio modality weight.")
            
            # Transcribe audio
            transcription = self.transcribe(audio_bytes)
            
            # Simulate audio-based emotion classification
            np.random.seed(int(np.sum(mfccs) % 10000))
            probs = np.random.dirichlet(np.ones(4), size=1)[0]
            
            return probs, float(confidence), transcription

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return np.array([0.25, 0.25, 0.25, 0.25]), 0.1, ""


class VisualModality:
    def __init__(self):
        self.model = None
        self.facecasc = None
        if TENSORFLOW_AVAILABLE:
            try:
                model_path = r"c:/Users/govin/MER/mock_dataset/Emotion-Detection-FER2013-master/Emotion-Detection-FER2013-master/emotion_detection_model.h5"
                cascade_path = r"c:/Users/govin/MER/mock_dataset/Emotion-Detection-FER2013-master/Emotion-Detection-FER2013-master/haarcascade_frontalface_default.xml"
                
                if os.path.exists(model_path) and os.path.exists(cascade_path):
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                    self.model = tf.keras.models.load_model(model_path)
                    self.facecasc = cv2.CascadeClassifier(cascade_path)
                    logger.info("Successfully loaded FER2013 visual model.")
                else:
                    logger.warning("FER2013 model or cascade not found at mock_dataset.")
            except Exception as e:
                logger.error(f"Failed to load visual model: {e}")

    def analyze(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Visual input could not be decoded into an image.")
                return np.array([0.25, 0.25, 0.25, 0.25])
                
            if self.model is not None and self.facecasc is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    
                    prediction = self.model.predict(cropped_img, verbose=0)[0]
                    
                    # Map FER2013 emotions to our 4 categories
                    # 0=Angry, 1=Disgusted, 2=Fearful, 3=Happy, 4=Neutral, 5=Sad, 6=Surprised
                    mapped_probs = np.array([
                        prediction[3],  # Happy
                        prediction[5],  # Sad
                        prediction[0],  # Angry
                        prediction[4]   # Neutral
                    ])
                    
                    total = np.sum(mapped_probs)
                    if total > 0:
                        mapped_probs = mapped_probs / total
                    else:
                        mapped_probs = np.array([0.25, 0.25, 0.25, 0.25])
                        
                    return mapped_probs
                else:
                    logger.warning("No face detected in the image.")
                    return np.array([0.25, 0.25, 0.25, 0.25])
            else:
                # Simulate if model not available
                np.random.seed(int(np.sum(img) % 10000))
                probs = np.random.dirichlet(np.ones(4), size=1)[0]
                return probs
        except Exception as e:
            logger.error(f"Visual processing error: {e}")
            return np.array([0.25, 0.25, 0.25, 0.25])


# --- Fusion Engine with Modality Selection ---

class FusionEngine:
    def fuse(self, 
             text_probs: Optional[np.ndarray] = None,
             audio_probs: Optional[np.ndarray] = None,
             visual_probs: Optional[np.ndarray] = None,
             audio_confidence: float = 1.0,
             modalities: List[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Dynamic Late Fusion with selectable modalities.
        Allows user to choose which modalities to use.
        
        Args:
            text_probs: Text emotion probabilities
            audio_probs: Audio emotion probabilities
            visual_probs: Visual emotion probabilities
            audio_confidence: Confidence score for audio
            modalities: List of modalities to include ['text', 'audio', 'visual']
        """
        if modalities is None:
            modalities = ['text', 'audio', 'visual']
        
        # Default fallback
        default_probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Collect available modalities
        probs_dict = {}
        if 'text' in modalities and text_probs is not None:
            probs_dict['text'] = text_probs
        if 'audio' in modalities and audio_probs is not None:
            probs_dict['audio'] = audio_probs
        if 'visual' in modalities and visual_probs is not None:
            probs_dict['visual'] = visual_probs
        
        if not probs_dict:
            return default_probs, {"Text": 0.0, "Audio": 0.0, "Visual": 0.0}
        
        # Adaptive weighting based on modality confidence
        num_modalities = len(probs_dict)
        
        if num_modalities == 1:
            # Single modality
            modality = list(probs_dict.keys())[0]
            weights = {
                "Text": 1.0 if modality == "text" else 0.0,
                "Audio": 1.0 if modality == "audio" else 0.0,
                "Visual": 1.0 if modality == "visual" else 0.0
            }
        elif num_modalities == 2:
            # Two modalities
            if 'audio' in probs_dict and audio_confidence < 0.5:
                # Reduce audio weight if noisy
                if 'text' in probs_dict:
                    weights = {"Text": 0.7, "Audio": 0.3, "Visual": 0.0}
                elif 'visual' in probs_dict:
                    weights = {"Text": 0.0, "Audio": 0.3, "Visual": 0.7}
            else:
                # Equal weighting
                weight_value = 0.5
                weights = {
                    "Text": weight_value if 'text' in probs_dict else 0.0,
                    "Audio": weight_value if 'audio' in probs_dict else 0.0,
                    "Visual": weight_value if 'visual' in probs_dict else 0.0
                }
        else:
            # Three modalities
            if audio_confidence < 0.5:
                weights = {"Text": 0.45, "Audio": 0.15, "Visual": 0.4}
            else:
                weights = {"Text": 0.35, "Audio": 0.35, "Visual": 0.3}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Late Fusion (Weighted Sum)
        fused_probs = np.zeros(4)
        for modality, prob in probs_dict.items():
            fused_probs += weights[modality.capitalize()] * prob
        
        # Normalize
        fused_probs = fused_probs / (np.sum(fused_probs) + 1e-6)
        
        return fused_probs, weights


# Instantiate pipelines
text_pipe = TextModality()
audio_pipe = AudioModality()
visual_pipe = VisualModality()
fusion_engine = FusionEngine()


# --- Pydantic Models ---

class AnalysisResponse(BaseModel):
    overall_emotion: str
    emotion_probabilities: Dict[str, float]
    dynamic_weights: Dict[str, float]
    text_probs: Optional[Dict[str, float]] = None
    audio_probs: Optional[Dict[str, float]] = None
    audio_transcription: Optional[str] = None
    visual_probs: Optional[Dict[str, float]] = None
    modalities_used: List[str] = []


class BatchEvaluationRequest(BaseModel):
    dataset_path: str


class EvaluationItemResult(BaseModel):
    id: str
    ground_truth: str
    predicted: str
    is_correct: bool
    fused_probs: Dict[str, float]
    text_probs: Optional[Dict[str, float]] = None
    audio_probs: Optional[Dict[str, float]] = None
    visual_probs: Optional[Dict[str, float]] = None


class BatchEvaluationResponse(BaseModel):
    overall_accuracy: float
    text_accuracy: Optional[float] = None
    audio_accuracy: Optional[float] = None
    visual_accuracy: Optional[float] = None
    total_items: int
    results: List[EvaluationItemResult]


# --- API Endpoints ---

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_multimodal(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    language: Optional[str] = Form("en"),
    use_text: bool = Form(True),
    use_audio: bool = Form(True),
    use_visual: bool = Form(True),
    db: Session = Depends(get_db)
):
    """
    Main endpoint for multimodal emotion recognition.
    Accepts text, audio, and image files with modality selection.
    
    Query parameters:
    - use_text: Include text modality
    - use_audio: Include audio modality
    - use_visual: Include visual modality
    """
    logger.info(f"Received request: text='{text}', audio={audio is not None}, image={image is not None}")
    logger.info(f"Modalities: text={use_text}, audio={use_audio}, visual={use_visual}")
    
    try:
        modalities_used = []
        text_probs = None
        audio_probs = None
        audio_transcription = None
        visual_probs = None
        
        # 1. Process Text Modality
        if use_text and text:
            text_probs, text_metadata = text_pipe.analyze(text.strip() if text else "", language)
            modalities_used.append("text")
            logger.info(f"Text analysis: {text_metadata}")
        elif use_text:
            text_probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        # 2. Process Audio Modality
        if use_audio and audio:
            audio_bytes = await audio.read()
            audio_probs, audio_confidence, transcription = audio_pipe.analyze(audio_bytes)
            audio_transcription = transcription
            modalities_used.append("audio")
            logger.info(f"Audio analysis: confidence={audio_confidence}, transcription='{transcription}'")
        elif use_audio:
            audio_probs = np.array([0.25, 0.25, 0.25, 0.25])
            audio_confidence = 1.0
        
        # 3. Process Visual Modality
        if use_visual and image:
            image_bytes = await image.read()
            visual_probs = visual_pipe.analyze(image_bytes)
            modalities_used.append("visual")
            logger.info("Visual analysis complete")
        elif use_visual:
            visual_probs = np.array([0.25, 0.25, 0.25, 0.25])
        
        # 4. Dynamic Late Fusion with selected modalities
        selected_modalities = []
        if use_text and text_probs is not None:
            selected_modalities.append("text")
        if use_audio and audio_probs is not None:
            selected_modalities.append("audio")
        if use_visual and visual_probs is not None:
            selected_modalities.append("visual")
        
        fused_probs, weights = fusion_engine.fuse(
            text_probs=text_probs if use_text else None,
            audio_probs=audio_probs if use_audio else None,
            visual_probs=visual_probs if use_visual else None,
            audio_confidence=audio_confidence if use_audio else 1.0,
            modalities=selected_modalities
        )
        
        # Determine overall emotion
        max_idx = np.argmax(fused_probs)
        overall_emotion = EMOTIONS[max_idx]
        
        # Format response
        response = AnalysisResponse(
            overall_emotion=overall_emotion,
            emotion_probabilities={EMOTIONS[i]: float(fused_probs[i]) for i in range(4)},
            dynamic_weights=weights,
            text_probs={EMOTIONS[i]: float(text_probs[i]) for i in range(4)} if text_probs is not None else None,
            audio_probs={EMOTIONS[i]: float(audio_probs[i]) for i in range(4)} if audio_probs is not None else None,
            audio_transcription=audio_transcription,
            visual_probs={EMOTIONS[i]: float(visual_probs[i]) for i in range(4)} if visual_probs is not None else None,
            modalities_used=modalities_used
        )
        
        # Log to Database
        try:
            db_log = models.PredictionLog(
                text_input=text,
                predicted_emotion=overall_emotion,
                happy_prob=float(fused_probs[0]),
                sad_prob=float(fused_probs[1]),
                angry_prob=float(fused_probs[2]),
                neutral_prob=float(fused_probs[3])
            )
            db.add(db_log)
            db.commit()
        except Exception as db_e:
            logger.error(f"Failed to log prediction to DB: {db_e}")
            
        return response
    except Exception as e:
        logger.error(f"Internal server error during analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference pipeline failed: {str(e)}")


@app.post("/evaluate_batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(
    request: BatchEvaluationRequest,
    use_text: bool = Query(True),
    use_audio: bool = Query(True),
    use_visual: bool = Query(True),
    db: Session = Depends(get_db)
):
    """
    Batch evaluation endpoint with modality selection.
    """
    dataset_path = request.dataset_path
    csv_path = os.path.join(dataset_path, "dataset.csv")
    
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail=f"dataset.csv not found in {dataset_path}")
        
    results = []
    correct_fused = 0
    correct_text = 0
    correct_audio = 0
    correct_visual = 0
    total = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = row.get("id", str(total))
                text = row.get("text", "")
                audio_rel_path = row.get("audio_path", "")
                image_rel_path = row.get("image_path", "")
                ground_truth = row.get("ground_truth", "").strip().capitalize()
                
                # Load media if specified
                audio_bytes = None
                if audio_rel_path and use_audio:
                    a_path = os.path.join(dataset_path, audio_rel_path)
                    if os.path.exists(a_path):
                        with open(a_path, "rb") as af:
                            audio_bytes = af.read()
                            
                image_bytes = None
                if image_rel_path and use_visual:
                    i_path = os.path.join(dataset_path, image_rel_path)
                    if os.path.exists(i_path):
                        with open(i_path, "rb") as imf:
                            image_bytes = imf.read()
                
                # Process Modalities
                text_probs = None
                if use_text and text:
                    text_probs, _ = text_pipe.analyze(text)
                
                audio_probs = None
                audio_confidence = 1.0
                if use_audio and audio_bytes:
                    audio_probs, audio_confidence, _ = audio_pipe.analyze(audio_bytes)
                
                visual_probs = None
                if use_visual and image_bytes:
                    visual_probs = visual_pipe.analyze(image_bytes)
                
                # Fuse
                selected_modalities = []
                if use_text and text_probs is not None:
                    selected_modalities.append("text")
                if use_audio and audio_probs is not None:
                    selected_modalities.append("audio")
                if use_visual and visual_probs is not None:
                    selected_modalities.append("visual")
                
                fused_probs, weights = fusion_engine.fuse(
                    text_probs=text_probs,
                    audio_probs=audio_probs,
                    visual_probs=visual_probs,
                    audio_confidence=audio_confidence,
                    modalities=selected_modalities
                )
                
                # Predictions
                fused_pred = EMOTIONS[np.argmax(fused_probs)]
                text_pred = EMOTIONS[np.argmax(text_probs)] if text_probs is not None else None
                audio_pred = EMOTIONS[np.argmax(audio_probs)] if audio_probs is not None else None
                visual_pred = EMOTIONS[np.argmax(visual_probs)] if visual_probs is not None else None
                
                # Accuracy tracking
                is_correct = (fused_pred == ground_truth)
                if is_correct:
                    correct_fused += 1
                if text_pred and text_pred == ground_truth:
                    correct_text += 1
                if audio_pred and audio_pred == ground_truth:
                    correct_audio += 1
                if visual_pred and visual_pred == ground_truth:
                    correct_visual += 1
                total += 1
                
                results.append(EvaluationItemResult(
                    id=item_id,
                    ground_truth=ground_truth,
                    predicted=fused_pred,
                    is_correct=is_correct,
                    fused_probs={EMOTIONS[i]: float(fused_probs[i]) for i in range(4)},
                    text_probs={EMOTIONS[i]: float(text_probs[i]) for i in range(4)} if text_probs is not None else None,
                    audio_probs={EMOTIONS[i]: float(audio_probs[i]) for i in range(4)} if audio_probs is not None else None,
                    visual_probs={EMOTIONS[i]: float(visual_probs[i]) for i in range(4)} if visual_probs is not None else None
                ))
                
        response = BatchEvaluationResponse(
            overall_accuracy=correct_fused / max(1, total) * 100,
            text_accuracy=correct_text / max(1, total) * 100 if use_text else None,
            audio_accuracy=correct_audio / max(1, total) * 100 if use_audio else None,
            visual_accuracy=correct_visual / max(1, total) * 100 if use_visual else None,
            total_items=total,
            results=results
        )
        
        # Log to Database
        try:
            db_log = models.BatchEvaluationLog(
                dataset_path=dataset_path,
                total_items=total,
                overall_accuracy=response.overall_accuracy,
                text_accuracy=response.text_accuracy or 0.0,
                audio_accuracy=response.audio_accuracy or 0.0,
                visual_accuracy=response.visual_accuracy or 0.0
            )
            db.add(db_log)
            db.commit()
        except Exception as db_e:
            logger.error(f"Failed to log batch evaluation to DB: {db_e}")
            
        return response
    except Exception as e:
        logger.error(f"Batch evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process batch: {str(e)}")


@app.get("/admin/logs")
def get_admin_logs(db: Session = Depends(get_db), limit: int = 10):
    """Retrieve recent prediction and batch evaluation logs"""
    predictions = db.query(models.PredictionLog).order_by(models.PredictionLog.timestamp.desc()).limit(limit).all()
    batch_logs = db.query(models.BatchEvaluationLog).order_by(models.BatchEvaluationLog.timestamp.desc()).limit(limit).all()
    return {"predictions": predictions, "batch_logs": batch_logs}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "whisper_available": WHISPER_AVAILABLE,
        "transformer_available": TRANSFORMERS_AVAILABLE,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "speech_recognition_available": SR_AVAILABLE,
        "translator_available": TRANSLATOR_AVAILABLE,
        "nltk_available": NLTK_AVAILABLE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
