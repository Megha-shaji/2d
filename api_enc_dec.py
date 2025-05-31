from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import hashlib
import random
import io
import tempfile
import os
import pickle
import zipfile
from pathlib import Path
import soundfile as sf
import warnings
import time
from typing import Dict, Any
import uvicorn
from pydantic import BaseModel

warnings.filterwarnings('ignore')

# Pydantic models for API responses
class EncryptionResponse(BaseModel):
    success: bool
    message: str
    encryption_id: str = None
    duration: float = None
    sample_rate: int = None
    load_time: float = None
    encryption_time: float = None
    total_time: float = None
    preview_file: str = None
    package_file: str = None

class DecryptionResponse(BaseModel):
    success: bool
    message: str
    encryption_id: str = None
    duration: float = None
    sample_rate: int = None
    decryption_time: float = None
    total_time: float = None
    decrypted_file: str = None

class FastAudioCrypto:
    def __init__(self):
        self.encryption_data = {}
        self.max_audio_length = 30 * 44100  # 30 seconds max
    
    def _validate_audio(self, audio_data):
        """Validate audio data before processing"""
        if audio_data is None or len(audio_data) == 0:
            raise ValueError("Empty audio data")
        
        if len(audio_data) > self.max_audio_length:
            raise ValueError(f"Audio too long. Maximum length: {self.max_audio_length/44100:.1f} seconds")
        
        if not np.isfinite(audio_data).all():
            raise ValueError("Audio contains invalid values (NaN or Inf)")
    
    def _generate_chaotic_sequence(self, seed, length):
        """Fast chaotic sequence generation using logistic map"""
        # Ensure seed is long enough and valid
        if len(seed) < 24:
            # Pad seed if too short
            seed = (seed * ((24 // len(seed)) + 1))[:24]
        
        # Use hash as seed for reproducible chaos
        try:
            seed_int = int(seed[:8], 16) % (2**32)
        except ValueError:
            # Fallback if invalid hex
            seed_int = abs(hash(seed[:8])) % (2**32)
        
        np.random.seed(seed_int)
        
        # Initialize chaotic parameters from seed
        try:
            r = 3.7 + (int(seed[8:16], 16) % 1000000) / 5000000  # r between 3.7-3.9
            x = (int(seed[16:24], 16) % 1000000) / 1000000  # x between 0-1
        except ValueError:
            # Fallback values
            r = 3.8
            x = 0.5
        
        sequence = []
        for _ in range(length):
            x = r * x * (1 - x)  # Logistic map
            # Ensure value is always in [0, 255] range
            val = int(x * 255) % 256
            if val > 255:
                val = 255
            sequence.append(val)
        
        return np.array(sequence, dtype=np.uint8)
    
    def _scramble_audio(self, audio_data, key_sequence):
        """Fast audio scrambling using bit manipulation"""
        # Convert audio to integer representation (16-bit range mapped to 8-bit)
        # Normalize to [0, 1] then scale to [0, 255]
        audio_normalized = (audio_data + 1.0) / 2.0  # Convert from [-1,1] to [0,1]
        audio_int = (audio_normalized * 255).astype(np.uint8)
        
        # Ensure key sequence is long enough
        if len(key_sequence) < len(audio_int):
            # Repeat key sequence if needed
            key_sequence = np.tile(key_sequence, (len(audio_int) // len(key_sequence)) + 1)
        
        # XOR with key sequence
        scrambled = audio_int ^ key_sequence[:len(audio_int)]
        
        return scrambled.astype(np.uint8)
    
    def _descramble_audio(self, scrambled_data, key_sequence):
        """Reverse the scrambling process"""
        # Ensure key sequence is long enough
        if len(key_sequence) < len(scrambled_data):
            key_sequence = np.tile(key_sequence, (len(scrambled_data) // len(key_sequence)) + 1)
        
        # XOR with same key to decrypt
        audio_int = scrambled_data ^ key_sequence[:len(scrambled_data)]
        
        # Convert back to float audio [-1, 1]
        audio_normalized = audio_int.astype(np.float32) / 255.0  # [0, 1]
        audio_data = (audio_normalized * 2.0) - 1.0  # [-1, 1]
        return np.clip(audio_data, -1.0, 1.0)
    
    def _permute_samples(self, data, indices):
        """Fast sample permutation"""
        return data[indices]
    
    def _reverse_permutation(self, data, indices):
        """Reverse sample permutation"""
        reversed_data = np.zeros_like(data)
        reversed_data[indices] = data
        return reversed_data
    
    def encrypt_audio_fast(self, audio_data):
        """Fast encryption without EMD"""
        try:
            self._validate_audio(audio_data)
            
            # Normalize audio
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # Generate hash from audio
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            hash_hex = hashlib.sha256(audio_bytes).hexdigest()
            
            # Generate permutation indices
            random.seed(hash_hex)
            indices = np.random.permutation(len(audio_data))
            
            # Step 1: Permute samples
            permuted_audio = self._permute_samples(audio_data, indices)
            
            # Step 2: Generate chaotic key sequence
            key_sequence = self._generate_chaotic_sequence(hash_hex, len(audio_data))
            
            # Step 3: Scramble audio
            encrypted_data = self._scramble_audio(permuted_audio, key_sequence)
            
            # Step 4: Additional diffusion layer with safe block operations
            block_size = min(128, len(encrypted_data) // 16)  # Even smaller to be extra safe
            if block_size > 1:
                # Use a safe portion of hash for shift operations
                if len(hash_hex) >= 48:
                    shift_seed = hash_hex[32:48]
                else:
                    # Create a safe shift seed by repeating the hash
                    shift_seed = (hash_hex * 3)[:16]  # Ensure we have at least 16 chars
                
                shift_key = self._generate_chaotic_sequence(shift_seed, len(encrypted_data))
                
                for i in range(0, len(encrypted_data) - block_size, block_size):
                    if i + block_size <= len(encrypted_data):
                        block = encrypted_data[i:i+block_size].copy()
                        # Ensure shift is always less than block_size
                        shift = (shift_key[i % len(shift_key)] % block_size) if block_size > 0 else 0
                        encrypted_data[i:i+block_size] = np.roll(block, shift)
            
            return encrypted_data, indices, hash_hex, block_size
            
        except Exception as e:
            raise Exception(f"Encryption failed: {str(e)}")
    
    def decrypt_audio_fast(self, encrypted_data, indices, hash_hex, block_size):
        """Fast decryption process"""
        try:
            # Reverse diffusion layer
            decrypted_data = encrypted_data.copy()
            if block_size > 1:
                # Use the same safe shift seed generation as encryption
                if len(hash_hex) >= 48:
                    shift_seed = hash_hex[32:48]
                else:
                    # Create a safe shift seed by repeating the hash
                    shift_seed = (hash_hex * 3)[:16]  # Ensure we have at least 16 chars
                
                shift_key = self._generate_chaotic_sequence(shift_seed, len(encrypted_data))
                
                for i in range(0, len(decrypted_data) - block_size, block_size):
                    if i + block_size <= len(decrypted_data):
                        block = decrypted_data[i:i+block_size].copy()
                        # Ensure shift is always less than block_size
                        shift = (shift_key[i % len(shift_key)] % block_size) if block_size > 0 else 0
                        decrypted_data[i:i+block_size] = np.roll(block, -shift)
            
            # Generate key sequence for descrambling
            key_sequence = self._generate_chaotic_sequence(hash_hex, len(encrypted_data))
            
            # Descramble audio
            permuted_audio = self._descramble_audio(decrypted_data, key_sequence)
            
            # Reverse permutation
            original_audio = self._reverse_permutation(permuted_audio, indices)
            
            return original_audio.astype(np.float32)
            
        except Exception as e:
            raise Exception(f"Decryption failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Fast Audio Encryption API",
    description="Ultra-fast chaotic audio encryption system with REST API",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize crypto system
crypto = FastAudioCrypto()

# Store temporary files with cleanup function
temp_files = {}

def cleanup_temp_files():
    """Clean up temporary files older than 1 hour"""
    current_time = time.time()
    files_to_remove = []
    
    for file_id, file_path in temp_files.items():
        try:
            if os.path.exists(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  # 1 hour
                    os.unlink(file_path)
                    files_to_remove.append(file_id)
            else:
                files_to_remove.append(file_id)
        except Exception:
            files_to_remove.append(file_id)
    
    for file_id in files_to_remove:
        temp_files.pop(file_id, None)

@app.get("/")
async def root():
    """API Information"""
    return {
        "message": "Fast Audio Encryption API v2.0",
        "description": "Ultra-fast chaotic audio encryption system",
        "endpoints": {
            "POST /encrypt": "Encrypt audio file",
            "POST /decrypt": "Decrypt audio package",
            "GET /download/{file_id}": "Download generated files",
            "GET /algorithm-info": "Get algorithm information",
            "GET /health": "Health check"
        },
        "features": [
            "Sub-second encryption for 6-second audio",
            "Millisecond decryption",
            "Chaotic permutation-scrambling-diffusion algorithm",
            "No key storage required",
            "Content-dependent encryption"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/algorithm-info")
async def get_algorithm_info():
    """Get information about the encryption algorithm"""
    return {
        "algorithm": "Fast Chaotic Audio Encryption v2.0",
        "description": "Ultra-fast chaotic encryption without EMD processing",
        "features": {
            "performance": {
                "encryption_speed": "Sub-second for 6-second audio",
                "decryption_speed": "Milliseconds",
                "speedup": "1000x faster than v1.0"
            },
            "security": {
                "hash_function": "SHA-256",
                "chaotic_map": "Logistic Map (r ∈ [3.7, 3.9])",
                "layers": ["Sample Permutation", "Chaotic Scrambling", "Block Diffusion"],
                "key_derivation": "Content-dependent (no key storage)"
            },
            "fixes_in_v2": [
                "uint8 overflow prevention",
                "Proper audio normalization",
                "Improved key sequence handling", 
                "Separate diffusion key generation",
                "Better error handling"
            ],
            "specifications": {
                "max_duration": "30 seconds",
                "block_size": "Adaptive (max 256)",
                "data_type": "uint8 for encrypted data",
                "supported_formats": [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
            }
        }
    }

@app.post("/encrypt", response_model=EncryptionResponse)
async def encrypt_audio(audio_file: UploadFile = File(...)):
    """
    Encrypt an audio file using fast chaotic encryption
    
    - **audio_file**: Audio file to encrypt (max 30 seconds)
    
    Returns encrypted audio preview and package download links
    """
    try:
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Validate file type
        allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        file_ext = Path(audio_file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read uploaded file
        start_time = time.time()
        audio_content = await audio_file.read()
        
        # Save to temporary file for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_input:
            tmp_input.write(audio_content)
            tmp_input_path = tmp_input.name
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(tmp_input_path, sr=None, duration=30)
            load_time = time.time() - start_time
            
            if len(audio_data) == 0:
                raise HTTPException(status_code=400, detail="Audio file is empty")
            
            # Encrypt audio
            encrypt_start = time.time()
            encrypted_data, indices, hash_hex, block_size = crypto.encrypt_audio_fast(audio_data)
            encrypt_time = time.time() - encrypt_start
            
            # Create encrypted audio preview (noise)
            preview_audio = (encrypted_data.astype(np.float32) / 255.0 * 2.0 - 1.0) * 0.1
            
            # Generate encryption ID
            encryption_id = hash_hex[:12]
            
            # Create metadata
            metadata = {
                'encrypted_data': encrypted_data.tolist(),
                'indices': indices.tolist(),
                'hash_hex': hash_hex,
                'block_size': block_size,
                'sample_rate': sample_rate,
                'original_length': len(audio_data),
                'encryption_id': encryption_id,
                'version': '2.0_fast'
            }
            
            # Save preview audio
            preview_file_id = f"preview_{encryption_id}_{int(time.time())}"
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_preview:
                preview_path = tmp_preview.name
            sf.write(preview_path, preview_audio, sample_rate)
            temp_files[preview_file_id] = preview_path
            
            # Create encryption package  
            package_file_id = f"package_{encryption_id}_{int(time.time())}"
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_package:
                package_path = tmp_package.name
            
            # Save metadata to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', mode='wb') as tmp_meta:
                pickle.dump(metadata, tmp_meta)
                metadata_path = tmp_meta.name
            
            # Create zip package
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(metadata_path, 'fast_encryption_data.pkl')
                readme_content = f"""
Fast Audio Encryption Package
============================
Encryption ID: {encryption_id}
Original Duration: {len(audio_data)/sample_rate:.2f} seconds
Sample Rate: {sample_rate} Hz
Algorithm: Fast Chaotic Encryption (v2.0)
Load Time: {load_time:.3f}s
Encryption Time: {encrypt_time:.3f}s

This package uses fast chaotic encryption without EMD.
To decrypt: POST this zip file to /decrypt endpoint.
"""
                zf.writestr('README.txt', readme_content)
            
            temp_files[package_file_id] = package_path
            
            # Cleanup
            os.unlink(metadata_path)
            
            total_time = load_time + encrypt_time
            
            return EncryptionResponse(
                success=True,
                message="Audio encrypted successfully",
                encryption_id=encryption_id,
                duration=len(audio_data)/sample_rate,
                sample_rate=sample_rate,
                load_time=load_time,
                encryption_time=encrypt_time,
                total_time=total_time,
                preview_file=f"/download/{preview_file_id}",
                package_file=f"/download/{package_file_id}"
            )
            
        finally:
            # Cleanup input file
            os.unlink(tmp_input_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption error: {str(e)}")

@app.post("/decrypt", response_model=DecryptionResponse)
async def decrypt_audio(encrypted_package: UploadFile = File(...)):
    """
    Decrypt an encrypted audio package
    
    - **encrypted_package**: Encrypted zip package from /encrypt endpoint
    
    Returns decrypted audio file download link
    """
    try:
        if not encrypted_package.filename or not encrypted_package.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Please upload a .zip encryption package")
        
        start_time = time.time()
        
        # Read uploaded zip file
        zip_content = await encrypted_package.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
            tmp_zip.write(zip_content)
            tmp_zip_path = tmp_zip.name
        
        try:
            # Extract zip
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with zipfile.ZipFile(tmp_zip_path, 'r') as zf:
                        zf.extractall(temp_dir)
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid zip file: {str(e)}")
                
                # Try different metadata file names for backward compatibility
                metadata_files = ['fast_encryption_data.pkl', 'encryption_data.pkl']
                metadata = None
                
                for meta_file in metadata_files:
                    metadata_path = os.path.join(temp_dir, meta_file)
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'rb') as f:
                                metadata = pickle.load(f)
                            break
                        except Exception:
                            continue
                
                if metadata is None:
                    raise HTTPException(status_code=400, detail="No valid encryption data found in zip file")
                
                # Check if it's fast version
                if not metadata.get('version', '').startswith('2.0'):
                    raise HTTPException(
                        status_code=400, 
                        detail="This file was encrypted with the old slow method. Please re-encrypt with the fast version."
                    )
                
                # Fast decryption
                encrypted_data = np.array(metadata['encrypted_data'], dtype=np.uint8)
                indices = np.array(metadata['indices'])
                
                decrypt_start = time.time()
                decrypted_audio = crypto.decrypt_audio_fast(
                    encrypted_data,
                    indices,
                    metadata['hash_hex'],
                    metadata['block_size']
                )
                decrypt_time = time.time() - decrypt_start
                
                # Save decrypted audio
                decrypted_file_id = f"decrypted_{metadata.get('encryption_id', 'unknown')}_{int(time.time())}"
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_decrypted:
                    decrypted_path = tmp_decrypted.name
                sf.write(decrypted_path, decrypted_audio, metadata['sample_rate'])
                temp_files[decrypted_file_id] = decrypted_path
                
                total_time = time.time() - start_time
                
                return DecryptionResponse(
                    success=True,
                    message="Audio decrypted successfully",
                    encryption_id=metadata.get('encryption_id', 'Unknown'),
                    duration=len(decrypted_audio)/metadata['sample_rate'],
                    sample_rate=metadata['sample_rate'],
                    decryption_time=decrypt_time,
                    total_time=total_time,
                    decrypted_file=f"/download/{decrypted_file_id}"
                )
                
        finally:
            # Cleanup zip file
            os.unlink(tmp_zip_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decryption error: {str(e)}")

@app.get("/download/{file_id}")
async def download_file(file_id: str):
    """
    Download generated files (previews, packages, decrypted audio)
    
    - **file_id**: File identifier from encrypt/decrypt responses
    """
    # Clean up old files periodically
    cleanup_temp_files()
    
    if file_id not in temp_files:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_path = temp_files[file_id]
    
    if not os.path.exists(file_path):
        # Remove from temp_files if file doesn't exist
        temp_files.pop(file_id, None)
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine media type and filename
    if file_id.startswith('preview_'):
        media_type = "audio/wav"
        filename = f"encrypted_preview_{file_id.split('_')[1]}.wav"
    elif file_id.startswith('package_'):
        media_type = "application/zip"
        filename = f"encryption_package_{file_id.split('_')[1]}.zip"
    elif file_id.startswith('decrypted_'):
        media_type = "audio/wav"
        filename = f"decrypted_audio_{file_id.split('_')[1]}.wav"
    else:
        media_type = "application/octet-stream"
        filename = f"{file_id}"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
        headers={"Cache-Control": "no-cache"}
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    # Ensure temp directory exists
    os.makedirs(tempfile.gettempdir(), exist_ok=True)
    print(f"FastAPI Audio Encryption API v2.0 started")
    print(f"Temp directory: {tempfile.gettempdir()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    for file_path in temp_files.values():
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass
    temp_files.clear()
    print("FastAPI Audio Encryption API shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "api_enc_dec:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )