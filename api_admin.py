from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any
import secrets
import uvicorn
from pydantic import BaseModel
import aiofiles
import shutil
import asyncio

warnings.filterwarnings('ignore')

# Pydantic models for API requests/responses
class AdminLoginRequest(BaseModel):
    password: str

class KeySearchRequest(BaseModel):
    encryption_id: str

class KeyDeleteRequest(BaseModel):
    encryption_id: str

class EncryptionResponse(BaseModel):
    success: bool
    message: str
    encryption_id: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    load_time: Optional[float] = None
    encryption_time: Optional[float] = None
    total_time: Optional[float] = None
    download_url: Optional[str] = None  # Added download URL

class DecryptionResponse(BaseModel):
    success: bool
    message: str
    encryption_id: Optional[str] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    decryption_time: Optional[float] = None
    total_time: Optional[float] = None
    download_url: Optional[str] = None  # Added download URL

class AdminStatsResponse(BaseModel):
    total_encryptions: int
    total_decryptions: int
    total_keys_stored: int
    uptime: str
    recent_activity: int
    system_start_time: str

class KeyInfo(BaseModel):
    encryption_id: str
    created_at: str
    access_count: int
    last_accessed: Optional[str]
    user_ip: str

class AdminKeyManager:
    def __init__(self):
        self.admin_password = "admin123"  # Change this in production
        self.encryption_keys = {}  # Store all encryption keys
        self.user_sessions = {}  # Track user sessions
        self.encryption_history = []  # Track all encryptions
        self.system_stats = {
            'total_encryptions': 0,
            'total_decryptions': 0,
            'total_users': 0,
            'system_start_time': datetime.datetime.now()
        }
        
        # Ensure temp directory exists
        self.temp_dir = Path("/tmp")
        self.temp_dir.mkdir(exist_ok=True)
        
    def authenticate_admin(self, password: str) -> bool:
        """Authenticate admin access"""
        return password == self.admin_password
    
    def store_encryption_key(self, encryption_id: str, key_data: dict, user_info: dict = None):
        """Store encryption key and metadata"""
        self.encryption_keys[encryption_id] = {
            'key_data': key_data,
            'created_at': datetime.datetime.now().isoformat(),
            'user_info': user_info or {},
            'access_count': 0,
            'last_accessed': None
        }
        
        # Add to history
        self.encryption_history.append({
            'action': 'encryption',
            'encryption_id': encryption_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'user_info': user_info or {}
        })
        
        self.system_stats['total_encryptions'] += 1
    
    def get_encryption_key(self, encryption_id: str) -> Optional[dict]:
        """Retrieve encryption key by ID"""
        if encryption_id in self.encryption_keys:
            self.encryption_keys[encryption_id]['access_count'] += 1
            self.encryption_keys[encryption_id]['last_accessed'] = datetime.datetime.now().isoformat()
            return self.encryption_keys[encryption_id]['key_data']
        return None
    
    def list_all_keys(self) -> List[dict]:
        """List all stored encryption keys"""
        keys_list = []
        for enc_id, data in self.encryption_keys.items():
            keys_list.append({
                'encryption_id': enc_id,
                'created_at': data['created_at'],
                'access_count': data['access_count'],
                'last_accessed': data['last_accessed'],
                'user_info': data['user_info']
            })
        return sorted(keys_list, key=lambda x: x['created_at'], reverse=True)
    
    def delete_key(self, encryption_id: str) -> bool:
        """Delete an encryption key"""
        if encryption_id in self.encryption_keys:
            del self.encryption_keys[encryption_id]
            return True
        return False
    
    def get_system_stats(self) -> dict:
        """Get system statistics"""
        uptime = datetime.datetime.now() - self.system_stats['system_start_time']
        return {
            **self.system_stats,
            'uptime': str(uptime),
            'total_keys_stored': len(self.encryption_keys),
            'recent_activity': len([h for h in self.encryption_history 
                                 if datetime.datetime.fromisoformat(h['timestamp']) > 
                                 datetime.datetime.now() - datetime.timedelta(hours=24)])
        }
    
    def export_keys(self) -> str:
        """Export all keys to JSON"""
        export_data = {
            'keys': self.encryption_keys,
            'history': self.encryption_history,
            'stats': self.get_system_stats(),
            'export_timestamp': datetime.datetime.now().isoformat()
        }
        return json.dumps(export_data, indent=2)
    
    def import_keys(self, json_data: str) -> bool:
        """Import keys from JSON"""
        try:
            data = json.loads(json_data)
            if 'keys' in data:
                self.encryption_keys.update(data['keys'])
            if 'history' in data:
                self.encryption_history.extend(data['history'])
            return True
        except Exception:
            return False

class FastAudioCrypto:
    def __init__(self, admin_manager: AdminKeyManager):
        self.encryption_data = {}
        self.max_audio_length = 30 * 44100  # 30 seconds max
        self.admin_manager = admin_manager
    
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
    
    async def encrypt_audio_fast(self, audio_data, user_info=None):
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
            
            # Generate encryption ID
            encryption_id = hash_hex[:12]
            
            # Store key in admin manager
            key_data = {
                'encrypted_data': encrypted_data.tolist(),
                'indices': indices.tolist(),
                'hash_hex': hash_hex,
                'block_size': block_size,
                'original_length': len(audio_data)
            }
            
            self.admin_manager.store_encryption_key(encryption_id, key_data, user_info)
            
            return encrypted_data, indices, hash_hex, block_size, encryption_id
            
        except Exception as e:
            raise Exception(f"Encryption failed: {str(e)}")
    
    async def decrypt_audio_fast(self, encrypted_data, indices, hash_hex, block_size):
        """Fast decryption process"""
        try:
            # Update admin stats
            self.admin_manager.system_stats['total_decryptions'] += 1
            
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

# Initialize components
admin_manager = AdminKeyManager()
crypto = FastAudioCrypto(admin_manager)

# FastAPI app setup
app = FastAPI(
    title="Fast Audio Encryption API",
    description="Ultra-fast chaotic audio encryption with admin management",
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

# Security
security = HTTPBearer()

def verify_admin_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin authentication token"""
    token = credentials.credentials
    if not admin_manager.authenticate_admin(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

# API Routes

@app.get("/")
async def root():
    """API status endpoint"""
    return {
        "message": "Fast Audio Encryption API",
        "version": "2.0.0",
        "status": "active",
        "features": ["audio_encryption", "admin_panel", "key_management"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = admin_manager.get_system_stats()
    return {
        "status": "healthy",
        "uptime": stats["uptime"],
        "total_encryptions": stats["total_encryptions"],
        "total_decryptions": stats["total_decryptions"]
    }

@app.post("/encrypt", response_model=EncryptionResponse)
async def encrypt_audio(
    audio_file: UploadFile = File(..., description="Audio file to encrypt (max 30 seconds)")
):
    """Encrypt an audio file"""
    try:
        # Validate file type
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read uploaded file
        start_time = asyncio.get_event_loop().time()
        contents = await audio_file.read()
        
        # Save to temporary file for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Load audio
            load_start = asyncio.get_event_loop().time()
            audio_data, sample_rate = librosa.load(tmp_path, sr=None, duration=30)
            load_time = asyncio.get_event_loop().time() - load_start
            
            if len(audio_data) == 0:
                raise HTTPException(status_code=400, detail="Audio file is empty")
            
            # Encrypt audio
            encrypt_start = asyncio.get_event_loop().time()
            user_info = {'ip': 'api_client', 'timestamp': datetime.datetime.now().isoformat()}
            encrypted_data, indices, hash_hex, block_size, encryption_id = await crypto.encrypt_audio_fast(
                audio_data, user_info
            )
            encrypt_time = asyncio.get_event_loop().time() - encrypt_start
            
            # Create metadata
            metadata = {
                'encrypted_data': encrypted_data.tolist(),
                'indices': indices.tolist(),
                'hash_hex': hash_hex,
                'block_size': block_size,
                'sample_rate': sample_rate,
                'original_length': len(audio_data),
                'encryption_id': encryption_id,
                'version': '2.0_fast_admin_api'
            }
            
            # Ensure temp directory and create encrypted package
            os.makedirs(admin_manager.temp_dir, exist_ok=True)
            package_path = admin_manager.temp_dir / f"encrypted_{encryption_id}.zip"
            
            try:
                with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    # Save metadata using a temporary file
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as meta_tmp:
                        pickle.dump(metadata, meta_tmp)
                        meta_tmp.flush()
                        # Ensure file is closed before adding to zip
                        meta_tmp.close()
                        zf.write(meta_tmp.name, 'fast_encryption_data.pkl')
                        # Clean up temp file
                        os.unlink(meta_tmp.name)
                    
                    # Add README
                    readme_content = f"""
Fast Audio Encryption Package (API Version)
==========================================
Encryption ID: {encryption_id}
Original Duration: {len(audio_data)/sample_rate:.2f} seconds
Sample Rate: {sample_rate} Hz
Algorithm: Fast Chaotic Encryption (v2.0 API)
Load Time: {load_time:.3f}s
Encryption Time: {encrypt_time:.3f}s

This package uses fast chaotic encryption with admin key management.
Keys are automatically stored in the admin panel for recovery.
To decrypt: Use the /decrypt endpoint or upload in the web interface.
"""
                    zf.writestr('README.txt', readme_content)
                
                # Verify the zip file was created successfully
                if not package_path.exists():
                    raise Exception("Failed to create encrypted package")
                    
            except Exception as zip_error:
                # Clean up if zip creation failed
                if package_path.exists():
                    package_path.unlink()
                raise Exception(f"Failed to create encrypted package: {str(zip_error)}")
            
            total_time = asyncio.get_event_loop().time() - start_time
            
            return EncryptionResponse(
                success=True,
                message="Audio encrypted successfully",
                encryption_id=encryption_id,
                duration=len(audio_data)/sample_rate,
                sample_rate=sample_rate,
                load_time=load_time,
                encryption_time=encrypt_time,
                total_time=total_time,
                download_url=f"/download/{encryption_id}"
            )
            
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {str(e)}")

@app.get("/download/{encryption_id}")
async def download_encrypted_package(encryption_id: str):
    """Download encrypted package by ID"""
    package_path = admin_manager.temp_dir / f"encrypted_{encryption_id}.zip"
    if not package_path.exists():
        raise HTTPException(status_code=404, detail="Encrypted package not found")
    
    return FileResponse(
        str(package_path),
        media_type='application/zip',
        filename=f'encrypted_audio_{encryption_id}.zip'
    )

@app.post("/decrypt", response_model=DecryptionResponse)
async def decrypt_audio(
    encrypted_file: UploadFile = File(..., description="Encrypted zip package")
):
    """Decrypt an encrypted audio package"""
    try:
        start_time = asyncio.get_event_loop().time()
        
        # Read uploaded file
        contents = await encrypted_file.read()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        decrypted_path = None
        try:
            # Extract and decrypt
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    with zipfile.ZipFile(tmp_path, 'r') as zf:
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
                
                # Check if it's admin version
                if metadata.get('version', '').endswith('_admin') or metadata.get('version', '').endswith('_admin_api'):
                    # Try to get key from admin manager first
                    encryption_id = metadata.get('encryption_id')
                    admin_key = admin_manager.get_encryption_key(encryption_id) if encryption_id else None
                    
                    if admin_key:
                        # Use admin-stored key
                        encrypted_data = np.array(admin_key['encrypted_data'], dtype=np.uint8)
                        indices = np.array(admin_key['indices'])
                        hash_hex = admin_key['hash_hex']
                        block_size = admin_key['block_size']
                    else:
                        # Fallback to package data
                        encrypted_data = np.array(metadata['encrypted_data'], dtype=np.uint8)
                        indices = np.array(metadata['indices'])
                        hash_hex = metadata['hash_hex']
                        block_size = metadata['block_size']
                    
                    decrypt_start = asyncio.get_event_loop().time()
                    decrypted_audio = await crypto.decrypt_audio_fast(
                        encrypted_data, indices, hash_hex, block_size
                    )
                    decrypt_time = asyncio.get_event_loop().time() - decrypt_start
                    
                elif metadata.get('version', '').startswith('2.0'):
                    # Fast decryption
                    encrypted_data = np.array(metadata['encrypted_data'], dtype=np.uint8)
                    indices = np.array(metadata['indices'])
                    
                    decrypt_start = asyncio.get_event_loop().time()
                    decrypted_audio = await crypto.decrypt_audio_fast(
                        encrypted_data,
                        indices,
                        metadata['hash_hex'],
                        metadata['block_size']
                    )
                    decrypt_time = asyncio.get_event_loop().time() - decrypt_start
                else:
                    raise HTTPException(status_code=400, detail="Unsupported encryption version")
                
                # Save decrypted audio
                os.makedirs(admin_manager.temp_dir, exist_ok=True)
                decrypted_path = admin_manager.temp_dir / f"decrypted_{metadata.get('encryption_id', 'unknown')}.wav"
                sf.write(str(decrypted_path), decrypted_audio, metadata['sample_rate'])
                
                total_time = asyncio.get_event_loop().time() - start_time
                
                return DecryptionResponse(
                    success=True,
                    message="Audio decrypted successfully",
                    encryption_id=metadata.get('encryption_id', 'unknown'),
                    duration=len(decrypted_audio)/metadata['sample_rate'],
                    sample_rate=metadata['sample_rate'],
                    decryption_time=decrypt_time,
                    total_time=total_time,
                    download_url=f"/download-decrypted/{metadata.get('encryption_id', 'unknown')}"
                )
                
        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decryption failed: {str(e)}")

@app.get("/download-decrypted/{encryption_id}")
async def download_decrypted_audio(encryption_id: str):
    """Download decrypted audio by ID"""
    audio_path = admin_manager.temp_dir / f"decrypted_{encryption_id}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Decrypted audio not found")
    
    return FileResponse(
        str(audio_path),
        media_type='audio/wav',
        filename=f'decrypted_audio_{encryption_id}.wav'
    )

# Admin endpoints
@app.post("/admin/login")
async def admin_login(request: AdminLoginRequest):
    """Admin authentication"""
    if admin_manager.authenticate_admin(request.password):
        return {
            "success": True,
            "message": "Admin authenticated successfully",
            "token": request.password  # In production, use proper JWT tokens
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid admin password")

@app.get("/admin/stats", response_model=AdminStatsResponse)
async def get_admin_stats(admin_token: str = Depends(verify_admin_token)):
    """Get system statistics"""
    stats = admin_manager.get_system_stats()
    return AdminStatsResponse(
        total_encryptions=stats['total_encryptions'],
        total_decryptions=stats['total_decryptions'],
        total_keys_stored=stats['total_keys_stored'],
        uptime=stats['uptime'],
        recent_activity=stats['recent_activity'],
        system_start_time=stats['system_start_time'].isoformat()
    )

@app.get("/admin/keys", response_model=List[KeyInfo])
async def list_encryption_keys(
    limit: int = 20,
    admin_token: str = Depends(verify_admin_token)
):
    """List all encryption keys"""
    keys = admin_manager.list_all_keys()
    limited_keys = keys[:limit]
    
    return [
        KeyInfo(
            encryption_id=key['encryption_id'],
            created_at=key['created_at'],
            access_count=key['access_count'],
            last_accessed=key['last_accessed'],
            user_ip=key['user_info'].get('ip', 'unknown')
        )
        for key in limited_keys
    ]

@app.post("/admin/search-key")
async def search_encryption_key(
    request: KeySearchRequest,
    admin_token: str = Depends(verify_admin_token)
):
    """Search for specific encryption key"""
    key_data = admin_manager.get_encryption_key(request.encryption_id)
    if key_data:
        return {
            "success": True,
            "encryption_id": request.encryption_id,
            "original_length": key_data['original_length'],
            "hash_hex": key_data['hash_hex'][:32] + "...",
            "block_size": key_data['block_size']
        }
    else:
        raise HTTPException(status_code=404, detail=f"Key not found: {request.encryption_id}")
@app.delete("/admin/delete-key")
async def delete_encryption_key(
    request: KeyDeleteRequest,
    admin_token: str = Depends(verify_admin_token)
):
    """Delete an encryption key"""
    success = admin_manager.delete_key(request.encryption_id)
    if success:
        return {
            "success": True,
            "message": f"Key {request.encryption_id} deleted successfully"
        }
    else:
        raise HTTPException(status_code=404, detail=f"Key not found: {request.encryption_id}")

@app.get("/admin/export-keys")
async def export_all_keys(admin_token: str = Depends(verify_admin_token)):
    """Export all keys to JSON"""
    export_data = admin_manager.export_keys()
    
    # Create temporary file for export
    export_path = admin_manager.temp_dir / f"keys_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    async with aiofiles.open(export_path, 'w') as f:
        await f.write(export_data)
    
    return FileResponse(
        str(export_path),
        media_type='application/json',
        filename=f'encryption_keys_export_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

@app.post("/admin/import-keys")
async def import_keys(
    import_file: UploadFile = File(..., description="JSON file with keys to import"),
    admin_token: str = Depends(verify_admin_token)
):
    """Import keys from JSON file"""
    try:
        contents = await import_file.read()
        json_data = contents.decode('utf-8')
        
        success = admin_manager.import_keys(json_data)
        if success:
            return {
                "success": True,
                "message": "Keys imported successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to import keys - invalid format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")

@app.delete("/admin/clear-all-keys")
async def clear_all_keys(admin_token: str = Depends(verify_admin_token)):
    """Clear all encryption keys (dangerous operation)"""
    admin_manager.encryption_keys.clear()
    admin_manager.encryption_history.clear()
    return {
        "success": True,
        "message": "All keys cleared successfully",
        "warning": "This action cannot be undone"
    }

@app.get("/admin/system-info")
async def get_system_info(admin_token: str = Depends(verify_admin_token)):
    """Get detailed system information"""
    stats = admin_manager.get_system_stats()
    
    # Get memory usage info
    import psutil
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage('/')
    
    return {
        "system_stats": stats,
        "memory_usage": {
            "total": memory_info.total,
            "available": memory_info.available,
            "percent": memory_info.percent,
            "used": memory_info.used
        },
        "disk_usage": {
            "total": disk_info.total,
            "used": disk_info.used,
            "free": disk_info.free,
            "percent": (disk_info.used / disk_info.total) * 100
        },
        "temp_dir": str(admin_manager.temp_dir),
        "max_audio_length": crypto.max_audio_length / 44100
    }

@app.post("/admin/cleanup-temp")
async def cleanup_temp_files(admin_token: str = Depends(verify_admin_token)):
    """Clean up temporary files"""
    try:
        temp_files_deleted = 0
        temp_dir = admin_manager.temp_dir
        
        # Clean up old temporary files (older than 1 hour)
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                file_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        temp_files_deleted += 1
                    except Exception:
                        pass  # Skip files that can't be deleted
        
        return {
            "success": True,
            "message": f"Cleaned up {temp_files_deleted} temporary files",
            "files_deleted": temp_files_deleted
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/admin/recent-activity")
async def get_recent_activity(
    limit: int = 50,
    admin_token: str = Depends(verify_admin_token)
):
    """Get recent encryption/decryption activity"""
    recent_history = sorted(
        admin_manager.encryption_history,
        key=lambda x: x['timestamp'],
        reverse=True
    )[:limit]
    
    return {
        "success": True,
        "activity": recent_history,
        "total_records": len(admin_manager.encryption_history)
    }

# Additional utility endpoints
@app.get("/verify-package/{encryption_id}")
async def verify_package(encryption_id: str):
    """Verify if an encrypted package exists and get basic info"""
    package_path = admin_manager.temp_dir / f"encrypted_{encryption_id}.zip"
    
    if not package_path.exists():
        raise HTTPException(status_code=404, detail="Package not found")
    
    # Try to get key info from admin manager
    key_data = admin_manager.encryption_keys.get(encryption_id)
    
    return {
        "exists": True,
        "encryption_id": encryption_id,
        "package_size": package_path.stat().st_size,
        "created_time": datetime.datetime.fromtimestamp(package_path.stat().st_ctime).isoformat(),
        "has_admin_key": key_data is not None,
        "access_count": key_data['access_count'] if key_data else 0
    }

@app.get("/list-packages")
async def list_available_packages():
    """List all available encrypted packages"""
    packages = []
    temp_dir = admin_manager.temp_dir
    
    for package_file in temp_dir.glob("encrypted_*.zip"):
        encryption_id = package_file.stem.replace("encrypted_", "")
        key_data = admin_manager.encryption_keys.get(encryption_id)
        
        packages.append({
            "encryption_id": encryption_id,
            "filename": package_file.name,
            "size": package_file.stat().st_size,
            "created": datetime.datetime.fromtimestamp(package_file.stat().st_ctime).isoformat(),
            "has_admin_key": key_data is not None,
            "access_count": key_data['access_count'] if key_data else 0
        })
    
    return {
        "packages": sorted(packages, key=lambda x: x['created'], reverse=True),
        "total_packages": len(packages)
    }

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🚀 Fast Audio Encryption API Starting...")
    print(f"📁 Temp directory: {admin_manager.temp_dir}")
    print(f"🔐 Admin password: {admin_manager.admin_password}")
    print("✅ API is ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down Fast Audio Encryption API...")
    
    # Optional: Save encryption keys to persistent storage
    # This could be implemented to save keys to a database or file
    
    print("✅ Shutdown complete")

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast Audio Encryption API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--admin-password", default="admin123", help="Admin password")
    
    args = parser.parse_args()
    
    # Update admin password if provided
    if args.admin_password != "admin123":
        admin_manager.admin_password = args.admin_password
        print(f"🔐 Admin password updated")
    
    # Configure app debug mode
    app.debug = args.debug
    
    print(f"🌐 Starting server on {args.host}:{args.port}")
    print(f"📖 API docs available at: http://{args.host}:{args.port}/docs")
    print(f"🔧 Admin endpoints require authentication with password: {admin_manager.admin_password}")
    
    uvicorn.run(
        "main:app" if __name__ != "__main__" else app,
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info"
    )