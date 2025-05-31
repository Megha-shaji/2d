import gradio as gr
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
from typing import Dict, List, Tuple, Optional
import secrets
warnings.filterwarnings('ignore')

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
    
    def encrypt_audio_fast(self, audio_data, user_info=None):
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
    
    def decrypt_audio_fast(self, encrypted_data, indices, hash_hex, block_size):
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

# Initialize admin manager and crypto system
admin_manager = AdminKeyManager()
crypto = FastAudioCrypto(admin_manager)

def encrypt_audio(input_audio):
    """Fast audio encryption with admin key storage"""
    try:
        if input_audio is None:
            return None, None, "❌ Please upload an audio file"
        
        # Load audio
        start_time = __import__('time').time()
        try:
            audio_data, sample_rate = librosa.load(input_audio, sr=None, duration=30)
        except Exception as e:
            return None, None, f"❌ Failed to load audio file: {str(e)}"
        
        if len(audio_data) == 0:
            return None, None, "❌ Audio file is empty"
        
        load_time = __import__('time').time() - start_time
        
        # Encrypt audio
        encrypt_start = __import__('time').time()
        user_info = {'ip': 'local', 'timestamp': datetime.datetime.now().isoformat()}
        encrypted_data, indices, hash_hex, block_size, encryption_id = crypto.encrypt_audio_fast(audio_data, user_info)
        encrypt_time = __import__('time').time() - encrypt_start
        
        # Create encrypted audio preview (noise)
        preview_audio = (encrypted_data.astype(np.float32) / 255.0 * 2.0 - 1.0) * 0.1
        
        # Create metadata
        metadata = {
            'encrypted_data': encrypted_data.tolist(),
            'indices': indices.tolist(),
            'hash_hex': hash_hex,
            'block_size': block_size,
            'sample_rate': sample_rate,
            'original_length': len(audio_data),
            'encryption_id': encryption_id,
            'version': '2.0_fast_admin'
        }
        
        # Save preview audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
            sf.write(tmp_audio.name, preview_audio, sample_rate)
            preview_path = tmp_audio.name
        
        # Create encryption package
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl', mode='wb') as tmp_meta:
            pickle.dump(metadata, tmp_meta)
            metadata_path = tmp_meta.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
            with zipfile.ZipFile(tmp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(metadata_path, 'fast_encryption_data.pkl')
                readme_content = f"""
Fast Audio Encryption Package (Admin Version)
============================================
Encryption ID: {encryption_id}
Original Duration: {len(audio_data)/sample_rate:.2f} seconds
Sample Rate: {sample_rate} Hz
Algorithm: Fast Chaotic Encryption (v2.0 Admin)
Load Time: {load_time:.3f}s
Encryption Time: {encrypt_time:.3f}s

This package uses fast chaotic encryption with admin key management.
Keys are automatically stored in the admin panel for recovery.
To decrypt: Upload this zip file in the 'Decrypt Audio' tab.
"""
                zf.writestr('README.txt', readme_content)
            zip_path = tmp_zip.name
        
        # Cleanup
        os.unlink(metadata_path)
        
        total_time = load_time + encrypt_time
        success_msg = f"""✅ Audio encrypted successfully!
🆔 Encryption ID: {encryption_id}
⏱️ Duration: {len(audio_data)/sample_rate:.2f} seconds
📊 Sample Rate: {sample_rate} Hz
⚡ Load Time: {load_time:.3f} seconds
🔐 Encryption Time: {encrypt_time:.3f} seconds
📊 Total Time: {total_time:.3f} seconds
🚀 Algorithm: Fast Chaotic Encryption (Admin)
🔑 Key stored in admin panel for recovery

📥 Download the zip package below."""
        
        return preview_path, zip_path, success_msg
        
    except Exception as e:
        return None, None, f"❌ Encryption error: {str(e)}"

def decrypt_audio(encrypted_zip):
    """Fast audio decryption with admin key fallback"""
    try:
        if encrypted_zip is None:
            return None, "❌ Please upload an encrypted zip file"
        
        start_time = __import__('time').time()
        
        # Extract zip
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with zipfile.ZipFile(encrypted_zip, 'r') as zf:
                    zf.extractall(temp_dir)
            except Exception as e:
                return None, f"❌ Invalid zip file: {str(e)}"
            
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
                return None, "❌ No valid encryption data found in zip file"
            
            # Check if it's admin version
            if metadata.get('version', '').endswith('_admin'):
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
                
                decrypt_start = __import__('time').time()
                decrypted_audio = crypto.decrypt_audio_fast(
                    encrypted_data, indices, hash_hex, block_size
                )
                decrypt_time = __import__('time').time() - decrypt_start
                
            elif metadata.get('version', '').startswith('2.0'):
                # Fast decryption
                encrypted_data = np.array(metadata['encrypted_data'], dtype=np.uint8)
                indices = np.array(metadata['indices'])
                
                decrypt_start = __import__('time').time()
                decrypted_audio = crypto.decrypt_audio_fast(
                    encrypted_data,
                    indices,
                    metadata['hash_hex'],
                    metadata['block_size']
                )
                decrypt_time = __import__('time').time() - decrypt_start
            else:
                return None, "❌ This file was encrypted with an unsupported version."
            
            # Save decrypted audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                sf.write(tmp_file.name, decrypted_audio, metadata['sample_rate'])
                decrypted_path = tmp_file.name
            
            total_time = __import__('time').time() - start_time
            
            success_msg = f"""✅ Audio decrypted successfully!
🆔 Encryption ID: {metadata.get('encryption_id', 'Unknown')}
⏱️ Duration: {len(decrypted_audio)/metadata['sample_rate']:.2f} seconds
📊 Sample Rate: {metadata['sample_rate']} Hz
🔓 Decryption Time: {decrypt_time:.3f} seconds
📊 Total Time: {total_time:.3f} seconds
🚀 Algorithm: Fast Chaotic Encryption
{'🔑 Key retrieved from admin panel' if metadata.get('version', '').endswith('_admin') else ''}"""
            
            return decrypted_path, success_msg
            
    except Exception as e:
        return None, f"❌ Decryption error: {str(e)}"

# Admin functions
def admin_login(password):
    """Admin authentication"""
    if admin_manager.authenticate_admin(password):
        stats = admin_manager.get_system_stats()
        keys = admin_manager.list_all_keys()
        
        # Format keys table
        keys_table = []
        for key in keys[:20]:  # Show last 20 keys
            keys_table.append([
                key['encryption_id'],
                key['created_at'][:19],  # Remove microseconds
                str(key['access_count']),
                key['last_accessed'][:19] if key['last_accessed'] else 'Never',
                key['user_info'].get('ip', 'Unknown')
            ])
        
        stats_text = f"""🔑 Admin Panel - System Statistics
=====================================
🚀 Total Encryptions: {stats['total_encryptions']}
🔓 Total Decryptions: {stats['total_decryptions']}
📊 Keys Stored: {stats['total_keys_stored']}
⏰ System Uptime: {stats['uptime']}
🕐 Recent Activity (24h): {stats['recent_activity']}
🖥️ System Started: {stats['system_start_time'].strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return True, stats_text, keys_table, ""
    else:
        return False, "❌ Invalid admin password", [], "❌ Authentication failed"

def admin_search_key(encryption_id, password):
    """Search for specific encryption key"""
    if not admin_manager.authenticate_admin(password):
        return "❌ Invalid admin password"
    
    if not encryption_id:
        return "❌ Please enter an encryption ID"
    
    key_data = admin_manager.get_encryption_key(encryption_id)
    if key_data:
        return f"""✅ Key Found: {encryption_id}
📊 Original Length: {key_data['original_length']} samples
🔑 Hash: {key_data['hash_hex'][:32]}...
📦 Block Size: {key_data['block_size']}
🕐 Last Accessed: Just now"""
    else:
        return f"❌ Key not found: {encryption_id}"

def admin_delete_key(encryption_id, password):
    """Delete encryption key"""
    if not admin_manager.authenticate_admin(password):
        return "❌ Invalid admin password"
    
    if admin_manager.delete_key(encryption_id):
        return f"✅ Key deleted successfully: {encryption_id}"
    else:
        return f"❌ Key not found: {encryption_id}"

def admin_export_keys(password):
    """Export all keys"""
    if not admin_manager.authenticate_admin(password):
        return "❌ Invalid admin password", None
    
    export_data = admin_manager.export_keys()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_file:
        tmp_file.write(export_data)
        export_path = tmp_file.name
    
    return "✅ Keys exported successfully", export_path

# Enhanced Gradio Interface
def create_interface():
    with gr.Blocks(
        title="⚡ Fast Audio Encryption System - Admin Edition",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .status-box { min-height: 120px; font-family: monospace; }
        .info-section { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
        .speed-highlight { background: linear-gradient(45deg, #00ff88, #00ccff); padding: 0.2rem 0.5rem; border-radius: 4px; font-weight: bold; }
        .admin-section { background: linear-gradient(45deg, #ff6b6b, #ffa500); padding: 0.2rem 0.5rem; border-radius: 4px; color: white; font-weight: bold; }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>⚡ Fast Audio Encryption System</h1>
            <p class="admin-section">🔐 ADMIN EDITION - Complete Key Management</p>
            <p class="speed-highlight">🚀 Ultra-Fast Chaotic Encryption with Centralized Admin Panel!</p>
        </div>
        """)
        
        with gr.Tabs():
            # Encryption Tab
            with gr.Tab("🔒 Encrypt Audio", elem_id="encrypt-tab"):
                gr.Markdown("### 📤 Upload an audio file to encrypt (max 30 seconds)")
                gr.Markdown("⚡ **Admin Version:** All keys automatically stored in admin panel for recovery!")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        encrypt_input = gr.Audio(
                            label="🎵 Input Audio File",
                            type="filepath",
                            sources=["upload"]
                        )
                        encrypt_btn = gr.Button(
                            "⚡ Fast Encrypt", 
                            variant="primary", 
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        encrypted_preview = gr.Audio(
                            label="🔊 Encrypted Audio Preview", 
                            interactive=False
                        )
                        encrypted_download = gr.File(
                            label="📥 Download Encryption Package",
                            file_count="single"
                        )
                        encrypt_status = gr.Textbox(
                            label="📋 Encryption Status & Timing",
                            interactive=False,
                            lines=10,
                            elem_classes=["status-box"]
                        )
                
                encrypt_btn.click(
                    fn=encrypt_audio,
                    inputs=[encrypt_input],
                    outputs=[encrypted_preview, encrypted_download, encrypt_status]
                )
            
            # Decryption Tab
            with gr.Tab("🔓 Decrypt Audio", elem_id="decrypt-tab"):
                gr.Markdown("### 📥 Upload encrypted package to decrypt")
                gr.Markdown("⚡ **Admin Recovery:** Can decrypt using admin-stored keys if package is corrupted!")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        decrypt_input = gr.File(
                            label="📦 Encrypted Package (.zip)",
                            file_types=[".zip"]
                        )
                        decrypt_btn = gr.Button(
                            "⚡ Fast Decrypt", 
                            variant="secondary", 
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        decrypted_audio = gr.Audio(
                            label="🎵 Decrypted Audio", 
                            interactive=False
                        )
                        decrypt_status = gr.Textbox(
                            label="📋 Decryption Status & Timing",
                            interactive=False,
                            lines=8,
                            elem_classes=["status-box"]
                        )
                
                decrypt_btn.click(
                    fn=decrypt_audio,
                    inputs=[decrypt_input],
                    outputs=[decrypted_audio, decrypt_status]
                )
            
            # Admin Panel Tab
            with gr.Tab("🔐 Admin Panel", elem_id="admin-tab"):
                gr.HTML('<div class="admin-section" style="text-align: center; margin-bottom: 1rem;">🔐 ADMINISTRATOR ACCESS REQUIRED</div>')
                
                with gr.Row():
                    admin_password = gr.Textbox(
                        label="🔑 Admin Password",
                        type="password",
                        placeholder="Enter admin password..."
                    )
                    admin_login_btn = gr.Button("🔓 Login", variant="primary")
                
                # Admin sections (initially hidden)
                with gr.Column(visible=False) as admin_content:
                    # System Statistics
                    gr.Markdown("### 📊 System Statistics")
                    admin_stats = gr.Textbox(
                        label="📈 System Stats",
                        interactive=False,
                        lines=8,
                        elem_classes=["status-box"]
                    )
                    
                    # Keys Management
                    gr.Markdown("### 🔑 Encryption Keys Management")
                    
                    with gr.Row():
                        with gr.Column():
                            keys_table = gr.Dataframe(
                                headers=["Encryption ID", "Created", "Access Count", "Last Accessed", "User IP"],
                                label="🗃️ Recent Encryption Keys (Last 20)",
                                interactive=False,
                                wrap=True
                            )
                        
                        with gr.Column():
                            # Key Search
                            gr.Markdown("#### 🔍 Search Key")
                            search_key_id = gr.Textbox(
                                label="Encryption ID",
                                placeholder="Enter encryption ID..."
                            )
                            search_btn = gr.Button("🔍 Search Key", size="sm")
                            search_result = gr.Textbox(
                                label="Search Result",
                                interactive=False,
                                lines=6
                            )
                            
                            # Key Deletion
                            gr.Markdown("#### 🗑️ Delete Key")
                            delete_key_id = gr.Textbox(
                                label="Encryption ID to Delete",
                                placeholder="Enter encryption ID..."
                            )
                            delete_btn = gr.Button("🗑️ Delete Key", variant="stop", size="sm")
                            delete_result = gr.Textbox(
                                label="Delete Result",
                                interactive=False,
                                lines=2
                            )
                    
                    # Export/Import Section
                    gr.Markdown("### 📤 Export/Import Keys")
                    with gr.Row():
                        with gr.Column():
                            export_btn = gr.Button("📤 Export All Keys", variant="secondary")
                            export_file = gr.File(
                                label="📥 Download Keys Export",
                                file_count="single"
                            )
                            export_status = gr.Textbox(
                                label="Export Status",
                                interactive=False,
                                lines=2
                            )
                        
                        with gr.Column():
                            import_file = gr.File(
                                label="📤 Import Keys File (.json)",
                                file_types=[".json"]
                            )
                            import_btn = gr.Button("📥 Import Keys", variant="secondary")
                            import_status = gr.Textbox(
                                label="Import Status",
                                interactive=False,
                                lines=2
                            )
                    
                    # Advanced Admin Tools
                    gr.Markdown("### ⚙️ Advanced Admin Tools")
                    with gr.Row():
                        refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
                        clear_history_btn = gr.Button("🧹 Clear History", variant="stop", size="sm")
                        reset_system_btn = gr.Button("⚠️ Reset System", variant="stop", size="sm")
                    
                    admin_tools_status = gr.Textbox(
                        label="Admin Tools Status",
                        interactive=False,
                        lines=3
                    )
                
                # Login state
                login_state = gr.State(False)
                
                # Admin login logic
                def handle_admin_login(password):
                    success, stats, keys, status = admin_login(password)
                    return (
                        gr.update(visible=success),  # admin_content visibility
                        stats,  # admin_stats
                        keys,   # keys_table
                        status, # login status
                        success # login_state
                    )
                
                # Admin tool functions
                def admin_refresh_stats(password):
                    if not admin_manager.authenticate_admin(password):
                        return "❌ Invalid admin password"
                    
                    stats = admin_manager.get_system_stats()
                    return f"""🔄 Stats Refreshed!
🚀 Total Encryptions: {stats['total_encryptions']}
🔓 Total Decryptions: {stats['total_decryptions']}
📊 Keys Stored: {stats['total_keys_stored']}
⏰ System Uptime: {stats['uptime']}"""
                
                def admin_clear_history(password):
                    if not admin_manager.authenticate_admin(password):
                        return "❌ Invalid admin password"
                    
                    admin_manager.encryption_history = []
                    return "✅ History cleared successfully"
                
                def admin_reset_system(password):
                    if not admin_manager.authenticate_admin(password):
                        return "❌ Invalid admin password"
                    
                    admin_manager.encryption_keys = {}
                    admin_manager.encryption_history = []
                    admin_manager.system_stats = {
                        'total_encryptions': 0,
                        'total_decryptions': 0,
                        'total_users': 0,
                        'system_start_time': datetime.datetime.now()
                    }
                    return "⚠️ System reset successfully - All data cleared!"
                
                def admin_import_keys_func(import_file, password):
                    if not admin_manager.authenticate_admin(password):
                        return "❌ Invalid admin password"
                    
                    if import_file is None:
                        return "❌ Please select a JSON file to import"
                    
                    try:
                        with open(import_file, 'r') as f:
                            json_data = f.read()
                        
                        if admin_manager.import_keys(json_data):
                            return "✅ Keys imported successfully"
                        else:
                            return "❌ Failed to import keys - Invalid file format"
                    except Exception as e:
                        return f"❌ Import error: {str(e)}"
                
                # Event handlers
                admin_login_btn.click(
                    fn=handle_admin_login,
                    inputs=[admin_password],
                    outputs=[admin_content, admin_stats, keys_table, admin_tools_status, login_state]
                )
                
                search_btn.click(
                    fn=admin_search_key,
                    inputs=[search_key_id, admin_password],
                    outputs=[search_result]
                )
                
                delete_btn.click(
                    fn=admin_delete_key,
                    inputs=[delete_key_id, admin_password],
                    outputs=[delete_result]
                )
                
                export_btn.click(
                    fn=admin_export_keys,
                    inputs=[admin_password],
                    outputs=[export_status, export_file]
                )
                
                import_btn.click(
                    fn=admin_import_keys_func,
                    inputs=[import_file, admin_password],
                    outputs=[import_status]
                )
                
                refresh_stats_btn.click(
                    fn=admin_refresh_stats,
                    inputs=[admin_password],
                    outputs=[admin_tools_status]
                )
                
                clear_history_btn.click(
                    fn=admin_clear_history,
                    inputs=[admin_password],
                    outputs=[admin_tools_status]
                )
                
                reset_system_btn.click(
                    fn=admin_reset_system,
                    inputs=[admin_password],
                    outputs=[admin_tools_status]
                )
            
            # Information Tab
            with gr.Tab("ℹ️ Algorithm Info & Admin Features"):
                gr.HTML("""
                <div class="info-section">
                    <h3>🔐 New Admin Features in v2.0</h3>
                    <ul>
                        <li><strong>Centralized Key Storage:</strong> All encryption keys automatically stored in admin panel</li>
                        <li><strong>Key Recovery:</strong> Admin can decrypt files even if package is corrupted</li>
                        <li><strong>User Activity Tracking:</strong> Monitor all encryption/decryption activities</li>
                        <li><strong>System Statistics:</strong> Real-time stats on usage and performance</li>
                        <li><strong>Key Management:</strong> Search, view, and delete encryption keys</li>
                        <li><strong>Export/Import:</strong> Backup and restore all encryption data</li>
                        <li><strong>Admin Authentication:</strong> Secure password-protected access</li>
                    </ul>
                </div>
                
                <div class="info-section">
                    <h3>🛡️ Enhanced Security with Admin Panel</h3>
                    <ul>
                        <li><strong>Dual Key Storage:</strong> Keys stored both in package and admin panel</li>
                        <li><strong>Access Logging:</strong> All key access attempts are logged</li>
                        <li><strong>Admin Oversight:</strong> Complete visibility into all encryption activities</li>
                        <li><strong>Emergency Recovery:</strong> Admin can recover lost keys</li>
                        <li><strong>System Monitoring:</strong> Track system health and usage patterns</li>
                        <li><strong>Data Backup:</strong> Export/import functionality for disaster recovery</li>
                    </ul>
                </div>
                
                <div class="info-section">
                    <h3>⚡ Fast Encryption Algorithm (v2.0 Admin)</h3>
                    <p><strong>Same ultra-fast performance with added admin capabilities!</strong></p>
                    <ol>
                        <li><strong>Hash Generation:</strong> SHA-256 from audio data (fast)</li>
                        <li><strong>Sample Permutation:</strong> Pseudo-random shuffle using hash seed</li>
                        <li><strong>Chaotic Key Generation:</strong> Fast logistic map iteration</li>
                        <li><strong>XOR Scrambling:</strong> Simple but effective bit manipulation</li>
                        <li><strong>Block Diffusion:</strong> Additional confusion layer with safe rotations</li>
                        <li><strong>Admin Key Storage:</strong> Automatic key backup to admin panel</li>
                        <li><strong>Package Creation:</strong> Lightweight metadata packaging</li>
                    </ol>
                </div>
                
                <div class="info-section">
                    <h3>🔑 Admin Panel Usage</h3>
                    <ol>
                        <li><strong>Login:</strong> Use admin password to access admin panel</li>
                        <li><strong>View Keys:</strong> See all stored encryption keys and their metadata</li>
                        <li><strong>Search Keys:</strong> Find specific keys by encryption ID</li>
                        <li><strong>Delete Keys:</strong> Remove keys from storage (use with caution)</li>
                        <li><strong>Export Data:</strong> Backup all keys and system data</li>
                        <li><strong>Import Data:</strong> Restore from backup files</li>
                        <li><strong>Monitor Stats:</strong> Track system usage and performance</li>
                    </ol>
                </div>
                
                <div class="info-section">
                    <h3>⚙️ Technical Specs (Admin Edition)</h3>
                    <ul>
                        <li><strong>Algorithm:</strong> Chaotic Permutation-Scrambling-Diffusion</li>
                        <li><strong>Key Storage:</strong> In-memory with export/import capability</li>
                        <li><strong>Admin Auth:</strong> Password-based authentication</li>
                        <li><strong>Activity Logging:</strong> Complete audit trail</li>
                        <li><strong>Data Export:</strong> JSON format for easy backup</li>
                        <li><strong>Recovery Mode:</strong> Admin key fallback for corrupted packages</li>
                        <li><strong>Max Duration:</strong> 30 seconds per file</li>
                        <li><strong>Default Admin Password:</strong> admin123 (change in production!)</li>
                    </ul>
                </div>
                
                <div class="info-section" style="background: #fff3cd; border: 1px solid #ffeaa7;">
                    <h3>⚠️ Security Notice</h3>
                    <p><strong>IMPORTANT:</strong> Change the default admin password in production environments!</p>
                    <p>The current default password is "admin123" - this should be changed in the AdminKeyManager class for security.</p>
                    <p>All keys are stored in memory and will be lost when the application restarts unless exported.</p>
                </div>
                """)
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: linear-gradient(45deg, #f0f8ff, #e6f3ff); border-radius: 8px;">
            <p><strong>⚡ Fast Audio Encryption System v2.0 - Admin Edition</strong></p>
            <p class="admin-section">🔐 Complete Key Management & Recovery System</p>
            <p>🚀 Ultra-fast chaotic encryption with centralized admin control!</p>
        </div>
        """)
    
    return demo

# Launch interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7860,
        show_error=False
    )