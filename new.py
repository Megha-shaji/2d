import gradio as gr
import numpy as np
import hashlib
import random
from PyEMD import EMD
import json

# ========== ENCRYPTION ==========
def encrypt(audio_file):
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    header = audio_bytes[:44]
    audio = list(audio_bytes[44:])

    # Create hash value for ID
    hash_val = hashlib.sha3_512((repr(audio).encode())).hexdigest()

    # Divide hash value into 8 equal components
    components = [int(hash_val[i:i+16], 16) for i in range(0, 128, 16)]
    k = [(components[i] ^ components[i+1]) / (2**68) + (2.9 if i in (4, 6) else 0) for i in range(0, 7, 2)]
    x = components[0] / (2 ** 64)
    q = components[1] / (2 ** 64)

    # Generate shuffle index
    random.seed(hash_val)
    index = random.sample(range(0, len(audio)), len(audio))
    suff_aud = np.array(audio)[index]

    # Key generation
    def keygen(x, y, r1, r2, size):
        key1, key2 = [], []
        for i in range(size):
            x = k[0] * x * (q ** 2 - 1)
            y = y + x
            key1.append((int(x*pow(10,16))%256))
            key2.append((int(y*pow(10,16))%256))
        return key1, key2
    key1, key2 = keygen(k[0], k[1], k[2], k[3], len(audio))
    key = (np.array(key1)) ^ (np.array(key2))

    # 2D reshape
    factor = [i for i in range(1, len(suff_aud) + 1) if len(suff_aud) % i == 0]
    width, height = factor[len(factor) // 2 - 1], len(suff_aud) // factor[len(factor) // 2 - 1]
    td_mat = np.array(np.reshape(suff_aud, (height, width)), dtype='uint8')

    # EMD (slow, but only here)
    sum_IMF = []
    all_residual = []
    for i in range(height):
        signal = td_mat[i][:]
        emd = EMD(DTYPE=np.float32)
        energy = float('inf')
        IMFs = []
        engy = []
        while energy:
            IMF = emd.emd(signal, max_imf=1)
            IMF = np.int32(IMF)
            IMFs.append(IMF[0])
            signal = signal - IMF[0]
            energy = ((np.sum(signal ** 2)) / len(signal))
            engy.append(energy)
            if (len(engy) >= 2) and (engy[-1] == engy[-2]):
                break
        sum_IMF.append(np.sum(IMFs, axis=0))
        all_residual.append(signal)
    sum_IMF = np.array(sum_IMF)
    all_residual = np.array(all_residual)

    # Encrypt residual
    enc_residual = np.zeros(shape=[height, width], dtype='uint8')
    kidx = 0
    for i in range(height):
        for j in range(width):
            enc_residual[i][j] = ((all_residual[i][j])^key[kidx])
            kidx += 1

    # Save encrypted audio
    encrypt_signal = np.zeros(shape=[enc_residual.shape[0], enc_residual.shape[1]], dtype='uint8')
    for i in range(enc_residual.shape[0]):
        encrypt_signal[i,:] = sum_IMF[i] + enc_residual[i,:]
    encpt_aud = np.reshape(encrypt_signal, -1)
    encpt_aud_bytes = bytes(encpt_aud)
    encpt_main_audio = header + encpt_aud_bytes
    out_audio = "encrypted.wav"
    with open(out_audio, "wb") as f:
        f.write(encpt_main_audio)

    # Save parameters as a .txt file
    params = {
        "sum_IMF": sum_IMF.tolist(),
        "enc_residual_shape": list(enc_residual.shape),
        "index": index,
        "key": key.tolist(),
        "header": list(header)
    }
    out_param = "parameters.txt"
    with open(out_param, "w") as f:
        f.write(json.dumps(params))

    return out_audio, out_param

# ========== DECRYPTION ==========
def decrypt(param_file, encrypted_audio_file):
    with open(param_file, "r") as f:
        params = json.loads(f.read())
    sum_IMF = np.array(params["sum_IMF"])
    shape = tuple(params["enc_residual_shape"])
    index = params["index"]
    key = np.array(params["key"])
    header = bytes(params["header"])

    with open(encrypted_audio_file, "rb") as f:
        enc_bytes = f.read()
    enc_audio = list(enc_bytes[44:])
    enc_residual = np.array(enc_audio).reshape(shape)

    height, width = enc_residual.shape

    # Decrypt residual
    decryp_residual = np.zeros(shape=[height, width], dtype='uint8')
    kidx = 0
    for i in range(height):
        for j in range(width):
            decryp_residual[i][j] = (enc_residual[i][j]) ^ key[kidx]
            kidx += 1

    # Decrypt signal
    decrypt_signal = np.zeros(shape=[height, width], dtype='uint8')
    for i in range(height):
        decrypt_signal[i,:] = sum_IMF[i] + decryp_residual[i,:]
    decry_aud_1d = np.reshape(decrypt_signal, -1)
    shuffle_aud = np.zeros(height * width)
    shuffle_aud[index] = decry_aud_1d
    shuffle_aud = shuffle_aud.astype('uint8')
    decrypted_audio = bytes(shuffle_aud)
    out_path = "decrypted.wav"
    with open(out_path, "wb") as f:
        f.write(header + decrypted_audio)
    return out_path

# ========== GRADIO INTERFACE ==========
with gr.Blocks() as demo:
    with gr.Tab("Encryption"):
        audio_in = gr.Audio(label="Upload Audio (WAV)", type="filepath")
        enc_audio_out = gr.File(label="Encrypted Audio")
        param_file_out = gr.File(label="Decryption Parameters (.txt)")
        encrypt_btn = gr.Button("Encrypt")
        encrypt_btn.click(encrypt, inputs=audio_in, outputs=[enc_audio_out, param_file_out])

    with gr.Tab("Decryption"):
        param_file_in = gr.File(label="Upload Parameter File (.txt)")
        enc_audio_in = gr.Audio(label="Upload Encrypted Audio (WAV)", type="filepath")
        dec_audio_out = gr.File(label="Decrypted Audio")
        decrypt_btn = gr.Button("Decrypt")
        decrypt_btn.click(decrypt, inputs=[param_file_in, enc_audio_in], outputs=dec_audio_out)

demo.launch()


###########
