import asyncio
import copy
from io import BytesIO
import json
import os
import queue
import threading

import modal

volume = modal.Volume.from_name("vllm", create_if_missing=True)
app = modal.App(name="pitchjams")

class SymmetricCipherHelper:
    def __init__(self, key: bytes):
        if len(key) not in [16, 24, 32]:
            raise ValueError("Invalid key size.")
        self.key = key

    def encrypt(self, plaintext: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        aesgcm = AESGCM(self.key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt(self, encrypted_payload: bytes) -> bytes:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if len(encrypted_payload) < 12:
            raise ValueError("malformed payload.")
        nonce = encrypted_payload[:12]
        ciphertext = encrypted_payload[12:]
        aesgcm = AESGCM(self.key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext
        except Exception as e:
            raise e
        
class P2PEncryption:
    """
    protocol representation of app level p2p encryption
    """
    def __init__(self, is_remote: bool):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        self._is_remote = is_remote
        if self._is_remote:
            self._secret = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
            )
            self._pem = self._secret.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            self._verifying_key = None
        self.cryptor: SymmetricCipherHelper = None
    
    def encryption_request(self):
        if not self._is_remote:
            print("Call this only on remote side.")
            return None
        if self.cryptor is not None:
            print("Secured protocol already established. No further action needed.")
            return None
        self._verifying_key = os.urandom(32)
        print(f"Public PEM is\n{self._pem.decode('utf-8')}")
        return self._pem, self._verifying_key
    
    def encryption_response(self, public_pem: bytes, verifying_key: bytes):
        from cryptography.hazmat.primitives import serialization, hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if self._is_remote:
            print("Call this only on local side.")
            return None
        if self.cryptor is not None:
            print("Secured protocol already established. No further action needed.")
            return None
        public_key = serialization.load_pem_public_key(public_pem)
        session_key = AESGCM.generate_key(bit_length=256)
        aesgcm = AESGCM(session_key)
        nonce = os.urandom(12)
        verifying_key_encrypted = aesgcm.encrypt(nonce, verifying_key, None)
        session_key_encrypted = public_key.encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        shared_secret = os.urandom(32)
        shared_secret_encrypted = aesgcm.encrypt(nonce, shared_secret, None)
        self.cryptor = SymmetricCipherHelper(shared_secret)
        return shared_secret_encrypted, verifying_key_encrypted, session_key_encrypted, nonce
    
    def encryption_acknowledged(self, shared_secret_encrypted: bytes, verifying_key_encrypted: bytes, session_key_encrypted: bytes, nonce: bytes):
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if not self._is_remote:
            print("Call this only on remote side, not local.")
            return False
        if self.cryptor is not None:
            print("Secured protocol already established. No further action needed.")
            return True
        try:
            session_key = self._secret.decrypt(
                session_key_encrypted,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            aesgcm = AESGCM(session_key)
            verifying_key = aesgcm.decrypt(nonce, verifying_key_encrypted, None)
            if verifying_key != self._verifying_key:
                print("Verifying key mismatch. Aborting.")
                return False
            shared_secret = aesgcm.decrypt(nonce, shared_secret_encrypted, None)
            self.cryptor = SymmetricCipherHelper(shared_secret)
            return True
        except Exception as e:
            print(f"Decryption or verification failed: {e}")
            return False

@app.function(
    image = modal.Image.debian_slim().run_commands("apt-get update && apt-get install -y git-lfs"),
    scaledown_window=2,
    volumes={"/vllm": volume},
)
def download_repo(remote_location: str):
    import threading
    import time
    name = remote_location.split("/")[-1]
    local_location = f"/vllm/{name}"
    if not os.path.exists(local_location):
        download_finish_event = threading.Event()
        def _download():
            os.system(f"git clone {remote_location} {local_location}")
            os.system(f"cd {local_location} && git lfs pull")
            download_finish_event.set()
        threading.Thread(target=_download).start()
        while not download_finish_event.is_set():
            time.sleep(10)
            yield f"heartbeat"
    yield f"completed"


@app.function(
    gpu="T4",
    image=modal.Image.debian_slim()
        .uv_pip_install([
            "diffusers[torch]==0.35.1",
            "transformers==4.56.1",
            "torch==2.8.0",
            "accelerate==1.10.1",
            "cryptography==45.0.7",
            "peft==0.17.1",
        ]),
    scaledown_window=2,
)
def transcribe(input_data: BytesIO) -> str:
    from transformers import pipeline
    import torch
    import numpy as np

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        torch_dtype=torch.float16,
        device="cuda:0",
    )
    waveform = np.frombuffer(input_data.getvalue(), dtype=np.float32)
    out = pipe(waveform, chunk_length_s=30, stride_length_s=5, return_timestamps=True)
    del pipe
    torch.cuda.empty_cache()
    return str(out["text"])

@app.cls(
    gpu="L40S", 
    image=modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install([
            "vllm==0.11.0",
            "transformers==4.56.1",
            "cryptography==45.0.7",
        ]
    ),
    enable_memory_snapshot=True,
    scaledown_window=2,
    volumes={"/vllm": volume},
) 
class VLLMModel:
    """
    use local module in future
    """
    model_name: str = modal.parameter()

    @modal.enter()
    def setup(self):
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        from transformers import AutoTokenizer
        self.cipher = P2PEncryption(is_remote=True)
        model_path = f"/vllm/{self.model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.vllm_scheduler = asyncio.new_event_loop()
        vllm_ready = threading.Event()
        def _vllm_thread():
            print("Starting vLLM event loop thread.")
            asyncio.set_event_loop(self.vllm_scheduler)
            async def _init_vllm():
                engine_args = AsyncEngineArgs(
                    model=model_path,
                )
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.vllm_scheduler.run_until_complete(_init_vllm())
            print("vLLM engine initialized.")
            vllm_ready.set()
            self.vllm_scheduler.run_forever()
        threading.Thread(target=_vllm_thread, daemon=True).start()
        vllm_ready.wait()

    @modal.method()
    def inference(self, serialized_payload: bytes):
        """
        this communication is encrypted, meaning given payload over the network must agree to the established cipher.
        essentially, it "predicts" the next token given the input token ids autoregressively 
        until vllm detects one of its stop tokens.
        """
        from vllm import SamplingParams
        # prepare input prompt
        decrypted_payload = self.cipher.cryptor.decrypt(serialized_payload)
        input_ids: list[int] = json.loads(decrypted_payload.decode('utf-8'))
        token_prompt = {
            # vllm.inputs.data.TokenPrompt
            "prompt_token_ids": input_ids
        }
        sampling_params = SamplingParams(
            temperature=0.7, 
            max_tokens=1024, 
        )

        # prepare data flow pipeline
        q = queue.Queue()
        async def _async_bridge():
            print("Starting async inference generation.")
            inference_stream = self.engine.generate(token_prompt, sampling_params, "0")
            async for request_output in inference_stream:
                # TODO: is it really a batch???
                # https://docs.vllm.ai/en/v0.11.0/api/vllm/outputs.html#vllm.outputs.RequestOutput
                for batch in request_output.outputs:
                    q.put(batch)
            q.put(None)

        # send inference request
        print("Scheduling async inference generation.")
        asyncio.run_coroutine_threadsafe(_async_bridge(), self.vllm_scheduler)
        output_ids: list[int] = []  # this is generated response after the while loop
        while output := q.get(): # output is type vllm.outputs.RequestOutput
            if output is None:
                break
            delta_ids = output.token_ids[len(output_ids):len(output.token_ids)]
            output_ids.extend(delta_ids)
            response_payload = json.dumps(delta_ids).encode('utf-8')
            encrypted_response = self.cipher.cryptor.encrypt(response_payload)
            yield encrypted_response


    @modal.method()
    def encryption_request(self):
        return self.cipher.encryption_request()
    
    @modal.method()
    def encryption_acknowledged(self, shared_secret_encrypted: bytes, verifying_key_encrypted: bytes, session_key_encrypted: bytes, nonce: bytes):
        return self.cipher.encryption_acknowledged(
            shared_secret_encrypted, 
            verifying_key_encrypted, 
            session_key_encrypted, 
            nonce
        )
