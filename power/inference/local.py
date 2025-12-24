"""
this module is currently unfunctional.
use vllm instead
"""
from abc import ABC, abstractmethod
import time
import threading

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from transformers import DynamicCache

class ConcurrentModelInference(ABC):
    """
    TODO: async safety

    Most the use of this class is thread-safe but is not guaranteed to be async-safe.
    """

    def __init__(self, model: str):
        self._model_name = model
        self._model = None
        self._processor = None

        self._encoder_lock = threading.Lock()
        self._decoder_lock = threading.Lock()
        self._encoder_worker = None
        self._decoder_worker = None

        self._decoder_params = {
            'temperature': torch.empty((0, 1), dtype=torch.float32),
            'top_k': torch.empty((0, 1), dtype=torch.int32),
            'top_p': torch.empty((0, 1), dtype=torch.float32)
        }
        self._batch_ref_set = []
        self._enc_inputs_lock = threading.Lock()
        self._dec_inputs_lock = threading.Lock()
        self._encoder_inputs = {'input_ids': None, 'attention_mask': None}
        self._decoder_inputs = {'input_ids': None, 'attention_mask': None}
        self._encoder_outputs = {}
        self._decoder_outputs = {}
        self._kv_cache = None
        self._autoregressive_sync = threading.Event()
    
    @property
    @abstractmethod
    def encoder(self):
        if self._model is None:
            raise ValueError("Model is not loaded. Call load() method first.")
        return self._model

    @property
    @abstractmethod
    def decoder(self):
        if self._model is None:
            raise ValueError("Model is not loaded. Call load() method first.")
        return self._model
    
    @property
    @abstractmethod
    def automodel(self) -> type:
        pass

    @property
    @abstractmethod
    def autoprocessor(self) -> type:
        pass

    def _forward(self, side: str, inputs: dict, **kwargs) -> dict:
        '''
        Forward passes the inputs
        Single call of this is effectively a single batch inference.
        '''
        model = self.encoder if side == 'encoder' else self.decoder if side == 'decoder' else None
        if model is None:
            raise ValueError("Invalid side:", side)
        with self._encoder_lock if side == 'encoder' else self._decoder_lock:
            outputs = model(**inputs, **kwargs)
        return outputs

    def __inference_loop(self, side: str):
        """
        This will run only once when the model is loaded.
        Both on encoder and decoder side where applicable.
        """
        def _encoder_loop() -> int:
            # Accept Batches
            time.sleep(20 / 1000)
            # Inference
            with self._enc_inputs_lock:
                if self._encoder_inputs['input_ids'].shape[0] == 0:
                    return 0
                outputs = self._forward(
                    'encoder',
                    **self._encoder_inputs
                )
            # Update Outputs
            self._encoder_outputs = outputs

        def _decoder_loop() -> int:
            # Accept Batches
            time.sleep(1 / 1000)
            self._dec_inputs_lock.acquire()
            if self._decoder_inputs['input_ids'] is None:
                self._dec_inputs_lock.release()
                return 0

            # Inference
            # batches detected

             # TODO: enable kv cache flag
            if self._kv_cache is None:
                self._kv_cache = []

            temperature = 1.0
            top_k = 50
            top_p = 0.95

            with torch.no_grad():
                # debug size of input ids, attention mask, and key and value cache
                outputs = self._forward(
                    'decoder',
                    self._decoder_inputs,
                    use_cache=True,
                    past_key_values=DynamicCache.from_legacy_cache(self._kv_cache) if len(self._kv_cache) > 0 else None
                )
            params = self._decoder_params
            hot_logits = outputs.logits[:, -1, :] #/ params['temperature']
            probs = torch.nn.functional.softmax(hot_logits, dim=-1).squeeze()

            # # top-k sample
            # top_k_values, _ = torch.topk(probs, top_k)
            # min_top_k = top_k_values[-1]
            # probs = torch.where(probs < min_top_k, torch.zeros_like(probs), probs)

            # # top-p sample
            # sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # cutoff = cumulative_probs > top_p
            # cutoff_indices = torch.where(cutoff)[0]
            # if cutoff_indices.numel() == 0:
            #     pass
            # else:
            #     cutoff_idx = cutoff_indices[0] + 1
            #     probs[sorted_indices[cutoff_idx:]] = 0
            #     probs = probs / probs.sum()

            #next_token_ids = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            next_token_ids = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            # Update batches
            keep_indices = []
            prev_batch_size = self._decoder_inputs['input_ids'].shape[0]
            dtype = self._decoder_inputs['input_ids'].dtype
            device = self._decoder_inputs['input_ids'].device
            batches = []
            # _decoder_inputs is the list of inputs to be given to the next forward pass
            # _decoder_outputs is the dict that holds the whole output from the decoder
            for i, token_id in enumerate(next_token_ids):
                if token_id != self._processor.eos_token_id:
                    keep_indices.append(i)
                    next_token_id = token_id.unsqueeze(0)
                    # only works when kv_cache is enabled
                    batches.append(next_token_id)
                # Update Outputs
                # TODO: output tensors should not reside on vRAM
                self._decoder_outputs[i] = torch.cat((self._decoder_outputs[i], token_id), dim=0) if (i in self._decoder_outputs) else torch.tensor([token_id], dtype=torch.int64, device=self._decoder_inputs['input_ids'].device)
            self._decoder_inputs['input_ids'] = torch.zeros((0, 1), dtype=dtype, device=device)
            for batch in batches:
                self._decoder_inputs['input_ids'] = torch.cat((self._decoder_inputs['input_ids'], batch), dim=0)
            self._decoder_inputs['attention_mask'] = None
            
            if len(keep_indices) != prev_batch_size:
                if len(keep_indices) == 0:
                    self._decoder_inputs['input_ids'] = None
                    self._decoder_inputs['attention_mask'] = None
                    self._kv_cache = None
                    # All batches are finished at this time so far
                else:
                    for i, ref in enumerate([self._batch_ref_set[i] for i in keep_indices]):
                        ref.value = i
                    keep_indices = torch.tensor(keep_indices, dtype=torch.int64, device=self._decoder_inputs['input_ids'].device)
                    self._decoder_inputs['input_ids'] = torch.index_select(self._decoder_inputs['input_ids'], 0, keep_indices)
                    #self._decoder_inputs['attention_mask'] = torch.index_select(self._decoder_inputs['attention_mask'], 0, keep_indices)

                    # edit cache accordingly based on current batch size
                    new_legacy_cache = []
                    batch_size = self._decoder_inputs['input_ids'].shape[0]
                    for key, value in outputs.past_key_values:
                        _, h, t, d = key.shape
                        reshaped_key = torch.index_select(key, 0, keep_indices).reshape(batch_size, h, t, d)
                        reshaped_value = torch.index_select(value, 0, keep_indices).reshape(batch_size, h, t, d)
                        new_legacy_cache.append((reshaped_key, reshaped_value))
                    self._kv_cache = new_legacy_cache
            else:
                self._kv_cache = outputs.past_key_values
            self._dec_inputs_lock.release()

            # Notify
            self._autoregressive_sync.set()
            self._autoregressive_sync.clear()

            return 0

        # Enc / Dec Inference Thread
        flag = 0
        while flag == 0:
            flag = _encoder_loop() if side == 'encoder' else _decoder_loop()

    def _schedule_batch(self, side: str, inputs: torch.Tensor, decoder_params: dict) -> 'ConcurrentModelInference._SynchronizedReference':
        """
        inputs: (seq_length)
        - This is input_ids, which is (99.99% surely) returned by processor() (i.e. tokenizer in case of text-text models)
        - Attention mask is manually set in this side of logic.

        further research required on additional input tensors returned by some models' processors
        e.g. BERT returns token_type_ids
        """
        if side not in ['encoder', 'decoder']:
            raise ValueError("Invalid side:", side)
        # shape of sided_inputs.input_ids always follows (batch_size, seq_length, +α...)
        sided_inputs = self._encoder_inputs if side == 'encoder' else self._decoder_inputs
        # sided_lock = inputs lock
        sided_lock = self._enc_inputs_lock if side == 'encoder' else self._dec_inputs_lock
        kv_cache: list = self._kv_cache if self._kv_cache is not None else None
        sided_lock.acquire()
        if sided_inputs['input_ids'] is None:
            sided_inputs['input_ids'] = inputs.unsqueeze(0)
            #sided_inputs['attention_mask'] = torch.ones_like(inputs).unsqueeze(0)
        else:
            seq_length = sided_inputs['input_ids'].shape[1]
            inputs_length = inputs.shape[0]
            if inputs_length > seq_length:
                sided_inputs['input_ids'] = F.pad(sided_inputs['input_ids'], (0, inputs.shape[0] - seq_length), value=self._processor.pad_token_id)
                #sided_inputs['attention_mask'] = F.pad(sided_inputs['attention_mask'], (0, inputs.shape[0] - seq_length), value=0)
                inputs_attention_mask = torch.ones_like(inputs)
            else:
                inputs_attention_mask = torch.ones_like(inputs)
                inputs = F.pad(inputs, (0, seq_length - inputs.shape[0]), value=self._processor.pad_token_id)
                inputs_attention_mask = F.pad(inputs_attention_mask, (0, seq_length - inputs_attention_mask.shape[0]), value=0)
            inputs = inputs.to(sided_inputs['input_ids'].device)
            sided_inputs['input_ids'] = torch.cat((sided_inputs['input_ids'], inputs.unsqueeze(0)), dim=0)
            #sided_inputs['attention_mask'] = torch.cat((sided_inputs['attention_mask'], inputs_attention_mask.unsqueeze(0)), dim=0)
        if side == 'decoder' and kv_cache is not None:
            new_legacy_cache = []
            for key, value in kv_cache:
                _, h, t, d = key.shape
                new_key = torch.zeros((1, h, t, d), dtype=key.dtype, device=key.device)
                new_value = torch.zeros_like(new_key)
                key = torch.cat((key, new_key), dim=0)
                value = torch.cat((value, new_value), dim=0)
                new_legacy_cache.append((key, value))
            self._kv_cache = new_legacy_cache
        batch_length = sided_inputs['input_ids'].shape[0]
        reference = ConcurrentModelInference._SynchronizedReference(batch_length - 1)
        if side == 'decoder' and decoder_params:
            params = self._decoder_params
            decoder_params = {k: torch.tensor([v], dtype=torch.float32, device=self._model.device).unsqueeze(0) for k, v in decoder_params.items() if k in ['temperature', 'top_k', 'top_p']}
            params['temperature'] = torch.cat(
                (params['temperature'], decoder_params['temperature']), dim=0
            )
            params['top_k'] = torch.cat(
                (params['top_k'], decoder_params['top_k']), dim=0
            )
            params['top_p'] = torch.cat(
                (params['top_p'], decoder_params['top_p']), dim=0
            )
        self._batch_ref_set.append(reference)

        # move all tensor to the same device as the model
        sided_inputs['input_ids'] = sided_inputs['input_ids'].to(self._model.device)
        #sided_inputs['attention_mask'] = sided_inputs['attention_mask'].to(self._model.device)
        if side == 'decoder':
            self._decoder_params['temperature'] = self._decoder_params['temperature'].to(self._model.device)
            self._decoder_params['top_k'] = self._decoder_params['top_k'].to(self._model.device)
            self._decoder_params['top_p'] = self._decoder_params['top_p'].to(self._model.device)
        sided_lock.release()
        return reference


    def _retrieve_batch(self, side: str, reference: 'ConcurrentModelInference._SynchronizedReference', terminate: bool=True):
        """
        Retrieves the batch from the inference queue.
        If terminate is True, it will remove the batch from the queue.
        Do not set terminate to False if inference side is encoder, unless in special circumstances.
        """
        if side not in ['encoder', 'decoder']:
            raise ValueError("Invalid side:", side)
        sided_output = self._encoder_outputs if side == 'encoder' else self._decoder_outputs
        sided_lock = self._enc_inputs_lock if side == 'encoder' else self._dec_inputs_lock
        self._autoregressive_sync.wait()
        with sided_lock:
            if reference.value in sided_output:
                batch = sided_output[reference.value]
                if terminate:
                    pass
                return batch
            else:
                return None

    @abstractmethod
    def pre_process(self, input_data: any, **kwargs):
        pass

    @abstractmethod
    def run_inference(self, input_data: any, streaming: bool=False):
        pass

    def load(self, device: str='cpu', **kwargs):
        """
        Thread unsafe
        """
        if not self._model:
            self._model = self.automodel.from_pretrained(self._model_name, device_map=device, **kwargs)
            self._processor = self.autoprocessor.from_pretrained(self._model_name)
            self._model.eval()
            for key, param in self._decoder_params.items():
                self._decoder_params[key] = param.to(self._model.device)
            if self.decoder:
                self._decoder_worker = threading.Thread(
                    target=self.__inference_loop,
                    args=('decoder',),
                    daemon=True
                )
                self._decoder_worker.start()
            else:
                del self._decoder_worker
            if self.encoder:
                self._encoder_worker = threading.Thread(
                    target=self.__inference_loop,
                    args=('encoder',),
                    daemon=True
                )
                self._encoder_worker.start()
            else:
                del self._encoder_worker

    class _SynchronizedReference:
        """
        Batches refered by this object are guranteed to be thread-safe.
        """
        def __init__(self, value: int):
            self._value = value

        @property
        def value(self) -> int:
            return self._value

        @value.setter
        def value(self, new_value: int):
            self._value = new_value

        @value.getter
        def value(self) -> int:
            return self._value

class TestDecoderOnlyModel(ConcurrentModelInference):

    @property
    def automodel(self):
        return AutoModelForCausalLM

    @property
    def autoprocessor(self):
        return AutoTokenizer

    @property
    def encoder(self):
        return None

    @property
    def decoder(self):
        if self._model is None:
            raise ValueError("Model is not loaded.")
        return self._model

    def load(self, device: str='cpu', **kwargs):
        super().load(device=device, **kwargs)
        self._processor.pad_token = self._processor.eos_token

    def pre_process(self, input_data, **kwargs):
        if isinstance(input_data, list):
            input_data = self._processor.apply_chat_template(input_data, add_generation_prompt=True, tokenize=False)
        input_data = self._processor(
            input_data, return_tensors="pt", padding=True, truncation=True
        )
        input_data = input_data["input_ids"]
        if input_data.dim() == 2 and input_data.size(0) == 1:
            input_data = input_data.squeeze(0)
        return input_data

    def run_inference(self, input_data, streaming=False):
        if isinstance(input_data, list):
            input_data = self._processor.apply_chat_template(input_data, add_generation_prompt=True, tokenize=False)
        inputs = self.pre_process(input_data)
        reference = self._schedule_batch(
            side='decoder',
            inputs=inputs,
            decoder_params={'temperature': 1.0, 'top_k': 50, 'top_p': 0.95}
        )
        while True:
            output = self._retrieve_batch(side='decoder', reference=reference)
            output = output[output.shape[0] - 1]
            try:
                if output != self._processor.eos_token_id:
                    yield self._processor.decode(
                        [output.item()],
                        skip_special_tokens=True,
                    )
                else:
                    break
            except:
                yield "<UNK>"