
# Speech-to-text model
# LLM for doing semantic text analysis
import os
from abc import ABC, abstractmethod
from io import BytesIO
import asyncio
from typing import TypedDict

import dotenv
import ffmpeg
import yt_dlp
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

from backend.power.inference.api import APIModel, OllamaAPIModel
from backend.logger import logger

class TextPitchEvalPipeline:
    """
    Much simpler version of PitchEval where model output are combined
    """
    MAX_QUEUE_SIZE = 10

    class PitchEvalState(TypedDict):
        transcription: str
        evaluations: str
        result: str

    def __init__(self):
        # Speech-to-text (transcription)
        # Uses Modal for now 
        # (FIXED history for langchain branch. You are still able to use local inference by substituting transformers pipeline here)
        import modal
        self.transcribe = modal.Function.from_name("pitchjams", "transcribe")

        # Base LLM
        llm = OllamaAPIModel(model='gpt-oss')

        # Task flow definition with LangGraph
        pitch_eval_prompts_path = 'backend/power/data/prompts/pitch_eval/'
        with open(os.path.join(pitch_eval_prompts_path, 'system.txt'), 'r') as f:
            system_prompt = f.read()

        def evaluate_transcription(state: TextPitchEvalPipeline.PitchEvalState) -> TextPitchEvalPipeline.PitchEvalState:
            evaluations = ''
            for metric in os.listdir(os.path.join(pitch_eval_prompts_path, 'metrics')):
                onetime_conversation = [{'role': 'system', 'content': system_prompt}]
                with open(os.path.join(pitch_eval_prompts_path, 'metrics', metric), 'r') as f:
                    metric_instruction = f.read()
                onetime_conversation.append({'role': 'user', 'content': f'{metric_instruction}\n\nTranscription:\n{state["transcription"]}'})
                evaluation_result = ''
                logger.info(f'Starting evaluation for metric: {metric}')
                for token_str in llm.call_inference(onetime_conversation):
                    evaluation_result += token_str
                logger.info(f'Completed evaluation for metric: {metric}')
                logger.info(f'{evaluation_result}')
                evaluations += f'\n\nMetric: {metric}\nEvaluation:\n{evaluation_result}'
            state['evaluations'] = evaluations
            return state

        def finalize_evaluation(state: TextPitchEvalPipeline.PitchEvalState) -> TextPitchEvalPipeline.PitchEvalState:
            langgraph_stream_writer = get_stream_writer()
            onetime_conversation = [{'role': 'user', 'content': f'Construct the summarization of this pitch evaluation based on the following evaluations:\n{state["evaluations"]}'}]
            result = ''
            for token_str in llm.call_inference(onetime_conversation):
                langgraph_stream_writer(token_str)
                result += token_str
            state['result'] = result
            return state

        graph = StateGraph(TextPitchEvalPipeline.PitchEvalState)
        graph.add_node('evaluate', evaluate_transcription)
        graph.add_node('finalize', finalize_evaluation)
        graph.add_edge(START, 'evaluate')
        graph.add_edge('evaluate', 'finalize')
        graph.add_edge('finalize', END)
        self.agent = graph.compile()

    def _status(self, status, message='', content=None):
            return {
                'status': status,
                'message': message,
                'content': content
            }
    
    def queue(self, input_video):
        """
        Specifically designed to be called for http streaming response.
        """
        yield self._status('transcribing', 'Video content is being transcribed')
        # Video to transcription
        if type(input_video) is str:
            print("Received input is a URL")
            # assuming string is a YouTube link
            audio = self._yt2audio(input_video)
        print("Transcribing audio...")
        transcription = self.transcribe.remote(audio)
        if not transcription:
            return self._status('error', 'Transcription failed or no text detected.')
        yield self._status('evaluating', 'Audio transcription completed.', transcription)

        # LLM inference
        initial_state = TextPitchEvalPipeline.PitchEvalState(
            transcription=transcription,
            evaluations='',
            result='',
        )
        result = ''
        for event in self.agent.stream(initial_state, stream_mode=['custom']):
            token_str = event[1]
            if token_str:
                result += token_str
                yield self._status('streaming', '', token_str)
        logger.info("Pitch evaluation completed.")
        logger.info(result)
        

    def _yt2audio(self, yt_pitch_link: str):
        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'forceurl': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(yt_pitch_link, download=False)
            if 'requested_formats' in info and info['requested_formats']:
                audio_url = next((f['url'] for f in info['requested_formats'] if f.get('vcodec') == 'none'), info['url'])
            else:
                audio_url = info['url']
        print(audio_url)
        try:
            out, error = ffmpeg.input(audio_url).output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar='16k').run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            print("FFmpeg failed:")
            print(e.stderr.decode('utf-8', errors='ignore'))
            return None
        return BytesIO(out)


    # def _speech_to_text(self, audio_data: BytesIO, dtype: str, sample_rate: int):
        
    #     print(f'Processing YouTube pitch link: {yt_pitch_link}')
    #     ydl_opts = {
    #         'format': 'bestaudio',
    #         'quiet': True,
    #         'no_warnings': True,
    #         'skip_download': True,
    #         'forceurl': True,
    #         'forcejson': True,
    #     }
    #     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #         info = ydl.extract_info(yt_pitch_link, download=False)
    #         audio_url = info['url']

    #     waveform = np.frombuffer(audio_data.getvalue(), dtype=dtype)
    #     inputs = self._whisper_processor(waveform, return_tensors="pt", sampling_rate=sample_rate)
    #     pred = self._whisper_model.generate(**inputs)
    #     text = self._whisper_processor.batch_decode(pred, skip_special_tokens=True)
    #     return text[0] if text and text[0] else None

pipeline = TextPitchEvalPipeline()
