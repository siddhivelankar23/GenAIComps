import base64
import json
import os
from pathlib import Path
import requests
from typing import List, Optional, Tuple, Union, Iterator, Any
import uuid

import cv2
import torch
import torch.nn.functional as F
import webvtt
import whisper
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
from moviepy.editor import VideoFileClip
from torch import nn
from torchvision.io import ImageReadMode, read_image
import torchvision.transforms.functional as transform
from transformers import BridgeTowerProcessor, BridgeTowerPreTrainedModel, BridgeTowerModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bridgetower.modeling_bridgetower import BridgeTowerTextModel


class BridgeTowerITCHead(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class _BridgeTowerTextModelWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = BridgeTowerTextModel(config)

    def forward(self, **kwargs):
        return self.text_model(**kwargs)


class BridgeTowerTextFeatureExtractor(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bridgetower = _BridgeTowerTextModelWrapper(config.text_config)
        self.itc_text_head = BridgeTowerITCHead(config.hidden_size, config.contrastive_hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        
        outputs = self.bridgetower(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        final_hidden_cls = outputs.hidden_states[-1][:,0,:]
        final_hidden_cls = F.normalize(self.itc_text_head(final_hidden_cls), dim=-1, p=2)

        return final_hidden_cls
    

class BridgeTowerForITC(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bridgetower = BridgeTowerModel(config)

        self.itc_text_head = BridgeTowerITCHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_image_head = BridgeTowerITCHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_cross_modal_head = BridgeTowerITCHead(config.hidden_size * 2, config.contrastive_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:

        assert output_hidden_states, 'output_hidden_states should be set to True for BridgeTowerForITC'
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bridgetower(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[2]

        hidden_states_txt, hidden_states_img, hidden_states_cross_modal = outputs.hidden_states

        final_hidden_txt = hidden_states_txt[-1]
        final_hidden_img = hidden_states_img[-1]

        image_embeds_with_ln = self.bridgetower.vision_model.visual.forward_post(final_hidden_img)
        image_token_type_embeddings = self.bridgetower.token_type_embeddings(
            torch.full((1,), 1, dtype=torch.long, device=self.bridgetower.token_type_embeddings.weight.device)
        ).expand_as(image_embeds_with_ln)

        final_hidden_img = (
            self.bridgetower.cross_modal_image_transform(image_embeds_with_ln)
            + image_token_type_embeddings
        )

        final_hidden_txt = F.normalize(self.itc_text_head(final_hidden_txt[:,0,:]), dim=-1, p=2)
        final_hidden_img = F.normalize(self.itc_image_head(final_hidden_img[:,0,:]), dim=-1, p=2)
        final_hidden_cross = F.normalize(self.itc_cross_modal_head(pooler_output), dim=-1, p=2)

        logits = torch.stack([final_hidden_txt, final_hidden_img, final_hidden_cross], dim=-2)

        if not return_dict:
            return tuple(logits)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """ BridgeTower embedding model """
    model_name: str = "BridgeTower/bridgetower-large-itm-mlm-itc"
    device: str = "cpu"
    text_model : Any
    processor: Any
    model: Any

    def __init__(self, **kwargs: Any):
        """Initialize the BridgeTowerEmbeddings class"""
        super().__init__(**kwargs)
        self.text_model = BridgeTowerTextFeatureExtractor.from_pretrained(self.model_name).to(self.device)
        self.processor = BridgeTowerProcessor.from_pretrained(self.model_name)
        self.model = BridgeTowerForITC.from_pretrained(self.model_name).to(self.device)

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BridgeTower.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        encodings = self.processor.tokenizer(texts, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**encodings)
        embeddings = outputs.cpu().numpy().tolist()
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using BridgeTower.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:
        """Embed a list of image-text pairs using BridgeTower.

        Args:
            texts: The list of texts to embed.
            images: The list of path-to-images to embed
            batch_size: the batch size to process, default to 2
        Returns:
            List of embeddings, one for each image-text pairs.
        """

        # the length of texts must be equal to the length of images
        assert len(texts)==len(images), "the len of captions should be equal to the len of images"
        
        image_list = []
        text_list = []
        embeddings = []
        for path_to_img, text in zip(images, texts):
            img = read_image(path_to_img, mode=ImageReadMode.RGB)
            img = transform.to_pil_image(img)
            image_list.append(img)
            text_list.append(text)
            if len(text_list) == batch_size:
                batch = self.processor(image_list, text_list, return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(self.device)
                with torch.no_grad():
                    batch_embeddings = self.model(**batch, output_hidden_states=True)
                for i in range(len(text_list)):
                    embeddings.append(batch_embeddings.logits[i,2,:].detach().cpu().numpy().tolist())
                image_list = []
                text_list = []
        # embedding the remaining        
        if len(text_list) > 0:
            batch = self.processor(image_list, text_list, return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model(**batch, output_hidden_states=True)
            for i in range(len(text_list)):
                embeddings.append(batch_embeddings.logits[i,2,:].detach().cpu().numpy().tolist())
            image_list = []
            text_list = []
        return embeddings
    

def create_upload_folder(upload_path):
    """Create a directory to store uploaded video data"""
    if not os.path.exists(upload_path):
        Path(upload_path).mkdir(parents=True, exist_ok=True)


def load_json_file(file_path):
    """Read contents of json file"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def clear_upload_folder(upload_path):
    """Clear the upload directory"""
    for root, dirs, files in os.walk(upload_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)


def generate_video_id():
    """Generates a unique identifier for a video file"""
    return str(uuid.uuid4())


def convert_video_to_audio(video_path: str, output_audio_path: str):
    """Converts video to audio using MoviePy library that uses `ffmpeg` under the hood.
    
    :param video_path: file path of video file (.mp4)
    :param output_audio_path: file path of audio file (.wav) to be created
    """
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)
    video_clip.close()
    audio_clip.close()


def load_whisper_model(model_name: str = "base"):
    """Load a whisper model for generating video transcripts"""
    return whisper.load_model(model_name)


def extract_transcript_from_audio(whisper_model, audio_path: str):
    """Generate transcript from audio file
    
    :param whisper_model: a pre-loaded whisper model object
    :param audio_path: file path of audio file (.wav)
    """
    options = dict(task="translate", best_of=5, language='en')
    return whisper_model.transcribe(audio_path, **options)


def format_timestamp_for_transcript(seconds: float, always_include_hours: bool = True, fractionalSeperator: str = '.'):
    """Format timestamp for video transcripts"""
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"


def write_vtt(transcript: Iterator[dict], vtt_path: str):
    """Write transcripts to a .vtt file"""
    with open(vtt_path, 'a') as file:
        file.write("WEBVTT\n\n")
        for segment in transcript['segments']:
            text = (segment['text']).replace('-->', '->')
            file.write(f"{format_timestamp_for_transcript(segment['start'])} --> {format_timestamp_for_transcript(segment['end'])}\n")
            file.write(f"{text.strip()}\n\n")


def delete_audio_file(audio_path: str):
    """Delete audio file after extracting transcript"""
    os.remove(audio_path)


def time_to_frame(time: float, fps: float):
    """Convert time in seconds into frame number"""
    return int(time * fps - 1)


def str2time(strtime: str):
    """Get time in seconds from string"""
    strtime = strtime.strip('"')
    hrs, mins, seconds = [float(c) for c in strtime.split(':')]

    total_seconds = hrs * 60**2 + mins * 60 + seconds

    return total_seconds


def convert_img_to_base64(image):
    "Convert image to base64 string"
    _, buffer = cv2.imencode('.jpg', image)
    encoded_string = base64.b64encode(buffer)
    return encoded_string.decode()


def extract_frames_and_annotations_from_transcripts(video_id: str, video_path: str, vtt_path: str, output_dir: str):
    """Extract frames (.jpg) and annotations (.json) from video file (.mp4) and captions file (.vtt)"""
    # Set up location to store frames and annotations
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)

    # Load video and get fps
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    # read captions file
    captions = webvtt.read(vtt_path)

    annotations = []
    for idx, caption in enumerate(captions):
        start_time = str2time(caption.start)
        end_time = str2time(caption.end)

        mid_time = (end_time + start_time) / 2
        text = caption.text.replace('\n', ' ')

        frame_no = time_to_frame(mid_time, fps)
        mid_time_ms = mid_time * 1000 
        vidcap.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        success, frame = vidcap.read()
        
        if success:
            # Save frame for further processing
            img_fname = f"frame_{idx}"
            img_fpath = os.path.join(output_dir, 'frames', img_fname + '.jpg')
            cv2.imwrite(img_fpath, frame)

            # Convert image to base64 encoded string
            b64_img_str = convert_img_to_base64(frame)

            # Create annotations for frame from transcripts
            annotations.append({
                'video_id': video_id,
                'video_name' : os.path.basename(video_path),
                'b64_img_str': b64_img_str,
                'caption': text,
                'time': mid_time_ms,
                'frame_no': frame_no,
                'sub_video_id': idx,
            })
    
    # Save transcript annotations as json file for further processing
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f)
    
    vidcap.release()


def use_lvm(endpoint: str, img_b64_string: str, prompt: str ="Provide a short description for this scene."):
    """Generate image captions/descriptions using LVM microservice"""
    inputs = {"image": img_b64_string, "prompt": prompt, "max_new_tokens": 32}
    response = requests.post(url=endpoint, data=json.dumps(inputs))
    return response.json()["text"]


def extract_frames_and_generate_captions(video_id: str, video_path: str, lvm_endpoint: str, output_dir: str, key_frame_per_second: int = 1):
    """Extract frames (.jpg) and annotations (.json) from video file (.mp4) by generating captions using LVM microservice"""
    # Set up location to store frames and annotations
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)

    # Load video and get fps
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    annotations = []
    hop = round(fps / key_frame_per_second)
    curr_frame = 0
    idx = -1
    
    while True:
        ret, frame = vidcap.read()
        if not ret: 
            break
        
        if curr_frame % hop == 0:
            idx += 1

            mid_time = vidcap.get(cv2.CAP_PROP_POS_MSEC)
            mid_time_ms = mid_time * 1000

            frame_no = curr_frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save frame for further processing
            img_fname = f"frame_{idx}"
            img_fpath = os.path.join(output_dir, 'frames', img_fname + '.jpg')
            cv2.imwrite(img_fpath, frame)

            # Convert image to base64 encoded string
            b64_img_str = convert_img_to_base64(frame)

            # Caption generation using LVM microservice
            caption = use_lvm(lvm_endpoint, b64_img_str)
            caption = caption.strip()
            text = caption.replace('\n', ' ')


            # Create annotations for frame from transcripts
            annotations.append({
                'video_id': video_id,
                'video_name' : os.path.basename(video_path),
                'b64_img_str': b64_img_str,
                'caption': text,
                'time': mid_time_ms,
                'frame_no': frame_no,
                'sub_video_id': idx,
            })
        
        curr_frame += 1

    # Save caption annotations as json file for further processing
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f)
    
    vidcap.release()