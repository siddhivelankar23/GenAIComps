# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import torch
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union, Tuple
from torch import nn

import torch.nn.functional as F
import torchvision.transforms.functional as transform
from config import EMBED_MODEL, INDEX_NAME, INDEX_SCHEMA, REDIS_URL
from fastapi import File, HTTPException, UploadFile
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils import get_from_dict_or_env
from langchain_community.vectorstores import Redis
from langchain_community.vectorstores.redis.base import _generate_field_schema, _prepare_metadata
from langchain_community.utilities.redis import _array_to_buffer
from langsmith import traceable
from torchvision.io import ImageReadMode, read_image
from transformers import BridgeTowerProcessor, BridgeTowerPreTrainedModel, BridgeTowerModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bridgetower.modeling_bridgetower import BridgeTowerTextModel

from comps import opea_microservices, register_microservice
from comps.dataprep.multimodal_utils import (
    create_upload_folder,
    load_json_file,
    convert_video_to_audio,
    load_whisper_model,
    extract_transcript_from_audio,
    write_vtt,
    delete_audio_file,
    extract_frames_and_annotations
)

try: 
    import habana_frameworks.torch.core as htcore
    device = "hpu"
except Exception as e: 
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


upload_folder = "./uploaded_files/"
whisper_model = load_whisper_model(model_name="base")


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
        TEXT_MODEL = BridgeTowerTextModel(config)

    def forward(self, **kwargs):
        return TEXT_MODEL(**kwargs)


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


TEXT_MODEL = BridgeTowerTextFeatureExtractor.from_pretrained(EMBED_MODEL).to(device)
PROCESSOR = BridgeTowerProcessor.from_pretrained(EMBED_MODEL)
MODEL = BridgeTowerForITC.from_pretrained(EMBED_MODEL).to(device)


class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """ BridgeTower embedding model """
    
    # TODO(tile): This does not work if batch and if texts have different length. 
    # Should implement as we did in FinetunedBridgeTowerEmbeddingChestXraypy
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BridgeTower.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        encodings = PROCESSOR.tokenizer(texts, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = TEXT_MODEL(**encodings)
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
            # print(path_to_img)
            img = read_image(path_to_img, mode=ImageReadMode.RGB)
            img = transform.to_pil_image(img)
            image_list.append(img)
            text_list.append(text)
            if len(text_list) == batch_size:
                batch = PROCESSOR(image_list, text_list, return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(device)
                with torch.no_grad():
                    batch_embeddings = MODEL(**batch, output_hidden_states=True)
                for i in range(len(text_list)):
                    embeddings.append(batch_embeddings.logits[i,2,:].detach().cpu().numpy().tolist())
                image_list = []
                text_list = []
        # embedding the remaining        
        if len(text_list) > 0:
            batch = PROCESSOR(image_list, text_list, return_tensors='pt', max_length=100, padding='max_length', truncation=True).to(device)
            with torch.no_grad():
                batch_embeddings = MODEL(**batch, output_hidden_states=True)
            for i in range(len(text_list)):
                embeddings.append(batch_embeddings.logits[i,2,:].detach().cpu().numpy().tolist())
            image_list = []
            text_list = []
        return embeddings



class MultimodalRedis(Redis):
    """ Redis vector database to process multimodal data"""

    @classmethod
    def from_text_image_pairs_return_keys(
        cls: Type[Redis],
        texts: List[str],
        images: List[str],
        embedding: Embeddings = BridgeTowerEmbeddings,
        metadatas: Optional[List[dict]] = None,
        index_name: Optional[str] = None,
        index_schema: Optional[Union[Dict[str, str], str, os.PathLike]] = None,
        vector_schema: Optional[Dict[str, Union[str, int]]] = None,
        **kwargs: Any, 
    ):
        """
        Args:
            texts (List[str]): List of texts to add to the vectorstore.
            images (List[str]): List of path-to-images to add to the vectorstore.
            embedding (Embeddings): Embeddings to use for the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadata
                dicts to add to the vectorstore. Defaults to None.
            index_name (Optional[str], optional): Optional name of the index to
                create or add to. Defaults to None.
            index_schema (Optional[Union[Dict[str, str], str, os.PathLike]], optional):
                Optional fields to index within the metadata. Overrides generated
                schema. Defaults to None.
            vector_schema (Optional[Dict[str, Union[str, int]]], optional): Optional
                vector schema to use. Defaults to None.
            **kwargs (Any): Additional keyword arguments to pass to the Redis client.

        Returns:
            Tuple[Redis, List[str]]: Tuple of the Redis instance and the keys of
                the newly created documents.

        Raises:
            ValueError: If the number of metadatas does not match the number of texts.
        """
        # the length of texts must be equal to the length of images
        assert len(texts)==len(images), "the len of captions should be equal to the len of images"
        
        redis_url = get_from_dict_or_env(kwargs, "redis_url", "REDIS_URL")

        if "redis_url" in kwargs:
            kwargs.pop("redis_url")

        # flag to use generated schema
        if "generate" in kwargs:
            kwargs.pop("generate")

        # see if the user specified keys
        keys = None
        if "keys" in kwargs:
            keys = kwargs.pop("keys")

        # Name of the search index if not given
        if not index_name:
            index_name = uuid.uuid4().hex

        # type check for metadata
        if metadatas:
            if isinstance(metadatas, list) and len(metadatas) != len(texts):  # type: ignore  # noqa: E501
                raise ValueError("Number of metadatas must match number of texts")
            if not (isinstance(metadatas, list) and isinstance(metadatas[0], dict)):
                raise ValueError("Metadatas must be a list of dicts")
            generated_schema = _generate_field_schema(metadatas[0])
        
        if index_schema:
            index_schema = generated_schema

        # Create instance
        instance = cls(
            redis_url,
            index_name,
            embedding,
            index_schema=index_schema,
            vector_schema=vector_schema,
            **kwargs,
        )
        # Add data to Redis
        keys = instance.add_text_image_pairs(texts, images, metadatas, keys=keys)
        return instance, keys

    def add_text_image_pairs(
        self,
        texts: Iterable[str],
        images: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 2,
        clean_metadata: bool = True,
        **kwargs: Any,
    ) -> List[str]:    
    
        """Add more embeddings of text-image pairs to the vectorstore.

        Args:
            texts (Iterable[str]): Iterable of strings/text to add to the vectorstore.
            images: Iterable[str]: Iterable of strings/text of path-to-image to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
                Defaults to None.
            embeddings (Optional[List[List[float]]], optional): Optional pre-generated
                embeddings. Defaults to None.
            keys (List[str]) or ids (List[str]): Identifiers of entries.
                Defaults to None.
            batch_size (int, optional): Batch size to use for writes. Defaults to 1000.

        Returns:
            List[str]: List of ids added to the vectorstore
        """
        ids = []
        # Get keys or ids from kwargs
        # Other vectorstores use ids
        keys_or_ids = kwargs.get("keys", kwargs.get("ids"))
        
        # type check for metadata
        if metadatas:
            if isinstance(metadatas, list) and len(metadatas) != len(texts):  # type: ignore  # noqa: E501
                raise ValueError("Number of metadatas must match number of texts")
            if not (isinstance(metadatas, list) and isinstance(metadatas[0], dict)):
                raise ValueError("Metadatas must be a list of dicts")
        embeddings = embeddings or self._embeddings.embed_image_text_pairs(list(texts), list(images), batch_size=batch_size)
        self._create_index_if_not_exist(dim=len(embeddings[0]))
        
        # Write data to redis
        pipeline = self.client.pipeline(transaction=False)
        for i, text in enumerate(texts):
            # Use provided values by default or fallback
            key = keys_or_ids[i] if keys_or_ids else str(uuid.uuid4().hex)
            if not key.startswith(self.key_prefix + ":"):
                key = self.key_prefix + ":" + key
            metadata = metadatas[i] if metadatas else {}
            metadata = _prepare_metadata(metadata) if clean_metadata else metadata
            pipeline.hset(
                key,
                mapping={
                    self._schema.content_key: text,
                    self._schema.content_vector_key: _array_to_buffer(
                        embeddings[i], self._schema.vector_dtype
                    ),
                    **metadata,
                },
            )
            ids.append(key)

            # Write batch
            if i % batch_size == 0:
                pipeline.execute()

        # Cleanup final batch
        pipeline.execute()
        return ids


def prepare_data_and_metadata_from_annotation(annotation, path_to_frames, title, description, num_transcript_concat_for_ingesting=2, num_transcript_concat_for_inference=7):
    text_list = []
    image_list = []
    metadatas = []
    for i, frame in enumerate(annotation):
        frame_index = frame['sub_video_id']
        path_to_frame = os.path.join(path_to_frames, f"frame_{frame_index}.jpg")
        lb_ingesting = max(0, i-num_transcript_concat_for_ingesting)
        ub_ingesting = min(len(annotation), i+num_transcript_concat_for_ingesting+1) 
        caption_for_ingesting = ' '.join([annotation[j]['caption'] for j in range(lb_ingesting, ub_ingesting)])

        lb_inference = max(0, i-num_transcript_concat_for_inference)
        ub_inference = min(len(annotation), i+num_transcript_concat_for_inference+1) 
        caption_for_inference = ' '.join([annotation[j]['caption'] for j in range(lb_inference, ub_inference)])
        
        time_of_frame = frame['time']
        embedding_type = 'pair'
        text_list.append(caption_for_ingesting)
        image_list.append(path_to_frame)
        metadatas.append({
            'content' : caption_for_ingesting,
            'source' : path_to_frame,
            'time_of_frame_ms' : float(time_of_frame),
            'embedding_type' : embedding_type,
            'title' : title,
            'description' : description,
            'transcript_for_inference' : caption_for_inference,
        })
    return text_list, image_list, metadatas


def ingest_multimodal(title, title_for_embedding, description, data_folder):
    """
    Ingest text image pairs to Redis from the data/ directory that consists of frames and annotations
    """
    data_folder = os.path.abspath(data_folder)
    annotation_file_path = os.path.join(data_folder, 'annotations.json')
    path_to_frames = os.path.join(data_folder, 'frames')

    annotation = load_json_file(annotation_file_path)

    #prepare data to ingest
    text_list, image_list, metadatas = prepare_data_and_metadata_from_annotation(annotation, path_to_frames, title, description)
    
    # Create vectorstore
    embedder = BridgeTowerEmbeddings(model_name=EMBED_MODEL, device=device)
    
    MultimodalRedis.from_text_image_pairs_return_keys(
        texts=[f"From {title_for_embedding}. " + text for text in text_list],
        images=image_list,
        embedding=embedder,
        metadatas=metadatas,
        index_name=INDEX_NAME,
        index_schema=INDEX_SCHEMA,
        redis_url=REDIS_URL,
    )


def drop_index(index_name, redis_url=REDIS_URL):
    print(f"[ drop index ] dropping index {index_name}")
    try:
        assert Redis.drop_index(index_name=index_name, delete_documents=True, redis_url=redis_url)
        print(f"[ drop index ] index {index_name} deleted")
    except Exception as e:
        print(f"[ drop index ] index {index_name} delete failed: {e}")
        return False
    return True


@register_microservice(name="opea_service@prepare_doc_redis", endpoint="/v1/dataprep", host="0.0.0.0", port=6007)
@traceable(run_type="tool")
async def ingest_videos(
    files:  List[UploadFile] = File(None)
):
    print(f"files:{files}")

    if files:
        video_files = []
        captions_files = []
        for file in files:
            if os.path.splitext(file.filename)[1] == ".mp4":
                video_files.append(file)
            elif os.path.splitext(file.filename)[1] == ".vtt":
                captions_files.append(file)
            else:
                print(f"Skipping file {file.filename} because of unsupported format.")
        
        if len(video_files) == 0:
            return HTTPException(status_code=400, detail="The uploaded files have unsupported formats. Please upload atleast one video file (.mp4) with or without captions (.vtt)")

        for video_file in video_files:
            video_file_name = os.path.splitext(video_file.filename)[0]
            
            # Save video file in upload_directory
            with open(os.path.join(upload_folder, video_file.filename), 'wb') as f:
                shutil.copyfileobj(video_file.file, f)

            # Check if corresponding vtt file has been uploaded
            vtt_file = video_file_name + ".vtt"
            vtt_idx = None
            for idx, caption_file in enumerate(captions_files):
                if caption_file.filename == vtt_file:
                    vtt_idx = idx
                    break

            if vtt_idx is not None:
                # Save captions file in upload_directory
                with open(os.path.join(upload_folder, vtt_file), 'wb') as f:
                    shutil.copyfileobj(captions_files[vtt_idx].file, f)
            else:
                # Convert mp4 to temporary wav file
                audio_file = video_file_name + ".wav"
                convert_video_to_audio(os.path.join(upload_folder, video_file.filename), os.path.join(upload_folder, audio_file))
                
                # Extract transcript from audio
                transcripts = extract_transcript_from_audio(whisper_model, os.path.join(upload_folder, audio_file))

                # Save transcript as vtt file and delete audio file
                write_vtt(transcripts, os.path.join(upload_folder, vtt_file))
                delete_audio_file

            # Store frames and caption annotations in a new directory
            extract_frames_and_annotations(os.path.join(upload_folder, video_file.filename), os.path.join(upload_folder, vtt_file), os.path.join(upload_folder, video_file_name))

            print(f"Stored frames and annotations in {os.path.join(upload_folder, video_file_name)}")
        
            # Delete temporary video and captions files
            os.remove(os.path.join(upload_folder, video_file.filename))
            os.remove(os.path.join(upload_folder, vtt_file))
        
            # Ingest multimodal data into redis
            ingest_multimodal(video_file_name, video_file_name, video_file_name, os.path.join(upload_folder, video_file_name))
        
        return {"status": 200, "message": "Data preparation succeeded"}

    raise HTTPException(status_code=400, detail="Must provide atleast one video (.mp4) file with or without captions.")


@register_microservice(
    name="opea_service@prepare_doc_redis_file", endpoint="/v1/dataprep/get_file", host="0.0.0.0", port=6008
)
@traceable(run_type="tool")
async def rag_get_file_structure():
    """Returns list of names of uploaded videos"""

    if not Path(upload_folder).exists():
        print("No file uploaded, return empty list.")
        return []

    uploaded_videos = os.listdir(upload_folder)
    return uploaded_videos


@register_microservice(
    name="opea_service@prepare_doc_redis_del", endpoint="/v1/dataprep/delete_file", host="0.0.0.0", port=6009
)
@traceable(run_type="tool")
async def delete_videos():
    """Delete all uploaded videos along with redis index"""
    index_deleted = drop_index(index_name=INDEX_NAME)

    if not index_deleted:
        raise HTTPException(status_code=409, detail="Uploaded videos could not be deleted.")

    shutil.rmtree(upload_folder)
    print("Successfully deleted all uploaded videos.")
    create_upload_folder(upload_folder)
    return {"status": True}


if __name__ == "__main__":
    create_upload_folder(upload_folder)
    opea_microservices["opea_service@prepare_doc_redis"].start()
    opea_microservices["opea_service@prepare_doc_redis_file"].start()
    opea_microservices["opea_service@prepare_doc_redis_del"].start()