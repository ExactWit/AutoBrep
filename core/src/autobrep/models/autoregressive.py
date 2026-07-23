
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from transformers import AutoModel, AutoTokenizer

from autobrep.data.token_mapping import MMTokenIndex
from autobrep.models.autoregressive_samplers import TopP
from autobrep.models.dataclass import BrepGenCAD, CheckpointPaths
from autobrep.models.generate_prepend import generate_with_prepend_kv_cache
from autobrep.models.point_cloud_encoder import PointCloudConditionEncoder
from autobrep.models.view_condition_encoder import ViewConditionEncoder
from autobrep.models.vaes import EdgeFSQVAE, SurfaceFSQVAE
from autobrep.network import XTransformer
from autobrep.utils import compute_bbox_center_and_size, dequantize
from x_transformers.autoregressive_wrapper import min_p, top_a, top_k, top_p
from functools import partial
from smart_open import open
import torch.nn.functional as F
from autobrep.utils import timer
from pathlib import Path
import torch.nn as nn


@dataclass
class ARGenCheckpointPaths(CheckpointPaths):
    surface_fsq: str = "surf-fsq.ckpt"
    edge_fsq: str = "edge-fsq.ckpt"
    autoregressive: str = "ar.ckpt"

    def __iter__(self):
        return iter(
            (
                self.surface_fsq,
                self.edge_fsq,
                self.autoregressive,
            )
        )


class AutoRegressiveSampler(LightningModule):
    class Complexity(Enum):
        easy = MMTokenIndex.GEN_EASY
        medium = MMTokenIndex.GEN_MID
        hard = MMTokenIndex.GEN_HARD
        uncond = MMTokenIndex.GEN_UNCOND

    def __init__(
        self,
        checkpoint_paths: ARGenCheckpointPaths,
        device: torch.device,
        multimodal: bool = False,
        pc_conditioned: bool = False,
        pc_ckpt: Optional[str] = None,
        view_conditioned: bool = False,
        view_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pc_conditioned = pc_conditioned
        self.view_conditioned = view_conditioned

        # Load vae
        self.surface_fsq = SurfaceFSQVAE.load_from_checkpoint(
            checkpoint_paths.surface_fsq
        ).drop_encoder() 
        self.surface_fsq.to(device=device).eval()
    
        self.edge_fsq = EdgeFSQVAE.load_from_checkpoint(
            checkpoint_paths.edge_fsq
        ).drop_encoder() 
        self.edge_fsq.to(device=device).eval()

        # Load AR (optionally PC- or view-conditioned)
        if view_conditioned:
            ckpt = view_ckpt or checkpoint_paths.autoregressive
            self.transformer = AutoBrepViewModel.load_from_checkpoint(
                ckpt,
                inference_mode=True,
                strict=False,
                map_location="cpu",
            )
        elif pc_conditioned:
            ckpt = pc_ckpt or checkpoint_paths.autoregressive
            self.transformer = AutoBrepPCModel.load_from_checkpoint(
                ckpt,
                inference_mode=True,
                strict=False,
                map_location="cpu",
            )
        else:
            self.transformer = AutoBrepModel.load_from_checkpoint(
                checkpoint_paths.autoregressive,
                inference_mode=True,
                strict=False,
            )
        self.transformer.to(device=device, dtype=torch.float16).eval() # half precision

    @torch.inference_mode()
    def sample_tokens(
        self,
        config,
        batch_size: int = 8,
        point_cloud: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        prim_types: Optional[torch.Tensor] = None,
        prim_linetypes: Optional[torch.Tensor] = None,
        prim_geom: Optional[torch.Tensor] = None,
        prim_mask: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
            samples: A list of sampled tokens
        """
        cx_mode = str(config.hyper_parameters.complexity).lower()
        gen_kwargs = {}
        if self.view_conditioned:
            if images is None:
                raise ValueError("view_conditioned sampler requires images (B,3,3,H,W)")
            if prim_types is None or prim_mask is None:
                raise ValueError("view_conditioned sampler requires TechDraw DXF tensors")
            gen_kwargs["images"] = images.to(
                device=self.transformer.device, dtype=torch.float16
            )
            gen_kwargs["prim_types"] = prim_types.to(device=self.transformer.device)
            gen_kwargs["prim_linetypes"] = prim_linetypes.to(
                device=self.transformer.device
            )
            gen_kwargs["prim_geom"] = prim_geom.to(
                device=self.transformer.device, dtype=torch.float16
            )
            gen_kwargs["prim_mask"] = prim_mask.to(device=self.transformer.device)
        elif self.pc_conditioned:
            if point_cloud is None:
                raise ValueError("pc_conditioned sampler requires point_cloud (B,N,3)")
            gen_kwargs["point_cloud"] = point_cloud.to(
                device=self.transformer.device, dtype=torch.float16
            )

        if cx_mode in ("from_condition", "auto", "cond"):
            if not self.view_conditioned:
                raise ValueError(
                    "complexity=from_condition requires view_conditioned sampler"
                )
            # Per-sample AR next-token among {easy,mid,hard} given view+DXF prepend.
            cx_ids = []
            for b in range(batch_size):
                c = int(
                    self.transformer.predict_complexity_from_condition(
                        images=gen_kwargs["images"][b : b + 1],
                        prim_types=gen_kwargs["prim_types"][b : b + 1],
                        prim_linetypes=gen_kwargs["prim_linetypes"][b : b + 1],
                        prim_geom=gen_kwargs["prim_geom"][b : b + 1],
                        prim_mask=gen_kwargs["prim_mask"][b : b + 1],
                    )
                )
                cx_ids.append(c)
        else:
            if cx_mode == "random":
                complexity = 17
            elif cx_mode == "easy":
                complexity = 14
            elif cx_mode == "medium":
                complexity = 15
            elif cx_mode == "hard":
                complexity = 16
            else:
                complexity = 15
            cx_ids = [complexity] * batch_size

        prompt_rows = []
        for c in cx_ids:
            prompt_rows.extend(
                [
                    MMTokenIndex.BOS.value,
                    MMTokenIndex.BOM.value,
                    c,
                    MMTokenIndex.EOM.value,
                    MMTokenIndex.BOC.value,
                ]
            )
        prompt = (
            torch.LongTensor(prompt_rows)
            .reshape(batch_size, 5)
            .to(self.transformer.device)
        )

        with timer(f"Total Time to generate B-Reps: %s seconds"):
            samples = self.transformer.generate(
                prompt, 
                config.hyper_parameters.temperature, 
                config.hyper_parameters.sample_method.top_p_threshold,
                **gen_kwargs,
            )

        return torch.concat([prompt, samples], -1)

    def decode_tokens(self, samples):
        """

        Args:
            samples: A list of sampled tokens

        Returns:
            batch_decoded: decoded tokens to bbox pos and fsq codes

        """
        batch_decoded = []
        for sample in samples:
            # Parse tokens
            try:
                geom_tokens = None
                if MMTokenIndex.BOGEOM.value in sample:
                    geom_s = np.where(sample==MMTokenIndex.BOGEOM.value)[0][0]
                    geom_e = np.where(sample==MMTokenIndex.EOGEOM.value)[0][0]
                    geom_tokens = sample[geom_s+1:geom_e]

                if MMTokenIndex.BOC.value in sample:
                    cad_s = np.where(sample==MMTokenIndex.BOC.value)[0][0]
                    cad_e = np.where(sample==MMTokenIndex.EOC.value)[0][0]
                    cad_tokens = sample[cad_s+1:cad_e]

                    # Include geometry tokens in decoding 
                    if geom_tokens is not None:
                        cad_tokens = np.concatenate((geom_tokens, cad_tokens))

                pos_faces, code_faces, pos_edges, code_edges, face_edge_adj = (
                    self.transformer.decode(cad_tokens) 
                )
            except Exception as err:
                print("Error in decoding tokens: %s", err)
                continue

            # Decode uv grid
            with torch.no_grad():
                # Face uv (ncs)
                geomZ_faces = self.surface_fsq.quantizer.indices_to_codes(
                    torch.LongTensor(code_faces).to(self.surface_fsq.device)
                ).permute(0, 2, 1)
                uv_ncs_faces = self.surface_fsq.decode(
                    geomZ_faces.unflatten(-1, (2, 2))
                ).sample
                uv_ncs_faces = (
                    rearrange(uv_ncs_faces, "b d ... -> b ... d").float().cpu().numpy()
                )
                # Edge uv (ncs)
                geomZ_edges = self.edge_fsq.quantizer.indices_to_codes(
                    torch.LongTensor(code_edges).to(self.edge_fsq.device)
                ).permute(0, 2, 1)
                uv_ncs_edges = self.edge_fsq.decode(geomZ_edges).sample
                uv_ncs_edges = (
                    rearrange(uv_ncs_edges, "b d ... -> b ... d")
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )

            batch_decoded.append(
                (pos_faces, pos_edges, uv_ncs_edges, uv_ncs_faces, face_edge_adj)
            )

        return batch_decoded

    @staticmethod
    def convert_to_cad_data(batch_data) -> List[BrepGenCAD]:
        """

        Args:
            batch_data: a batch of decoded tokens

        Returns:
            batch_cad_data: a batch of converted data of cad data format

        """
        batch_cad_data = []
        for cad_data in batch_data:
            pos_faces, pos_edges, uv_ncs_edges, uv_ncs_faces, face_edge_adj = cad_data

            num_faces = pos_faces.shape[0]
            assert uv_ncs_faces.shape[0] == num_faces, "Check num faces"
            assert face_edge_adj.shape[0] == num_faces, "Check num faces"

            num_edges = pos_edges.shape[0]
            assert num_edges == uv_ncs_edges.shape[0], (
                "Check number of edges is consistent"
            )
            assert num_edges == face_edge_adj.shape[1], (
                "Check number of edges is consistent"
            )

            # convert to old edge format of Face x Edge
            edge_pos_cad = np.zeros((face_edge_adj.shape[0], face_edge_adj.shape[1], 6))
            edge_ncs_cad = np.zeros(
                (face_edge_adj.shape[0], face_edge_adj.shape[1], 32, 3)
            )

            for face_id, edge_row in enumerate(face_edge_adj):
                edge_pos_cad[face_id, edge_row] = pos_edges[np.where(edge_row)[0]]
                edge_ncs_cad[face_id, edge_row] = uv_ncs_edges[np.where(edge_row)[0]]

            edge_wcs_cad = []
            for face_edge_pos, face_edge_ncs in zip(edge_pos_cad, edge_ncs_cad):
                wcs = []
                for pos, ncs in zip(face_edge_pos, face_edge_ncs):
                    center, size = compute_bbox_center_and_size(pos[0:3], pos[3:])
                    wcs.append(ncs * (size / 2) + center)
                edge_wcs_cad.append(np.stack(wcs))
            edge_wcs_cad = np.stack(edge_wcs_cad)

            # convert to old vertex format (start/end point of edge)
            vertex_cad = edge_wcs_cad[:, :, [0, -1]]
            vertex_cad = vertex_cad.reshape(
                vertex_cad.shape[0], vertex_cad.shape[1], -1
            )

            cad_data = BrepGenCAD(
                face_pos_cad=pos_faces,
                edge_pos_cad=pos_edges,
                face_ncs_cad=uv_ncs_faces,
                edge_ncs_cad=uv_ncs_edges,
                edge_mask_cad=~face_edge_adj,
                # Keep the legacy data.   Need to remove this ASAP
                vertex_cad_legacy=vertex_cad,
                edge_pos_cad_legacy=edge_pos_cad,
                edge_ncs_cad_legacy=edge_ncs_cad,
            )

            batch_cad_data.append(cad_data)

        return batch_cad_data

    def encode_surface(self, face_ncs):
        _, surf_id = self.surface_fsq.encode(face_ncs.permute(0, 3, 1, 2))
        surf_id = surf_id.flatten(-2, -1)
        return surf_id

    def encode_edge(self, edge_ncs):
        _, edge_id = self.edge_fsq.encode(edge_ncs.permute(0, 2, 1))
        return edge_id


class BrepBase(LightningModule):
    def __init__(self):
        super().__init__()

    def common_step(self, batch):
        pass

    def decode(self, sample: np.ndarray):
        pass

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        total_loss = self.common_step(batch)
        return total_loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        total_loss = self.common_step(batch)

    def load_vae(self, surf_fsq_ckpt, edge_fsq_ckpt, drop_decoder: bool = True):
        self.surf_vae = SurfaceFSQVAE.load_from_checkpoint(
            surf_fsq_ckpt, 
            map_location="cpu",
            use_dcae=True
        )
        # Freeze the VAE explicitly to work with DDP strategy
        self.surf_vae.requires_grad_(False)
        self.surf_vae.eval()

        self.edge_vae = EdgeFSQVAE.load_from_checkpoint(
            edge_fsq_ckpt, 
            map_location="cpu",
            use_dcae=True
        )
        # Freeze the VAE explicitly to work with DDP strategy
        self.edge_vae.requires_grad_(False)
        self.edge_vae.eval()

        if drop_decoder:
            self.surf_vae.drop_decoder()
            self.edge_vae.drop_decoder()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.AdamW(
            self.trainable_params,
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay,
            eps=1e-5,
        )

    def encode_fsq_code(self, face_ncs, edge_ncs):
        # Encode to fsq code
        bs = face_ncs.size(0)
        _, edge_id = self.edge_vae.encode(edge_ncs.flatten(0, 1).permute(0, 2, 1))
        edge_id = edge_id.unflatten(0, (bs, -1)).long()
        _, surf_id = self.surf_vae.encode(face_ncs.flatten(0, 1).permute(0, 3, 1, 2))
        surf_id = surf_id.flatten(-2, -1).unflatten(0, (bs, -1)).long()
        return surf_id, edge_id

    def copy_fsq_code(self, token, surf_id, edge_id):
        batch_data = -torch.ones(self.pad_len, 4).long().to(token.device)
        batch_data[:, 0] = token

        # Replace face code
        face_z_indices = (token >= self.face_z_pad) & (token < self.edge_z_pad)
        batch_data[face_z_indices] = (
            surf_id[token[face_z_indices] - self.face_z_pad] + self.face_z_pad
        )

        # Replace edge code
        edge_z_indices = token >= self.edge_z_pad
        batch_data[edge_z_indices, :2] = edge_id[
            token[edge_z_indices] - self.edge_z_pad
        ] + (self.face_z_pad + self.hparams.surf_codebook_size)

        # remove unused tokens
        batch_data = batch_data.flatten()
        batch_data = batch_data[batch_data >= 0]
        return batch_data


class AutoBrepModel(BrepBase):
    def __init__(
        self,
        surf_fsq_ckpt: str = None,
        edge_fsq_ckpt: str = None,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        sync_dist_train: bool = True,
        bit: int = 10,
        max_seq: int = 2500,
        max_face: int = 100,  # Saving as a hyperparameter and copying over from datamodule args
        inference_mode: bool = False,
        depth: int = 12,
        heads: int = 12,
        dim: int = 768,
        kv_groups: int = 4,
        surf_codebook_size: int = 10000,
        edge_codebook_size: int = 10000,
        drop_decoder: bool = True,
    ) -> None:
        # strict_loading = False
        super().__init__()
        self.save_hyperparameters()
       
        flag_pad = len(MMTokenIndex.__members__)
        id_pad = max_face
        pos_pad = 2**bit
        self.pad_len = self.hparams.max_seq
        self.face_z_pad = pos_pad + id_pad + flag_pad
        self.edge_z_pad = self.face_z_pad + max_face
        self.cad_pad = self.face_z_pad + surf_codebook_size + edge_codebook_size

        if not inference_mode:
            self.load_vae(surf_fsq_ckpt, edge_fsq_ckpt, drop_decoder)

        self.cad_gpt = XTransformer(
            max_seq=self.hparams.max_seq,  
            num_tokens=self.face_z_pad
            + self.hparams.surf_codebook_size
            + self.hparams.edge_codebook_size,
            depth=depth,
            heads=heads,
            dim=dim,
            kv_groups=kv_groups,
        )
        self.trainable_params = list(self.cad_gpt.parameters())

    def common_step(self, batch):
        token, face_ncs, edge_ncs = (
            batch["seq"],
            batch["face_ncs"].to(dtype=torch.bfloat16),
            batch["edge_ncs"].to(dtype=torch.bfloat16),
        )

        with torch.no_grad():
            # Encode to fsq code
            surf_id, edge_id = self.encode_fsq_code(face_ncs, edge_ncs)
           
        # Update tokens, replace z index with actual fsq code
        updated_token, loss_mask, attn_mask = [], [], []
        for _token_, _surf_id_, _edge_id_ in zip(token, surf_id, edge_id):
            batch_data = self.copy_fsq_code(_token_, _surf_id_, _edge_id_)
            # pad data
            updated_token.append(
                torch.nn.functional.pad(
                    batch_data,
                    (0, self.pad_len - len(batch_data)),
                    value=-1,
                )
            )

            # loss mask, ignore user conditional data
            mask = torch.zeros(self.pad_len).bool()
            mask[:4] = True
            loss_mask.append(mask)

            # Randomly dropout previous levels for uncond generation
            local_mask = torch.ones(
                (len(batch_data), len(batch_data))
            ).bool().to(batch_data.device)
            level_splits = torch.where(batch_data == MMTokenIndex.EOL.value)[0]
            if len(level_splits) >= 3:
                for level_idx in range(len(level_splits)):
                    # Dropout all previous levels 
                    if level_idx>=2 and np.random.rand()<=0.1:
                        local_mask[
                            level_splits[level_idx-1]+1 : level_splits[level_idx]+1, 
                            0 : level_splits[level_idx-2]+1
                        ] = False

            padding = (0, self.pad_len-len(batch_data), 0, self.pad_len-len(batch_data))
            attn_mask.append(
                F.pad(local_mask, padding, mode='constant', value=False)
            )

        updated_token = torch.stack(updated_token).detach()

        attn_mask = torch.stack(attn_mask).detach()
        loss_mask = torch.stack(loss_mask).detach()
     
        # Pass through transformer module
        loss = self.cad_gpt(
            updated_token,
            cond_mask=loss_mask,
            attn_mask=None,
        )
        return loss

    def decode(
        self,
        sample: np.ndarray,
    ):
        """

        Decode the position, geometry, and adjaceny matrix
        from the generated b-rep tokens

        Args:
            sample (numpy[int]): a list of generated brep tokens
        Return:
            pos_faces: face bbox in wcs
            geomCode_faces: face geometry fsq codes
            pos_edges: edge bbox in wcs
            geomCode_edges: edge geometry fsq codes
            face_edge_adj: face-edge bool adjacency matrix

        """
        flag_pad = len(MMTokenIndex.__members__)
        id_pad = self.hparams.max_face
        pos_pad = 2**self.hparams.bit

        # Remove invalid tokens and divide per-face
        valid_sample = sample[sample > 0]
        level_split = np.where(valid_sample == MMTokenIndex.BOL.value)[0]
        cad_levels = np.split(valid_sample, level_split)[1:]
        
        pos_faces = []
        geomCode_faces = []
        pos_edges = []
        geomCode_edges = []
        edge_face_pair = []
        level_faces = []

        face_count = 0
        for level_idx, cad_level in enumerate(cad_levels):
            face_indices = []
            cad_level = cad_level[1:-1] # remove start/end of level
            face_split = np.where(cad_level == MMTokenIndex.BOF.value)[0]
            cad_faces = np.split(cad_level, face_split)[1:]

            face_pos = np.stack([x[1:7] - flag_pad - id_pad for x in cad_faces])
            code_faces = np.stack(
                [x[7:11] - flag_pad - id_pad - pos_pad for x in cad_faces]
            )
            pos_faces.append(face_pos)
            geomCode_faces.append(code_faces)

            for face in cad_faces:
                face_indices.append(face_count)
                face_count+=1

                # skip first face
                if len(face) <= 13:
                    continue
        
                edges = face[11:]
                edges_valid = edges[edges >= flag_pad + id_pad].reshape(-1, 8)
                edge_pos = np.stack([x[0:6] - flag_pad - id_pad for x in edges_valid])
                code_edges = np.stack(
                    [
                        x[6:8]
                        - flag_pad
                        - id_pad
                        - pos_pad
                        - self.hparams.surf_codebook_size
                        for x in edges_valid
                    ]
                )

                prev_faces = edges[
                    (edges == MMTokenIndex.DUMMYID.value) | 
                    np.logical_and(edges >= flag_pad, edges < flag_pad + id_pad)
                ]

                # Make sure to skip dummy edges in user input
                for prev_face, pos, code in zip(prev_faces, edge_pos, code_edges):
                    if prev_face == MMTokenIndex.DUMMYID.value:
                        continue 
                    if len(level_faces) > 0:
                        prev_face_global = level_faces[-1][0] + prev_face - flag_pad
                    else:
                        # first level
                        prev_face_global = prev_face - flag_pad
                    cur_face_global = face_indices[-1] 
                    edge_face_pair.append([cur_face_global, prev_face_global])
                    pos_edges.append(pos)
                    geomCode_edges.append(code)

            level_faces.append(face_indices)
    
        geomCode_faces = np.vstack(geomCode_faces)
        geomCode_edges = np.vstack(geomCode_edges)

        pos_faces = dequantize(
            np.vstack(pos_faces),
            n_bits=self.hparams.bit,
            min_range=-1,
            max_range=1,
        )

        pos_edges = dequantize(
            np.vstack(pos_edges),
            n_bits=self.hparams.bit,
            min_range=-1,
            max_range=1,
        )

        face_edge_adj = np.zeros((len(pos_faces), len(pos_edges)), dtype=bool)
        for edge_idx, pair in enumerate(edge_face_pair):
            cur_face, prev_face = pair
            face_edge_adj[cur_face, edge_idx] = True
            face_edge_adj[prev_face, edge_idx] = True
            
        return pos_faces, geomCode_faces, pos_edges, geomCode_edges, face_edge_adj

    @torch.inference_mode()
    def generate(self, prompt, temperature, threshold):
        print(temperature, threshold)
        return self.cad_gpt.ar_decoder.generate(
            prompts=prompt,
            seq_len=self.hparams.max_seq,
            eos_token=MMTokenIndex.EOS.value,
            temperature=temperature,
            filter_logits_fn=partial(top_p, thres=threshold),
            cache_kv=True,
        )


class AutoBrepPCModel(AutoBrepModel):
    """
    Point-cloud conditioned AutoBrep.

    Freezes pretrained AR + FSQ; trains only `pc_encoder` soft prefix tokens
    injected via `prepend_embeds`.
    """

    def __init__(
        self,
        surf_fsq_ckpt: str = None,
        edge_fsq_ckpt: str = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        sync_dist_train: bool = True,
        bit: int = 10,
        max_seq: int = 3000,
        max_face: int = 200,
        inference_mode: bool = False,
        depth: int = 16,
        heads: int = 32,
        dim: int = 2048,
        kv_groups: int = 8,
        surf_codebook_size: int = 1024,
        edge_codebook_size: int = 1024,
        drop_decoder: bool = True,
        ar_ckpt: str = None,
        freeze_backbone: bool = True,
        pc_num_latents: int = 64,
        pc_hidden: int = 256,
        pc_dropout: float = 0.1,
    ) -> None:
        super().__init__(
            surf_fsq_ckpt=surf_fsq_ckpt,
            edge_fsq_ckpt=edge_fsq_ckpt,
            lr=lr,
            weight_decay=weight_decay,
            sync_dist_train=sync_dist_train,
            bit=bit,
            max_seq=max_seq,
            max_face=max_face,
            inference_mode=inference_mode,
            depth=depth,
            heads=heads,
            dim=dim,
            kv_groups=kv_groups,
            surf_codebook_size=surf_codebook_size,
            edge_codebook_size=edge_codebook_size,
            drop_decoder=drop_decoder,
        )

        self.pc_encoder = PointCloudConditionEncoder(
            dim=dim,
            hidden=pc_hidden,
            num_latents=pc_num_latents,
            dropout=pc_dropout,
        )

        if ar_ckpt and not inference_mode:
            self.load_pretrained_ar(ar_ckpt)

        if freeze_backbone:
            self.cad_gpt.requires_grad_(False)
            if hasattr(self, "surf_vae"):
                self.surf_vae.requires_grad_(False)
            if hasattr(self, "edge_vae"):
                self.edge_vae.requires_grad_(False)

        self.trainable_params = [p for p in self.pc_encoder.parameters() if p.requires_grad]

        n_train = sum(p.numel() for p in self.trainable_params)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[AutoBrepPCModel] trainable={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M "
            f"(freeze_backbone={freeze_backbone})"
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.common_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.common_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def load_pretrained_ar(self, ar_ckpt: str) -> None:
        path = Path(ar_ckpt)
        if not path.is_file():
            raise FileNotFoundError(f"ar_ckpt not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        # Drop unexpected PC keys if reloading a PC ckpt; ignore missing pc_encoder on base AR.
        missing, unexpected = self.load_state_dict(state, strict=False)
        miss_pc = [k for k in missing if k.startswith("pc_encoder")]
        miss_other = [k for k in missing if not k.startswith("pc_encoder")]
        print(
            f"[AutoBrepPCModel] loaded {path.name}: "
            f"missing_pc={len(miss_pc)} missing_other={len(miss_other)} unexpected={len(unexpected)}"
        )
        if miss_other:
            print("[AutoBrepPCModel] missing (non-pc) sample:", miss_other[:8])

    def encode_point_cloud(self, point_cloud: torch.Tensor) -> torch.Tensor:
        # Match pc_encoder parameter dtype (sampler may cast the whole module to fp16).
        param_dtype = next(self.pc_encoder.parameters()).dtype
        if point_cloud.dtype != param_dtype:
            point_cloud = point_cloud.to(dtype=param_dtype)
        return self.pc_encoder(point_cloud)

    def common_step(self, batch):
        token, face_ncs, edge_ncs = (
            batch["seq"],
            batch["face_ncs"].to(dtype=torch.bfloat16),
            batch["edge_ncs"].to(dtype=torch.bfloat16),
        )
        point_cloud = batch["point_cloud"].to(dtype=torch.bfloat16)

        with torch.no_grad():
            surf_id, edge_id = self.encode_fsq_code(face_ncs, edge_ncs)

        updated_token, loss_mask = [], []
        for _token_, _surf_id_, _edge_id_ in zip(token, surf_id, edge_id):
            batch_data = self.copy_fsq_code(_token_, _surf_id_, _edge_id_)
            updated_token.append(
                torch.nn.functional.pad(
                    batch_data,
                    (0, self.pad_len - len(batch_data)),
                    value=-1,
                )
            )
            mask = torch.zeros(self.pad_len).bool()
            mask[:4] = True  # ignore BOS + meta block
            loss_mask.append(mask)

        updated_token = torch.stack(updated_token).detach()
        loss_mask = torch.stack(loss_mask).detach()

        prepend = self.encode_point_cloud(point_cloud)
        loss = self.cad_gpt(
            updated_token,
            cond_mask=loss_mask,
            attn_mask=None,
            prepend_embeds=prepend,
        )
        return loss

    @torch.inference_mode()
    def generate(self, prompt, temperature, threshold, point_cloud=None):
        prepend = None
        if point_cloud is not None:
            prepend = self.encode_point_cloud(point_cloud)
        # Prepend condition only on first decode step so KV cache stays valid.
        return generate_with_prepend_kv_cache(
            self.cad_gpt.ar_decoder,
            prompt,
            seq_len=self.hparams.max_seq,
            prepend_embeds=prepend,
            eos_token=MMTokenIndex.EOS.value,
            temperature=temperature,
            filter_logits_fn=partial(top_p, thres=threshold),
        )


class AutoBrepViewModel(AutoBrepModel):
    """
    Multi-view conditioned AutoBrep (ECCV SFT).

    Freezes pretrained AR + FSQ; trains only `view_encoder` soft prefix tokens
    injected via `prepend_embeds`.
    """

    def __init__(
        self,
        surf_fsq_ckpt: str = None,
        edge_fsq_ckpt: str = None,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        sync_dist_train: bool = True,
        bit: int = 10,
        max_seq: int = 3000,
        max_face: int = 200,
        inference_mode: bool = False,
        depth: int = 16,
        heads: int = 32,
        dim: int = 2048,
        kv_groups: int = 8,
        surf_codebook_size: int = 1024,
        edge_codebook_size: int = 1024,
        drop_decoder: bool = True,
        ar_ckpt: str = None,
        freeze_backbone: bool = True,
        view_num_latents: int = 64,
        view_hidden: int = 256,
        view_dropout: float = 0.1,
        view_dropout_max: int = 2,
        num_views: int = 3,
    ) -> None:
        super().__init__(
            surf_fsq_ckpt=surf_fsq_ckpt,
            edge_fsq_ckpt=edge_fsq_ckpt,
            lr=lr,
            weight_decay=weight_decay,
            sync_dist_train=sync_dist_train,
            bit=bit,
            max_seq=max_seq,
            max_face=max_face,
            inference_mode=inference_mode,
            depth=depth,
            heads=heads,
            dim=dim,
            kv_groups=kv_groups,
            surf_codebook_size=surf_codebook_size,
            edge_codebook_size=edge_codebook_size,
            drop_decoder=drop_decoder,
        )

        self.view_encoder = ViewConditionEncoder(
            dim=dim,
            hidden=view_hidden,
            num_latents=view_num_latents,
            num_image_views=num_views if num_views <= 3 else 3,
            dropout=view_dropout,
            view_dropout_max=view_dropout_max,
            pretrained_backbone=True,
        )

        if ar_ckpt and not inference_mode:
            self.load_pretrained_ar(ar_ckpt)

        if freeze_backbone:
            self.cad_gpt.requires_grad_(False)
            if hasattr(self, "surf_vae"):
                self.surf_vae.requires_grad_(False)
            if hasattr(self, "edge_vae"):
                self.edge_vae.requires_grad_(False)

        self.trainable_params = [
            p for p in self.view_encoder.parameters() if p.requires_grad
        ]

        n_train = sum(p.numel() for p in self.trainable_params)
        n_total = sum(p.numel() for p in self.parameters())
        print(
            f"[AutoBrepViewModel] trainable={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M "
            f"(freeze_backbone={freeze_backbone})"
        )

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.common_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.common_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def load_pretrained_ar(self, ar_ckpt: str) -> None:
        path = Path(ar_ckpt)
        if not path.is_file():
            raise FileNotFoundError(f"ar_ckpt not found: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        miss_view = [k for k in missing if k.startswith("view_encoder")]
        miss_other = [k for k in missing if not k.startswith("view_encoder")]
        print(
            f"[AutoBrepViewModel] loaded {path.name}: "
            f"missing_view={len(miss_view)} missing_other={len(miss_other)} "
            f"unexpected={len(unexpected)}"
        )
        if miss_other:
            print("[AutoBrepViewModel] missing (non-view) sample:", miss_other[:8])

    def encode_views(
        self,
        images: torch.Tensor,
        prim_types: torch.Tensor,
        prim_linetypes: torch.Tensor,
        prim_geom: torch.Tensor,
        prim_mask: torch.Tensor,
    ) -> torch.Tensor:
        param_dtype = next(self.view_encoder.parameters()).dtype
        if images.dtype != param_dtype:
            images = images.to(dtype=param_dtype)
        if prim_geom.dtype != param_dtype and prim_geom.is_floating_point():
            prim_geom = prim_geom.to(dtype=param_dtype)
        return self.view_encoder(
            images, prim_types, prim_linetypes, prim_geom, prim_mask
        )

    @torch.inference_mode()
    def predict_complexity_from_condition(
        self,
        images: torch.Tensor,
        prim_types: torch.Tensor,
        prim_linetypes: torch.Tensor,
        prim_geom: torch.Tensor,
        prim_mask: torch.Tensor,
        *,
        allow_uncond: bool = False,
    ) -> "int | list[int]":
        """
        Infer complexity token from view+DXF prepend via one AR step after BOS,BOM.

        Training already places GT face-count complexity after BOM in the discrete
        sequence while prepend_embeds condition the same Transformer, so P(C|cond)
        is learned by CE. At infer we take argmax over {easy,mid,hard} (+ optional
        uncond). Returns int for B=1, else list[int] length B.
        """
        device = next(self.parameters()).device
        prepend = self.encode_views(
            images, prim_types, prim_linetypes, prim_geom, prim_mask
        )
        seed = torch.tensor(
            [[MMTokenIndex.BOS.value, MMTokenIndex.BOM.value]],
            device=device,
            dtype=torch.long,
        )
        if seed.shape[0] != prepend.shape[0]:
            seed = seed.repeat(prepend.shape[0], 1)

        logits, _ = self.cad_gpt.ar_decoder.net(
            seed,
            return_intermediates=True,
            prepend_embeds=prepend,
        )
        next_logits = logits[:, -1]  # (B, vocab)
        allowed = [
            MMTokenIndex.GEN_EASY.value,
            MMTokenIndex.GEN_MID.value,
            MMTokenIndex.GEN_HARD.value,
        ]
        if allow_uncond:
            allowed.append(MMTokenIndex.GEN_UNCOND.value)
        allowed_t = torch.tensor(allowed, device=device, dtype=torch.long)
        pick = next_logits.index_select(-1, allowed_t).argmax(dim=-1)
        ids = allowed_t[pick]
        if ids.numel() == 1:
            return int(ids[0].item())
        return [int(x) for x in ids.tolist()]

    def common_step(self, batch):
        token, face_ncs, edge_ncs = (
            batch["seq"],
            batch["face_ncs"].to(dtype=torch.bfloat16),
            batch["edge_ncs"].to(dtype=torch.bfloat16),
        )
        images = batch["images"].to(dtype=torch.bfloat16)
        prim_types = batch["prim_types"]
        prim_linetypes = batch["prim_linetypes"]
        prim_geom = batch["prim_geom"].to(dtype=torch.bfloat16)
        prim_mask = batch["prim_mask"]

        with torch.no_grad():
            surf_id, edge_id = self.encode_fsq_code(face_ncs, edge_ncs)

        updated_token, loss_mask = [], []
        for _token_, _surf_id_, _edge_id_ in zip(token, surf_id, edge_id):
            batch_data = self.copy_fsq_code(_token_, _surf_id_, _edge_id_)
            updated_token.append(
                torch.nn.functional.pad(
                    batch_data,
                    (0, self.pad_len - len(batch_data)),
                    value=-1,
                )
            )
            mask = torch.zeros(self.pad_len).bool()
            mask[:4] = True  # ignore BOS + meta block
            loss_mask.append(mask)

        updated_token = torch.stack(updated_token).detach()
        loss_mask = torch.stack(loss_mask).detach()

        prepend = self.encode_views(
            images, prim_types, prim_linetypes, prim_geom, prim_mask
        )
        loss = self.cad_gpt(
            updated_token,
            cond_mask=loss_mask,
            attn_mask=None,
            prepend_embeds=prepend,
        )
        return loss

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        temperature,
        threshold,
        images=None,
        prim_types=None,
        prim_linetypes=None,
        prim_geom=None,
        prim_mask=None,
    ):
        prepend = None
        if images is not None:
            prepend = self.encode_views(
                images, prim_types, prim_linetypes, prim_geom, prim_mask
            )
        # Prepend condition only on first decode step so KV cache stays valid.
        return generate_with_prepend_kv_cache(
            self.cad_gpt.ar_decoder,
            prompt,
            seq_len=self.hparams.max_seq,
            prepend_embeds=prepend,
            eos_token=MMTokenIndex.EOS.value,
            temperature=temperature,
            filter_logits_fn=partial(top_p, thres=threshold),
        )