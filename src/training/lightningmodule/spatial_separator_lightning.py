"""
SpatialSeparatorLightning
=========================
BaseLightningModule를 상속하여 SpatialSeparatorModel (new_models.py) 학습.

배치 계약 (DatasetS3 generate 모드 + fg_return에 metadata 포함):
    batch['mixture']      : [B, 4, T]
    batch['dry_sources']  : [B, K, 1, T]  ← dataset이 1-ch dry source를 줌
    batch['label_vector'] : [B, K, 18]    (silence_label_mode='zeros', label_vector_mode='stack')
    batch['metadata']     : list of dicts, metadata['fg_events'][k]['event_position'] = [[x,y,z]]
                            ※ return_meta=True 일 때만 존재

DoA ground-truth는 fg_events[k]['event_position'][0] 에서 꺼냄.
silence 슬롯(label all-zero)은 DoA target = [0, 0, 0].
"""
import torch
from .base_lightningmodule import BaseLightningModule
class SpatialSeparatorLightning(BaseLightningModule):
    """
    YAML lightning_module 섹션:
        module: src.training.lightningmodule.spatial_separator_lightning
        main: SpatialSeparatorLightning
        args:
            model:
                module: src.models.new_models
                main: SpatialSeparatorModel
                args: {}
            loss:
                module: src.training.loss.final_joint_pit_loss
                main: get_loss_func
                args:
                    w_wav: 1.0
                    w_doa: 0.5
                    w_cls: 0.5
            optimizer:
                module: torch.optim
                main: AdamW
                args:
                    params: null
                    lr: 0.0001
                    weight_decay: 0.01
            is_validation: false
    """

    # ──────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────

    def _build_targets(self, batch: dict) -> dict:
        return {
            'waveforms': batch['waveforms'],   # [B, K, T]
            'labels':    batch['labels'],       # [B, K]  long
            'doas':      batch['doas'],         # [B, K, 3]
            'active':    batch['active'],       # [B, K]  bool
        }

    # ──────────────────────────────────────────────────────────
    # training_step_processing (BaseLightningModule 추상 메서드)
    # ──────────────────────────────────────────────────────────

    def training_step_processing(self, batch: dict, batch_idx: int):
        batchsize = batch['mixture'].shape[0]
        output = self.model(batch['mixture'])   # SpatialSeparatorModel.forward()
        target = self._build_targets(batch)
        loss_dict = self.loss_func(output, target)

        return batchsize, loss_dict

    # ──────────────────────────────────────────────────────────
    # validation_step_processing (is_validation=True 일 때 호출)
    # ──────────────────────────────────────────────────────────

    def validation_step_processing(self, batch: dict, batch_idx: int):
        batchsize = batch['mixture'].shape[0]
        output = self.model(batch['mixture'])
        target = self._build_targets(batch)
        loss_dict = self.loss_func(output, target)
        loss_dict = {k: v.item() for k, v in loss_dict.items()}

        return batchsize, loss_dict
