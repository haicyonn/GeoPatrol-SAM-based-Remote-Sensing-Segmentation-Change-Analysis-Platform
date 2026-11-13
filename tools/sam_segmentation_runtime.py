"""Pinned SAM segmentation inside the existing authenticated model worker."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
RELEASE = Path('/mnt/h/RSAM-Seg-save/scfp_whu500_100_f1_20261112_01')


class SAMRuntime:
    def __init__(self, root, lock):
        self.root, self.lock = Path(root), lock
        self.model = self.metadata = None

    def predict(self, payload):
        import gc
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        from java_backend_bridge import identifier, safe_asset
        from segmentation.predict import load_release
        from segmentation.contract import mask_statistics
        task, attempt = identifier(payload.get('taskId')), identifier(payload.get('attemptId'))
        if payload.get('afterAssetId') is not None:
            raise ValueError('建筑分割只接受一张输入影像')
        with Image.open(safe_asset(self.root, payload.get('beforeAssetId'))) as opened:
            if opened.width*opened.height > 16_777_216:
                raise ValueError('影像像素数量超限')
            image = opened.convert('RGB')
        with self.lock:
            # GPU admission serializes requests; release the resident text model
            # before large SAM activations. The next chat loads Qwen on demand.
            import serve_change_agent_api as legacy
            legacy.MODEL = None
            gc.collect(); torch.cuda.empty_cache()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            try:
                if self.model is None:
                    self.model, self.metadata, _ = load_release(RELEASE, device)
                else:
                    self.model.to(device)
                transform = transforms.Compose([transforms.Resize((1024,1024)), transforms.ToTensor(),
                    transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
                tensor=transform(image).unsqueeze(0).to(device)
                with torch.inference_mode(), torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
                    logits=self.model.infer(tensor)
                logits=F.interpolate(logits.float(),(image.height,image.width),mode='bilinear',align_corners=False)
                mask=(logits[0,0]>0).cpu().numpy().astype(np.uint8)*255
                asset=f'pred-{task}-{attempt}.png'
                target=safe_asset(self.root,asset); temporary=target.with_suffix('.tmp.png')
                Image.fromarray(mask).save(temporary);temporary.replace(target)
                overlay=f'overlay-{task}-{attempt}.png'
                rgb=np.asarray(image).copy();chosen=mask>0
                rgb[chosen]=(rgb[chosen].astype(np.float32)*.55+np.array([255,70,40])*.45).astype(np.uint8)
                Image.fromarray(rgb).save(safe_asset(self.root,overlay))
                stats=mask_statistics(mask)
                return dict(taskId=task,attemptId=attempt,modelId='sam-scfp-whu',modelVersion=self.metadata['model_id'],
                    taskType='building_segmentation',predMaskAssetId=asset,overlayAssetId=overlay,width=image.width,height=image.height,
                    foregroundPixels=stats['foreground_pixels'],foregroundRatio=stats['foreground_ratio'],
                    componentCount=stats['component_count'],preprocessing='pinned-release;resize1024;bf16;threshold0.5')
            finally:
                if self.model is not None:
                    self.model.features=None
                    prompt=getattr(getattr(self.model,'image_encoder',None),'prompt_generator',None)
                    if prompt is not None and hasattr(prompt,'low_feature'):prompt.low_feature=None
                    self.model.to('cpu')
                # Delete activation references before returning GPU memory.
                for name in ('tensor','logits'):
                    if name in locals():
                        if name=='tensor':del tensor
                        else:del logits
                gc.collect();torch.cuda.empty_cache()
