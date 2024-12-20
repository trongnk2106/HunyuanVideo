from datetime import datetime
import time
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.config import parse_args
from pathlib import Path
import os
from hyvideo.utils.file_utils import save_videos_grid
from loguru import logger


class LoadModel:
    def __init__(self, ):
        self.config = parse_args()
        self.config.dit_weight = "/workspace/HunyuanVideo/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
        self.models_root_path = Path(self.config.model_base)
        self.neg_prompt = self.config.neg_prompt
        self.cfg_scale = self.config.cfg_scale
        self.batch_size = self.config.batch_size
        if not self.models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {self.models_root_path}")
        self.save_path = self.config.save_path if self.config.save_path_suffix=="" else f'{self.config.save_path}_{self.config.save_path_suffix}'
        if not os.path.exists(self.config.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(self.models_root_path, args=self.config)
        
    def save_output(self, outputs):
        samples = outputs['samples']
        try:
            if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                for i, sample in enumerate(samples):
                    sample = samples[i].unsqueeze(0)
                    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                    save_path = f"{self.save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
                    save_videos_grid(sample, save_path, fps=24)
                    logger.info(f'Sample save to: {save_path}')
            
            return {
                "status" : "Done",
                "Output_Path": save_path
            }
        except Exception as e:
            raise e 
