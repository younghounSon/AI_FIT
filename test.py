import argparse
from omegaconf import OmegaConf, DictConfig
from ultralytics import YOLO
from model_action.arch import MultiHeadAGCN

def main(cfg: DictConfig) -> None:
    # Vision Model Define
    if cfg.vision_type == 'yolo':
        pretrain = cfg.vision_pretrain_path if cfg.vision_pretrain_path else "yolo11s-pose.pt"
        vision_model = YOLO(pretrain)
    else:
        raise NotImplementedError("Others Not Implemented Yet")
    
    # Action Model Define
    if cfg.action_type == 'agcn':
        action_model = MultiHeadAGCN()
        if cfg.action_pretrain_path: action_model.load_pretrain(cfg.action_pretrain_path)
    else:
        raise NotImplementedError("Others Not Implemented Yet")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="usage: test.py --config your_cfg_yaml_file")
    parser.add_argument("--config",type=str,required=True,help="Your yaml file path")
    parser.add_argument("--mode",type=str,required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    main(cfg)
