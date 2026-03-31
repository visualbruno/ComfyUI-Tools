import subprocess
import os
from pathlib import Path
import folder_paths as folder_paths
import urllib.request
import torch

import yaml
from box import Box
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from typing import List
from math import ceil
import numpy as np
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from ..unirig.src.inference.download import download

from ..unirig.src.data.asset import Asset
from ..unirig.src.data.extract import get_files, extract_with_blender
from ..unirig.src.data.dataset import UniRigDatasetModule, DatasetConfig, ModelInput
from ..unirig.src.data.datapath import Datapath
from ..unirig.src.data.transform import TransformConfig
from ..unirig.src.tokenizer.spec import TokenizerConfig
from ..unirig.src.tokenizer.parse import get_tokenizer
from ..unirig.src.model.parse import get_model
from ..unirig.src.system.parse import get_system, get_writer

from tqdm import tqdm
import time

file_directory = os.path.dirname(os.path.abspath(__file__))
scripts_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)),'scripts')
root_directory = os.path.dirname(os.path.dirname(__file__))
comfy_directory = os.path.dirname(os.path.dirname(root_directory))
unirig_configs_directory = os.path.join(root_directory, 'unirig','configs')

class FakePipeline:
    def __init():
        self.isFake = True

def load(task: str, path: str) -> Box:
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    print(f"\033[92mload {task} config: {path}\033[0m")
    return Box(yaml.safe_load(open(path, 'r')))
 
def run_task(task,seed,input,input_dir,output,output_dir,npz_dir,cls,data_name,blender_exec_path=None): 
    torch.set_float32_matmul_precision('high')
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, required=True)
    # parser.add_argument("--seed", type=int, required=False, default=123,
                        # help="random seed")
    # parser.add_argument("--input", type=nullable_string, required=False, default=None,
                        # help="a single input file or files splited by comma")
    # parser.add_argument("--input_dir", type=nullable_string, required=False, default=None,
                        # help="input directory")
    # parser.add_argument("--output", type=nullable_string, required=False, default=None,
                        # help="filename for a single output")
    # parser.add_argument("--output_dir", type=nullable_string, required=False, default=None,
                        # help="output directory")
    # parser.add_argument("--npz_dir", type=nullable_string, required=False, default='tmp',
                        # help="intermediate npz directory")
    # parser.add_argument("--cls", type=nullable_string, required=False, default=None,
                        # help="class name")
    # parser.add_argument("--data_name", type=nullable_string, required=False, default=None,
                        # help="npz filename from skeleton phase")
    
    L.seed_everything(seed, workers=True)
    
    task = load('task', task)
    mode = task.mode
    assert mode in ['train', 'predict', 'validate']
    
    if input is not None or input_dir is not None:
        assert output_dir is not None or output is not None, 'output or output_dir must be specified'
        assert npz_dir is not None, 'npz_dir must be specified'
        files = get_files(
            data_name=task.components.data_name,
            inputs=input,
            input_dataset_dir=input_dir,
            output_dataset_dir=npz_dir,
            force_override=True,
            warning=False,
        )
        files = [f[1] for f in files]
        if len(files) > 1 and output is not None:
            print("\033[92mwarning: output is specified, but multiple files are detected. Output will be written.\033[0m")
        datapath = Datapath(files=files, cls=cls)
    else:
        datapath = None
    
    data_config = load('data', os.path.join(unirig_configs_directory,'data', task.components.data))
    transform_config = load('transform', os.path.join(unirig_configs_directory,'transform', task.components.transform))
    
    # get tokenizer
    tokenizer_config = task.components.get('tokenizer', None)
    if tokenizer_config is not None:
        tokenizer_config = load('tokenizer', os.path.join(unirig_configs_directory,'tokenizer', task.components.tokenizer))
        tokenizer_config.order_config.skeleton_path.vroid = os.path.join(unirig_configs_directory,'skeleton','vroid.yaml')
        tokenizer_config.order_config.skeleton_path.mixamo = os.path.join(unirig_configs_directory,'skeleton','mixamo.yaml')
        tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)
    
    # get data name
    data_name = task.components.get('data_name', 'raw_data.npz')
    if data_name is not None:
        data_name = data_name
        
    # get train dataset
    train_dataset_config = data_config.get('train_dataset_config', None)
    if train_dataset_config is not None:
        train_dataset_config = DatasetConfig.parse(config=train_dataset_config)
    
    # get train transform
    train_transform_config = transform_config.get('train_transform_config', None)
    if train_transform_config is not None:
        train_transform_config = TransformConfig.parse(config=train_transform_config)
        
    # get predict dataset
    predict_dataset_config = data_config.get('predict_dataset_config', None)
    if predict_dataset_config is not None:
        predict_dataset_config = DatasetConfig.parse(config=predict_dataset_config).split_by_cls()
    
    # get predict transform
    predict_transform_config = transform_config.get('predict_transform_config', None)
    if predict_transform_config is not None:
        predict_transform_config.order_config.skeleton_path.vroid = os.path.join(unirig_configs_directory,'skeleton','vroid.yaml')
        predict_transform_config.order_config.skeleton_path.mixamo = os.path.join(unirig_configs_directory,'skeleton','mixamo.yaml')        
        predict_transform_config = TransformConfig.parse(config=predict_transform_config)
        
    # get validate dataset
    validate_dataset_config = data_config.get('validate_dataset_config', None)
    if validate_dataset_config is not None:
        validate_dataset_config = DatasetConfig.parse(config=validate_dataset_config).split_by_cls()
    
    # get validate transform
    validate_transform_config = transform_config.get('validate_transform_config', None)
    if validate_transform_config is not None:
        validate_transform_config.order_config.skeleton_path.vroid = os.path.join(unirig_configs_directory,'skeleton','vroid.yaml')
        validate_transform_config.order_config.skeleton_path.mixamo = os.path.join(unirig_configs_directory,'skeleton','mixamo.yaml')           
        validate_transform_config = TransformConfig.parse(config=validate_transform_config)
    
    # get model
    model_config = task.components.get('model', None)
    if model_config is not None:
        model_config = load('model', os.path.join(unirig_configs_directory, 'model', model_config))
        if tokenizer_config is not None:
            tokenizer = get_tokenizer(config=tokenizer_config)
        else:
            tokenizer = None
        model = get_model(tokenizer=tokenizer, **model_config)
    else:
        model = None
    
    # set data
    data = UniRigDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=train_dataset_config,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        validate_dataset_config=validate_dataset_config,
        train_transform_config=train_transform_config,
        validate_transform_config=validate_transform_config,
        tokenizer_config=tokenizer_config,
        debug=False,
        data_name=data_name,
        datapath=datapath,
        cls=cls,
    )
    
    # add call backs
    callbacks = []

    ## get checkpoint callback
    checkpoint_config = task.get('checkpoint', None)
    if checkpoint_config is not None:
        checkpoint_config['dirpath'] = os.path.join('experiments', task.experiment_name)
        callbacks.append(ModelCheckpoint(**checkpoint_config))
    
    ## get writer callback
    writer_config = task.get('writer', None)
    if writer_config is not None:
        assert predict_transform_config is not None, 'missing predict_transform_config in transform'
        if output_dir is not None or output is not None:
            if output is not None:
                assert output.endswith('.fbx'), 'output must be .fbx'
            writer_config['npz_dir'] = npz_dir
            writer_config['output_dir'] = output_dir
            writer_config['output_name'] = output
            writer_config['user_mode'] = True
        writer_config['blender_exec_path'] = blender_exec_path
        callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))
    
    # get trainer
    trainer_config = task.get('trainer', {})
    
    # get scheduler
    scheduler_config = task.get('scheduler', None)
    
    optimizer_config = task.get('optimizer', None)
    loss_config = task.get('loss', None)
    
    # get system
    system_config = task.components.get('system', None)
    if system_config is not None:
        system_config = load('system', os.path.join(unirig_configs_directory, 'system', system_config))
        system = get_system(
            **system_config,
            model=model,
            optimizer_config=optimizer_config,
            loss_config=loss_config,
            scheduler_config=scheduler_config,
            steps_per_epoch=1 if train_dataset_config is None else 
            ceil(len(data.train_dataloader()) // trainer_config.devices // trainer_config.num_nodes),
        )
    else:
        system = None
    
    wandb_config = task.get('wandb', None)
    if wandb_config is not None:
        logger = WandbLogger(
            config={
                'task': task,
                'data': data_config,
                'tokenizer': tokenizer_config,
                'train_dataset_config': train_dataset_config,
                'validate_dataset_config': validate_dataset_config,
                'predict_dataset_config': predict_dataset_config,
                'train_transform_config': train_transform_config,
                'validate_transform_config': validate_transform_config,
                'predict_transform_config': predict_transform_config,
                'model_config': model_config,
                'optimizer_config': optimizer_config,
                'system_config': system_config,
                'checkpoint_config': checkpoint_config,
                'writer_config': writer_config,
            },
            log_model=True,
            **wandb_config
        )
        if logger.experiment.id is not None:
            print(f"\033[92mWandbLogger started: {logger.experiment.id}\033[0m")
            # Get the run URL using wandb.run.get_url() which is more reliable
            run_url = logger.experiment.get_url() if hasattr(logger.experiment, 'get_url') else logger.experiment.url
            print(f"\033[92mWandbLogger url: {run_url}\033[0m")
        else:
            print("\033[91mWandbLogger failed to start\033[0m")
    else:
        logger = None

    # set ckpt path
    resume_from_checkpoint = task.get('resume_from_checkpoint', None)
    resume_from_checkpoint = download(resume_from_checkpoint)
    if trainer_config.get('strategy', None) == "fsdp":
        strategy = FSDPStrategy(
            # Enable activation checkpointing on these layers
            auto_wrap_policy={
                torch.nn.MultiheadAttention
            },
            activation_checkpointing_policy={
                torch.nn.Linear,
                torch.nn.MultiheadAttention,
            },
        )
        trainer_config['strategy'] = strategy
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        **trainer_config,
    )
    
    if mode == 'train':
        trainer.fit(system, datamodule=data, ckpt_path=resume_from_checkpoint)
    elif mode == 'predict':
        assert resume_from_checkpoint is not None, 'expect resume_from_checkpoint in task'
        trainer.predict(system, datamodule=data, ckpt_path=resume_from_checkpoint, return_predictions=False)
    elif mode == 'validate':
        trainer.validate(system, datamodule=data, ckpt_path=resume_from_checkpoint)
    else:
        assert 0    

class VisualBrunoToolsUniRigModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {}
        
    RETURN_TYPES = ("UNIRIG_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "process"
    CATEGORY = "VisualBrunoTools/UniRig"
    OUTPUT_NODE = True
    
    def process(self):
        skeleton_url = 'https://huggingface.co/VAST-AI/UniRig/resolve/main/skeleton/articulation-xl_quantization_256/model.ckpt?download=true'
        skin_url = 'https://huggingface.co/VAST-AI/UniRig/resolve/main/skin/articulation-xl/model.ckpt?download=true'
        
        skeleton_file_path = os.path.join(folder_paths.models_dir,'UniRig','skeleton.ckpt')
        skin_file_path = os.path.join(folder_paths.models_dir,'UniRig','skin.ckpt')
        
        print('Checking if checkpoints exist ...')
        if not os.path.exists(skeleton_file_path):
            path = Path(skeleton_file_path)
            folder = path.parent
            folder.mkdir(parents=True, exist_ok=True)            
            
            urllib.request.urlretrieve(skeleton_url, skeleton_file_path)
            
        if not os.path.exists(skin_file_path):
            path = Path(skin_file_path)
            folder = path.parent
            folder.mkdir(parents=True, exist_ok=True)            
            
            urllib.request.urlretrieve(skin_url, skin_file_path)        
        
        fake_pipeline = FakePipeline()
        
        return (fake_pipeline,)

class VisualBrunoToolsUniRigSkeletonPrediction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("UNIRIG_PIPELINE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "mesh_file":("STRING",),
                "fbx_output_path":("STRING",),
                "target_face_count":("INT",{"default":50000,"min":1,"max":10000000,"step":1}),
                "blender_exec_path":("STRING",{"default":"C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe"}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("fbx_output_file", )
    FUNCTION = "process"
    CATEGORY = "VisualBrunoTools/UniRig"
    OUTPUT_NODE = True

    def process(self, pipeline, seed, mesh_file, fbx_output_path, target_face_count,blender_exec_path,):       
        config_file = os.path.join(root_directory, 'unirig','configs','data','quick_inference.yaml')
        force_override="false"
        config = Box(yaml.safe_load(open(config_file, "r")))
        
        output_path = os.path.join(comfy_directory,'output',fbx_output_path)
        
        #1 Extract data from mesh_file
        files = get_files(
            data_name='raw_data.npz',
            inputs=mesh_file,
            input_dataset_dir=None,
            output_dataset_dir=output_path,
            force_override=force_override,
            warning=True,
        )    
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        extract_with_blender(
            blender_exec_path=blender_exec_path,
            output_folder=output_path,
            target_count=target_face_count,
            num_runs=1,
            Id=1,
            time=timestamp,
            files=files,
        )
        
        skeleton_task = os.path.join(root_directory,'unirig','configs','task','quick_inference_skeleton_articulationxl_ar_256.yaml')
        #add_root="false"
        
        temp_dir = os.path.join(comfy_directory, "temp")
        
        run_task(skeleton_task,seed,mesh_file,None,output_path,None,temp_dir,None,None,blender_exec_path=blender_exec_path)
        
        return (output_path,)
     
     
class VisualBrunoToolsUniRigSkinningWeightPrediction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("UNIRIG_PIPELINE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7fffffff}),
                "skeleton_fbx_file":("STRING",),
                "fbx_output_path":("STRING",),
                "target_face_count":("INT",{"default":50000,"min":1,"max":10000000,"step":1}),
                "blender_exec_path":("STRING",{"default":"C:\\Program Files\\Blender Foundation\\Blender 5.0\\blender.exe"}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("fbx_output_file", )
    FUNCTION = "process"
    CATEGORY = "VisualBrunoTools/UniRig"
    OUTPUT_NODE = True

    def process(self, pipeline, seed, skeleton_fbx_file, fbx_output_path, target_face_count,blender_exec_path,):       
        config_file = os.path.join(root_directory, 'unirig','configs','data','quick_inference.yaml')
        force_override="false"
        config = Box(yaml.safe_load(open(config_file, "r")))
        
        output_path = os.path.join(comfy_directory,'output',fbx_output_path)
        
        #1 Extract data from skeleton_fbx_file
        files = get_files(
            data_name='raw_data.npz',
            inputs=skeleton_fbx_file,
            input_dataset_dir=None,
            output_dataset_dir=output_path,
            force_override=force_override,
            warning=True,
        )    
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        extract_with_blender(
            blender_exec_path=blender_exec_path,
            output_folder=output_path,
            target_count=target_face_count,
            num_runs=1,
            Id=1,
            time=timestamp,
            files=files
        )
        
        skin_task = os.path.join(root_directory,'unirig','configs','task','quick_inference_unirig_skin.yaml')
        
        temp_dir = os.path.join(comfy_directory, "temp")
        
        run_task(skin_task,seed,skeleton_fbx_file,None,output_path,None,temp_dir,None,"raw_data.npz",blender_exec_path=blender_exec_path)
        
        return (output_path,)     
