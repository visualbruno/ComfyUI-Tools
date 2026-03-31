from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Union, Tuple, List
import numpy as np
from numpy import ndarray
import os
from random import shuffle
from box import Box
from torch.onnx.symbolic_opset11 import index_copy

from .spec import ConfigSpec

@dataclass
class DatapathConfig(ConfigSpec):
    '''
    Config to handle input data paths.
    '''
    # root
    input_dataset_dir: str
    
    # use proportion data sampling
    use_prob: bool
    
    # cls: [(path_1, p_1), ...]
    data_path: Dict[str, List[Tuple[str, float]]]
    
    # how many files to return when using data sampling
    num_files: Union[int, None]
    
    @classmethod
    def from_args(cls, **kwargs) -> 'DatapathConfig':
        '''
        Make a temporary datapath from user inputs.
        '''
        input = kwargs.get('input', None)
        output = kwargs.get('output', None)
        recursive = kwargs.get('recursive', False)
        
    
    @classmethod
    def parse(cls, config) -> 'DatapathConfig':
        cls.check_keys(config)
        return DatapathConfig(
            input_dataset_dir=config.input_dataset_dir,
            use_prob=config.get('use_prob', True),
            data_path=config.data_path,
            num_files=config.get('num_files', None),
        )
    
    def split_by_cls(self) -> Dict[str, 'DatapathConfig']:
        res: Dict[str, DatapathConfig] = {}
        for cls in self.data_path:
            res[cls] = deepcopy(self)
            res[cls].data_path = {cls: self.data_path[cls]}
        return res

class Datapath():
    def __init__(
        self,
        config: Union[DatapathConfig, None]=None,
        files: Union[List[str], None]=None,
        cls: Union[str, None]=None,
    ):
        if config is not None:
            self.config     = config
            self.file_list  = []
            cls_probs_first = []
            cls_first       = []
            
            self.files_by_class: Dict[str, List[Dict]] = defaultdict(list)
            self.class_positions: Dict[str, List[int]] = defaultdict(list)
            self.cls_probs_second: Dict[str, ndarray] = defaultdict(List)
            
            for cls in self.config.data_path:
                prob = 0.
                probs_second = []
                for (path, p) in self.config.data_path[cls]:
                    prob += p
                    probs_second.append(p)
                    with open(path, 'r') as f:
                        file_items = []
                        missing = 0
                        for l in f.readlines():
                            raw_data_path = os.path.join(self.config.input_dataset_dir, l.strip(), 'raw_data.npz')
                            if not os.path.exists(raw_data_path):
                                missing += 1
                                continue
                            file_items.append({
                                'cls': cls,
                                'path': os.path.join(self.config.input_dataset_dir, l.strip()),
                                'prob': p
                            })
                        assert len(file_items) > 0, f"files in {path} are all missing! root: {self.config.input_dataset_dir}"
                        if missing > 0:
                            print(f"\033[31m{cls}: {missing} missing files\033[0m")
                        self.files_by_class[cls].append(file_items)
                        self.class_positions[cls].append(0)
                        self.file_list.extend(file_items)
                probs_second = np.array(probs_second)
                self.cls_probs_second[cls] = probs_second / probs_second.sum()
                cls_first.append(cls)
                cls_probs_first.append(prob)
            cls_probs_first = np.array(cls_probs_first)
            self.cls_first: List[str] = cls_first
            self.cls_probs_first: Dict[str, List[float]] = cls_probs_first / cls_probs_first.sum()
        elif files is not None:
            if cls is None:
                cls = 'inference'
            self.file_list  = [{'cls': cls, 'path': file} for file in files]
            cls_probs_first = np.array([1.])
            cls_first       = []
            
            self.files_by_class: Dict[str, List[Dict]] = {cls: self.file_list.copy()}
            self.class_positions: Dict[str, List[int]] = {cls: [0]}
            self.cls_probs_second: Dict[str, ndarray] = {cls: np.array([1.])}
            self.config = Box({'use_prob': False})
        else:
            assert(0)
    
    def __len__(self):
        if self.config.use_prob:
            assert self.config.num_files is not None, 'num_files is not specified'
            return self.config.num_files
        return len(self.file_list)
    
    def __getitem__(self, index) -> Tuple[str, str]:
        if self.config.use_prob:
            # first sample a class
            cls = np.random.choice(self.cls_first, p=self.cls_probs_first)
            
            # second sample in this class
            idx = np.random.choice(len(self.files_by_class[cls]), p=self.cls_probs_second[cls])
            
            # get the current position
            pos = self.class_positions[cls][idx]
            files = self.files_by_class[cls][idx]
            
            # get the item andd update position
            item = files[pos]
            self.class_positions[cls][idx] = (pos + 1) % len(files)
            if (pos + 1) % len(files) == 0:
                shuffle(self.files_by_class[cls][idx])
        else:
            item = self.file_list[index]
        return (item['cls'], item['path'])
    
    def get_data(self) -> List[Tuple[str, str]]:
        return [self[i] for i in range(len(self))]