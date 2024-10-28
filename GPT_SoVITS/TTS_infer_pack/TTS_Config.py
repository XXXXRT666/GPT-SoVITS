import os, sys
import yaml
from typing import List
from dataclasses import dataclass, field, asdict
from tools.i18n.i18n import scan_language_list
from tools.my_utils import check_infer_device

@dataclass
class BaseCfg:
    configs_path:str = field(init=False,default=None)
    _config_type:str = field(init=False,default=None)
    _configs_base_path:str = field(init=False,default=None)
    _config_name:str = field(init=False,default=None)
    _exception:list = field(init=False,default_factory=lambda:["_exception","_configs_base_path","_config_name","configs_path","_config_type"])
    
    def __post_init__(self):
        self._configs_base_path = os.path.dirname(self.configs_path) if self.configs_path is not None else self._configs_base_path
        self._config_name = os.path.basename(self.configs_path) if self.configs_path is not None else self._config_name
        os.makedirs(self._configs_base_path, exist_ok=True)
        self.configs_path = os.path.join(self._configs_base_path,self._config_name)
        
        self._load_configs(self.configs_path)
        self.check_config()
        self.update_configs()
        self.save_configs()        
    
    def _load_configs(self, configs_path:str=None)->dict:
        if os.path.exists(configs_path):
            ...
        else:
            with open(configs_path, 'w') as f:
                ...
            print("Using Default Config: " + configs_path)
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        if configs is not None and configs.get(self._config_type):
            for key, value in configs[self._config_type].items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def get_config(self):
        return {self._config_type:{k: v for k, v in self.__dict__.items() if k not in self._exception}}
    
    def save_configs(self, configs_path:str=None)->None:  
        if configs_path is None:
            configs_path = self.configs_path
        if os.path.exists(configs_path):
            ...
        else:
            self._load_configs(self.configs_path)
        with open(configs_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)
        if configs is None:
            configs = dict()
        with open(configs_path, 'w') as file:
            configs.update(self.get_config())
            yaml.dump(configs, file, sort_keys=False)
            
    def update_configs(self):
        raise NotImplementedError
    
    def check_config(self):
        raise NotImplementedError
    
    def __str__(self):
        string = self._config_type.center(60, '-') + '\n'
        for k, v in self.get_config()[self._config_type].items():
            string += f"{str(k).ljust(12)}: {str(v)}\n"
        string += "-" * 60 + '\n'
        return string
    
    def __repr__(self):
        return self.__str__()


@dataclass
class WebUI_Cfg(BaseCfg):
    configs_path:str = field(default=None)
    _config_type:str = field(init=False,default="WebUI_Cfg")
    _config_name:str = field(init=False,default="WebUI_Cfg.yaml")
    
    exp_root:str = field(default='logs')
    python_exec:str = field(default='python')
    webui_port_main:int = field(default=9874)
    webui_port_uvr5:int = field(default=9873)
    webui_port_infer_tts:int = field(default=9872)
    webui_port_subfix:int = field(default=9871)
    language:str = field(default=os.environ.get("language","Auto"))
    is_share:bool = field(default=False)
    
    def update_configs(self):
        self.python_exec = sys.executable or "python"
        
    def check_config(self):
        if self.language not in scan_language_list():
            self.language = 'Auto'
    
    
@dataclass    
class API_Cfg_batch(BaseCfg):
    configs_path:str = field(default=None)
    _config_type:str = field(init=False,default="API_Cfg_batch")
    _config_name:str = field(init=False,default="API_Cfg_batch.yaml")
    _exception:list = field(init=False,default_factory=lambda:["_exception","_configs_base_path","configs_path","_config_name","aux_ref_audio_paths","seed"])
    
    api_addr:str = field(default='::')
    api_port: int = field(default="9880")
    ref_audio_path:str = field(default=None)
    aux_ref_audio_paths: List[str] = field(default_factory=list)
    prompt_text:str = field(default=None)
    prompt_lang:str = field(default=None)
    top_k: int = field(default=5)
    top_p: float = field(default=1.0)
    temperature: float = field(default=1.0)
    text_split_method:str = field(default='cut0')
    batch_size: int = field(default=1)
    speed_factor: int = field(default=1.0)
    stream_mode: bool = field(default=False)
    seed: int = field(default=-1)
    parallel_infer: bool = field(default=True)
    reprtition_penalty: float = field(default=1.35)
    device:str = field(init=False)
    is_half:bool = field(init=False)
    
    def update_configs(self):
        self.device, self.is_half = check_infer_device()
    
    def check_config(self):
        if isinstance(self.api_port,int) and 0<self.api_port<=65536:
            ...
        elif isinstance(self.top_k,int) and 1 <= self.top_k <=100:
            ...
        elif isinstance(self.top_p,int) and 1 <= self.top_p <=100:
            ...
        elif isinstance(self.temperature,int) and 1 <= self.temperature <=100:
            ...
        elif self.text_split_method[-1] in "01234":
            ...
        else:
            raise RuntimeError
    
        
@dataclass        
class API_Cfg(BaseCfg):
    configs_path:str = field(default=None)
    _config_type:str = field(init=False,default="API_Cfg")
    _config_name:str = field(init=False,default="API_Cfg.yaml")
    _exception:list = field(init=False,default_factory=lambda:["_exception","_configs_base_path","_config_name","aux_ref_audio_paths"])
    
    api_addr:str = field(default='::')
    api_port: int = field(default="9880")
    vits_weights_path:str = field(default="GPT_SoVITS/pretrained_models/s2G488k.pth")
    t2s_weights_path:str = field(default="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
    ref_audio_path:str = field(default=None)
    aux_ref_audio_paths: List[str] = field(default_factory=list)  
    prompt_text:str = field(default=None)
    prompt_lang:str = field(default=None)
    cut_punc:str = field(default=None)
    top_k: int = field(default=5)
    top_p: float = field(default=1.0)
    temperature: float = field(default=1.0)
    device:str = field(init=False)
    is_half:str = field(init=False)
    
    def update_configs(self):
        self.device, self.is_half = check_infer_device() 
    
    def check_config(self):
        if isinstance(self.api_port,int) and 0<self.api_port<=65536:
            ...
        elif isinstance(self.top_k,int) and 1 <= self.top_k <=100:
            ...
        elif isinstance(self.top_p,int) and 1 <= self.top_p <=100:
            ...
        elif isinstance(self.temperature,int) and 1 <= self.temperature <=100:
            ...
        else:
            raise RuntimeError   


@dataclass
class TTS_Cfg(BaseCfg):
    configs_path:str = field(default=None)
    _config_type:str = field(init=False,default="TTS_Cfg")
    _config_name:str = field(init=False,default="TTS_Cfg.yaml")
    _exception:str = field(init=False,default_factory=lambda:["_exception","_configs_base_path","_config_name","configs_path","v1_languages","v2_languages","languages","cnhuhbert_default_path","bert_default_path","pretrained_t2s_weights_path_v1","pretrained_t2s_weights_path_v2","pretrained_vits_weights_path_v1","pretrained_vits_weights_path_v2","max_sec","hz","semantic_frame_rate","segment_size","filter_length","sampling_rate","hop_length","win_length","n_speakers"])
    
    v1_languages:list = field(init=False,default_factory=lambda:["auto", "en", "zh", "ja",  "all_zh", "all_ja"])
    v2_languages:list = field(init=False,default_factory=lambda:["auto", "auto_yue", "en", "zh", "ja", "yue", "ko", "all_zh", "all_ja", "all_yue", "all_ko"])
    languages:list = field(init=False,default_factory=list)
    # "all_zh",#全部按中文识别
    # "en",#全部按英文识别#######不变
    # "all_ja",#全部按日文识别
    # "all_yue",#全部按中文识别
    # "all_ko",#全部按韩文识别
    # "zh",#按中英混合识别####不变
    # "ja",#按日英混合识别####不变
    # "yue",#按粤英混合识别####不变
    # "ko",#按韩英混合识别####不变
    # "auto",#多语种启动切分识别语种
    # "auto_yue",#多语种启动切分识别语种
    
    cnhuhbert_base_path:str = field(default="GPT_SoVITS/pretrained_models/chinese-hubert-base")
    bert_base_path:str = field(default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    version:str = field(default="v2")
    
    v1:dict = field(default_factory=lambda:{"t2s_weights_path":"GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt","vits_weights_path":"GPT_SoVITS/pretrained_models/s2G488k.pth"})
    v2:dict = field(default_factory=lambda:{"t2s_weights_path":"GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt","vits_weights_path":"GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"})
    device:str = "cpu"
    is_half:bool = False
    
    cnhuhbert_default_path:str = field(init=False,default="GPT_SoVITS/pretrained_models/chinese-hubert-base")
    bert_default_path:str = field(init=False,default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")
    pretrained_t2s_weights_path_v1:str = field(init=False,default="GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
    pretrained_t2s_weights_path_v2:str = field(init=False,default="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
    pretrained_vits_weights_path_v1:str = field(init=False,default="GPT_SoVITS/pretrained_models/s2G488k.pth")
    pretrained_vits_weights_path_v2:str = field(init=False,default="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth")
    
    max_sec:str = field(init=False,default=None)
    hz:int = field(init=False,default=50)
    semantic_frame_rate:str = field(init=False,default="25hz")
    segment_size:int = field(init=False,default=20480)
    filter_length:int = field(init=False,default=2048)
    sampling_rate:int = field(init=False,default=32000)
    hop_length:int = field(init=False,default=640)
    win_length:int = field(init=False,default=2048)
    n_speakers:int = field(init=False,default=300)
    
    def __post_init__(self):
        super().__post_init__()
        # print(self.configs_path)
        assert self.version in ["v1", "v2"]
        self.languages = self.v2_languages if self.version=="v2" else self.v1_languages
    
    def update_configs(self):
        self.device, self.is_half = check_infer_device()
    
    def check_config(self):
        if not os.path.exists(self.cnhuhbert_base_path):
            self.cnhuhbert_base_path = self.cnhuhbert_default_path
            print(f"fall back to default cnhuhbert_base_path: {self.cnhuhbert_default_path}")
        if not os.path.exists(self.bert_base_path):
            self.bert_base_path = self.bert_default_path
            print(f"fall back to default bert_base_path: {self.bert_default_path}")
        if not os.path.exists(self.get_t2s_model_path(self.version)):
            getattr(self,self.version).update({"t2s_weights_path":self.get_pretrained_t2s_model_path(self.version)})
            print(f"fall back to default t2s_weights_path: {self.get_t2s_model_path(self.version)}")
        if not os.path.exists(self.get_vits_model_path(self.version)):
            getattr(self,self.version).update({"vits_weights_path":self.get_pretrained_vits_model_path(self.version)})
            print(f"fall back to default vits_weights_path: {self.get_vits_model_path(self.version)}")
                
    def get_t2s_model_path(self,version=None):
        if version is None:
            version = self.version
        return getattr(self,version)["t2s_weights_path"]
    
    def get_vits_model_path(self,version=None):
        if version is None:
            version = self.version
        return getattr(self,version)["vits_weights_path"]
    
    def get_pretrained_t2s_model_path(self,version=None):
        if version is None:
            version = self.version
        return getattr(self,f"pretrained_t2s_weights_path_{version}")
    
    def get_pretrained_vits_model_path(self,version=None):
        if version is None:
            version = self.version
        return getattr(self,f"pretrained_vits_weights_path_{version}")


def Cfg(configs_path:str=None,config_type:str=None):
    
    configs_path = configs_path or "GPT_SoVITS/configs/Cfg.yaml"
    cls_map:dict = {
                    "WebUI_Cfg":WebUI_Cfg,
                    "TTS_Cfg":TTS_Cfg,
                    "API_Cfg":API_Cfg,
                    "API_Cfg_batch":API_Cfg_batch,
                    }
    
    if os.path.exists(configs_path):
        ...
    else:
        for cfg in cls_map.values():
            print(cfg)
            cfg(configs_path)
    if config_type is None:
        raise ValueError
    return cls_map[config_type](configs_path)