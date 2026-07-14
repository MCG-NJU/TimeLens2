from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial
import os

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
VITA_ROOT = None
LLAVA_V1_7B_MODEL_PTH = "Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. "

video_models = {
    "Video-LLaVA-7B": partial(VideoLLaVA, model_path="LanguageBind/Video-LLaVA-7B"),
    "Video-LLaVA-7B-HF": partial(
        VideoLLaVA_HF, model_path="LanguageBind/Video-LLaVA-7B-hf"
    ),
    "VideoChat2-HD": partial(
        VideoChat2_HD,
        model_path="OpenGVLab/VideoChat2_HD_stage4_Mistral_7B",
        root=VideoChat2_ROOT,
        config_file="./vlmeval/vlm/video_llm/configs/videochat2_hd.json",
    ),
    "Chat-UniVi-7B": partial(Chatunivi, model_path="Chat-UniVi/Chat-UniVi"),
    "Chat-UniVi-7B-v1.5": partial(
        Chatunivi, model_path="Chat-UniVi/Chat-UniVi-7B-v1.5"
    ),
    "LLaMA-VID-7B": partial(
        LLaMAVID, model_path="YanweiLi/llama-vid-7b-full-224-video-fps-1"
    ),
    "Video-ChatGPT": partial(
        VideoChatGPT, model_path="MBZUAI/Video-ChatGPT-7B", dir_root=VideoChatGPT_ROOT
    ),
    "PLLaVA-7B": partial(PLLaVA, model_path="ermu2001/pllava-7b", dir_root=PLLaVA_ROOT),
    "PLLaVA-13B": partial(
        PLLaVA, model_path="ermu2001/pllava-13b", dir_root=PLLaVA_ROOT
    ),
    "PLLaVA-34B": partial(
        PLLaVA, model_path="ermu2001/pllava-34b", dir_root=PLLaVA_ROOT
    ),
}

ungrouped = {
    "AKI": partial(AKI, name="AKI", ckpt_pth="Sony/AKI-4B-phi-3.5-mini"),
    "TransCore_M": partial(TransCoreM, root=TransCore_ROOT),
    "PandaGPT_13B": partial(PandaGPT, name="PandaGPT_13B", root=PandaGPT_ROOT),
    "flamingov2": partial(
        OpenFlamingo,
        name="v2",
        mpt_pth="anas-awadalla/mpt-7b",
        ckpt_pth="openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    ),
    "VisualGLM_6b": partial(VisualGLM, model_path="THUDM/visualglm-6b"),
    "mPLUG-Owl2": partial(mPLUG_Owl2, model_path="MAGAer13/mplug-owl2-llama2-7b"),
    "mPLUG-Owl3": partial(mPLUG_Owl3, model_path="mPLUG/mPLUG-Owl3-7B-240728"),
    "OmniLMM_12B": partial(
        OmniLMM12B, model_path="openbmb/OmniLMM-12B", root=OmniLMM_ROOT
    ),
    "MGM_7B": partial(
        Mini_Gemini, model_path="YanweiLi/MGM-7B-HD", root=Mini_Gemini_ROOT
    ),
    "Bunny-llama3-8B": partial(BunnyLLama3, model_path="BAAI/Bunny-v1_1-Llama-3-8B-V"),
    "VXVERSE": partial(VXVERSE, model_name="XVERSE-V-13B", root=VXVERSE_ROOT),
    "360VL-70B": partial(QH_360VL, model_path="qihoo360/360VL-70B"),
    "Llama-3-MixSenseV1_1": partial(
        LLama3Mixsense, model_path="Zero-Vision/Llama-3-MixSenseV1_1"
    ),
    "Parrot": partial(Parrot, model_path="AIDC-AI/Parrot-7B"),
    "OmChat": partial(OmChat, model_path="omlab/omchat-v2.0-13B-single-beta_hf"),
    "RBDash_72b": partial(
        RBDash, model_path="RBDash-Team/RBDash-v1.5", root=RBDash_ROOT
    ),
    "Pixtral-12B": partial(Pixtral, model_path="mistralai/Pixtral-12B-2409"),
    "Falcon2-VLM-11B": partial(Falcon2VLM, model_path="tiiuae/falcon-11B-vlm"),
}

o1_key = os.environ.get('O1_API_KEY', None)
o1_base = os.environ.get('O1_API_BASE', None)
o1_apis = {
    'o1': partial(
        GPT4V,
        model="o1-2024-12-17",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800, 
        max_tokens=16384,
        verbose=False,

    ),
    'o3': partial(
        GPT4V, 
        model="o3-2025-04-16",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800, 
        max_tokens=16384, 
        verbose=False,
    ),
    'o4-mini': partial(
        GPT4V, 
        model="o4-mini-2025-04-16",
        key=o1_key,
        api_base=o1_base, 
        temperature=0,
        img_detail='high',
        retry=3,
        timeout=1800,
        max_tokens=16384,
        verbose=False,
    ),
}

api_models = {
    # GPT
    "GPT4V": partial(
        GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_HIGH": partial(
        GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409": partial(
        GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409_HIGH": partial(
        GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o": partial(
        GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4o_HIGH": partial(
        GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20240806": partial(
        GPT4V,
        model="gpt-4o-2024-08-06",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20241120": partial(
        GPT4V,
        model="gpt-4o-2024-11-20",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "ChatGPT4o": partial(
        GPT4V,
        model="chatgpt-4o-latest",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_MINI": partial(
        GPT4V,
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4.5": partial(
        GPT4V, 
        model='gpt-4.5-preview-2025-02-27',
        temperature=0, 
        timeout=600,
        img_size=-1, 
        img_detail='high', 
        retry=10, 
        verbose=False,
    ),
    "gpt-4.1-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-mini-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-mini-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-nano-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-nano-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-5-2025-08-07": partial(
        GPT4V,
        model="gpt-5-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    "gpt-5-mini-2025-08-07": partial(
        GPT4V,
        model="gpt-5-mini-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    "gpt-5-nano-2025-08-07": partial(
        GPT4V,
        model="gpt-5-nano-2025-08-07",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    "gpt-5.1-2025-11-13": partial(
        GPT4V,
        model="gpt-5.1-2025-11-13",
        img_detail="high",
        retry=3,
        verbose=False,
        max_tokens=2**14,
        timeout=300,
    ),
    # Gemini
    "GeminiPro1-0": partial(
        Gemini, model="gemini-1.0-pro", temperature=0, retry=10
    ),  # now GeminiPro1-0 is only supported by vertex backend
    "GeminiPro1-5": partial(
        Gemini, model="gemini-1.5-pro", temperature=0, retry=10
    ),
    "GeminiFlash1-5": partial(
        Gemini, model="gemini-1.5-flash", temperature=0, retry=10
    ),
    "GeminiPro1-5-002": partial(
        GPT4V, model="gemini-1.5-pro-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash1-5-002": partial(
        GPT4V, model="gemini-1.5-flash-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash2-0": partial(
        Gemini, model="gemini-2.0-flash", temperature=0, retry=10
    ),
    "GeminiFlashLite2-0": partial(
        Gemini, model="gemini-2.0-flash-lite", temperature=0, retry=10
    ),
    "GeminiFlash2-5": partial(
        Gemini, model="gemini-2.5-flash", temperature=0, retry=10
    ),
    "GeminiPro2-5": partial(
        Gemini, model="gemini-2.5-pro", temperature=0, retry=10
    ),
    
    # Qwen-VL
    "QwenVLPlus": partial(QwenVLAPI, model="qwen-vl-plus", temperature=0, retry=10),
    "QwenVLMax": partial(QwenVLAPI, model="qwen-vl-max", temperature=0, retry=10),
    "QwenVLMax-250408": partial(QwenVLAPI, model="qwen-vl-max-2025-04-08", temperature=0, retry=10),

    # Reka
    "RekaEdge": partial(Reka, model="reka-edge-20240208"),
    "RekaFlash": partial(Reka, model="reka-flash-20240226"),
    "RekaCore": partial(Reka, model="reka-core-20240415"),
    # Step1V
    "Step1V": partial(
        GPT4V,
        model="step-1v-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step1.5V-mini": partial(
        GPT4V,
        model="step-1.5v-mini",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    "Step1o": partial(
        GPT4V,
        model="step-1o-vision-32k",
        api_base="https://api.stepfun.com/v1/chat/completions",
        temperature=0,
        retry=10,
        img_size=-1,
        img_detail="high",
    ),
    # Yi-Vision
    "Yi-Vision": partial(
        GPT4V,
        model="yi-vision",
        api_base="https://api.lingyiwanwu.com/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    # Claude
    "Claude3V_Opus": partial(
        Claude3V, model="claude-3-opus-20240229", temperature=0, retry=10, verbose=False
    ),
    "Claude3V_Sonnet": partial(
        Claude3V,
        model="claude-3-sonnet-20240229",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3V_Haiku": partial(
        Claude3V,
        model="claude-3-haiku-20240307",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-5V_Sonnet": partial(
        Claude3V,
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-5V_Sonnet_20241022": partial(
        Claude3V,
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-7V_Sonnet": partial(
        Claude3V,
        model="claude-3-7-sonnet-20250219",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude4_Opus": partial(
        Claude3V,
        model="claude-4-opus-20250514",
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800
    ),
    "Claude4_Sonnet": partial(
        Claude3V,
        model="claude-4-sonnet-20250514",
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800
    ),
    # GLM4V
    "GLM4V": partial(GLMVisionAPI, model="glm4v-biz-eval", temperature=0, retry=10),
    "GLM4V_PLUS": partial(GLMVisionAPI, model="glm-4v-plus", temperature=0, retry=10),
    "GLM4V_PLUS_20250111": partial(
        GLMVisionAPI, model="glm-4v-plus-0111", temperature=0, retry=10
    ),
    # MiniMax abab
    "abab6.5s": partial(
        GPT4V,
        model="abab6.5s-chat",
        api_base="https://api.minimax.chat/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "abab7-preview": partial(
        GPT4V,
        model="abab7-chat-preview",
        api_base="https://api.minimax.chat/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    # CongRong
    "CongRong-v1.5": partial(CWWrapper, model="cw-congrong-v1.5", temperature=0, retry=10),
    "CongRong-v2.0": partial(CWWrapper, model="cw-congrong-v2.0", temperature=0, retry=10),
    # SenseNova
    "SenseNova-V6-Pro": partial(
        SenseChatVisionAPI, model="SenseNova-V6-Pro", temperature=0, retry=10
    ),
    "SenseNova-V6-Reasoner": partial(
        SenseChatVisionAPI, model="SenseNova-V6-Reasoner", temperature=0, retry=10
    ),
    "SenseNova-V6-5-Pro": partial(
        SenseChatVisionAPI, model="SenseNova-V6-5-Pro", retry=10
    ),
    "HunYuan-Vision": partial(
        HunyuanVision, model="hunyuan-vision", temperature=0, retry=10
    ),
    "HunYuan-Standard-Vision": partial(
        HunyuanVision, model="hunyuan-standard-vision", temperature=0, retry=10
    ),
    "HunYuan-Large-Vision": partial(
        HunyuanVision, model="hunyuan-large-vision", temperature=0, retry=10
    ),
    "BailingMM-Lite-1203": partial(
        bailingMMAPI, model="BailingMM-Lite-1203", temperature=0, retry=10
    ),
    "BailingMM-Pro-0120": partial(
        bailingMMAPI, model="BailingMM-Pro-0120", temperature=0, retry=10
    ),
    # BlueLM-2.5
    "BlueLM-2.5-3B": partial(BlueLM_API, model="BlueLM-2.5-3B", temperature=0, retry=3),
    # JiuTian-VL
    "JTVL": partial(JTVLChatAPI, model="jt-vl-chat", temperature=0, retry=10),
    "JTVL-Mini": partial(JTVLChatAPI_Mini, model="jt-vl-chat-mini", temperature=0, retry=10),
    "JTVL-2B": partial(JTVLChatAPI_2B, model="jt-vl-chat-2b", temperature=0, retry=10),
    "VideoChatOnlineV2": partial(VideoChatOnlineV2API, model="videochatonline_v2", temperature=0, retry=10),
    "Taiyi": partial(TaiyiAPI, model="taiyi", temperature=0, retry=10),
    # TeleMM
    "TeleMM": partial(TeleMMAPI, model="TeleAI/TeleMM", temperature=0, retry=10),
    "Qwen2.5-VL-32B-Instruct-SiliconFlow": partial(
        SiliconFlowAPI, model="Qwen/Qwen2.5-VL-32B-Instruct", temperature=0, retry=10),
    "Qwen3-VL-8B--crop--arm_thinker_prompt--sglang": partial(
        ARM_thinker,
        mode="agent",
        agent_repo_root="/path/to/your/ARM-Thinker",
        model="Qwen/Qwen3-VL-8B-Instruct",
        retry=10,
        timeout=300,
        api_base="http://100.97.158.184:38888/v1/chat/completions",
        key="EMPTY",
        temperature=0.0,
        max_tokens=4096,
        # agent params
        max_round=16,
        max_tool_response_length=4096,
        tool_config_path="/path/to/your/ARM-Thinker/examples/self/multiturn/config/tool_config/image_zoom_in_tool_config.yaml",
        # special for sglang server
        use_role_tool=False,
        system_template_type="CommonSystemTemplate",
        # extra prompt to adapt to the ARM-Thinker prompt template [CommonSystemTemplate]
        extra_pt="\n\n**Important Requirement:**\nThe given image is `original_image`. You must output your reasoning inside `<think>...</think>`. After reasoning, either output the final answer within `<answer>...</answer>` or call a tool within `<tool_call>...</tool_call>`. You may call tools multiple times across turns to assist with judgment or verification, **but only one tool per turn**. If a tool call fails, you can retry or stop and give your final answer. Once no more tool calls are needed, provide your final answer or judgment within `<answer>...</answer>`.",
    ),
    "Qwen3-VL-8B--crop--official_prompt--vllm": partial(
        ARM_thinker,
        mode="agent",
        agent_repo_root="/path/to/your/ARM-Thinker",
        model="Qwen/Qwen3-VL-8B-Instruct",
        retry=10,
        timeout=300,
        api_base="http://100.97.203.103:40001/v1/chat/completions",
        key="EMPTY",
        temperature=0.0,
        max_tokens=4096,
        extra_pt="",
        # agent params
        max_round=16,
        max_tool_response_length=4096,
        system_template_type="Qwen3VLSystemTemplateWithTools",
        tool_config_path="/path/to/your/ARM-Thinker/examples/self/multiturn/config/tool_config/image_zoom_in_tool_qwen3vl_config.yaml",
        use_role_tool=True,
    ),
    # lmdeploy api
    "lmdeploy": partial(
        LMDeployAPI,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "lmdeploy_internvl_78B_MPO": partial(
        LMDeployAPI,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
        timeout=100,
    ),
    "lmdeploy_qvq_72B_preview": partial(
        LMDeployAPI,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=10,
        timeout=300,
    ),
    'Taichu-VLR-3B': partial(
        TaichuVLRAPI, 
        model='taichu_vlr_3b', 
        url="https://platform.wair.ac.cn/maas/v1/chat/completions"
    ),
    'Taichu-VLR-7B': partial(
        TaichuVLRAPI, 
        model='taichu_vlr_7b', 
        url="https://platform.wair.ac.cn/maas/v1/chat/completions"
    ),
    # doubao_vl
    "DoubaoVL": partial(
        DoubaoVL, model="Doubao-1.5-vision-pro", temperature=0, retry=3, verbose=False
    ),
    "Seed1.5-VL": partial(
        DoubaoVL, 
        model="doubao-1-5-thinking-vision-pro-250428", 
        temperature=0,
        retry=3, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6": partial(
        DoubaoVL, 
        model="doubao-seed-1.6-250615", 
        temperature=0,
        retry=3, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6-Flash": partial(
        DoubaoVL, 
        model="doubao-seed-1.6-flash-250615", 
        temperature=0,
        retry=3, 
        verbose=False, 
        max_tokens=16384,
    ),
    "Seed1.6-Thinking": partial(
        DoubaoVL, 
        model="doubao-seed-1.6-thinking-250615", 
        temperature=0,
        retry=3, 
        verbose=False, 
        max_tokens=16384,
    ),
    # Shopee MUG-U
    'MUG-U-7B': partial(
        MUGUAPI, 
        model='MUG-U', 
        temperature=0,  
        retry=10, 
        verbose=False, 
        timeout=300),
    # grok
    "grok-vision-beta": partial(
        GPT4V,
        model="grok-vision-beta",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-2-vision-1212": partial(
        GPT4V,
        model="grok-2-vision",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-4-0709": partial(
        GPT4V,
        model="grok-4-0709",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=1200, 
        max_tokens=16384
    ),
    # kimi
    "moonshot-v1-8k": partial(
        GPT4V,
        model="moonshot-v1-8k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-32k": partial(
        GPT4V,
        model="moonshot-v1-32k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-128k": partial(
        GPT4V,
        model="moonshot-v1-128k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    'ernie4.5-turbo': partial(
        GPT4V,
        model='ernie-4.5-turbo-vl-32k', 
        temperature=0,
        retry=3, 
        max_tokens=12000, 
    ),
    'ernie4.5-a3b': partial(
        GPT4V,
        model='ernie-4.5-vl-28b-a3b', 
        temperature=0,
        retry=3, 
        max_tokens=8000,
    )
}

import copy as cp
api_models['gpt-5'] = cp.deepcopy(api_models['gpt-5-2025-08-07'])
api_models['gpt-5-mini'] = cp.deepcopy(api_models['gpt-5-mini-2025-08-07'])
api_models['gpt-5-nano'] = cp.deepcopy(api_models['gpt-5-nano-2025-08-07'])

emu_series = {
    "emu2_chat": partial(Emu, model_path="BAAI/Emu2-Chat"),
    "emu3_chat": partial(Emu3_chat, model_path="BAAI/Emu3-Chat"),
    "emu3_gen": partial(Emu3_gen, model_path="BAAI/Emu3-Gen"),
}

granite_vision_series = {
    'granite_vision_3.1_2b_preview': partial(GraniteVision3, model_path="ibm-granite/granite-vision-3.1-2b-preview"),
    'granite_vision_3.2_2b': partial(GraniteVision3, model_path="ibm-granite/granite-vision-3.2-2b"),
    'granite_vision_3.3_2b': partial(GraniteVision3, model_path="ibm-granite/granite-vision-3.3-2b"),
}

mmalaya_series = {
    "MMAlaya": partial(MMAlaya, model_path="DataCanvas/MMAlaya"),
    "MMAlaya2": partial(MMAlaya2, model_path="DataCanvas/MMAlaya2"),
}

minicpm_series = {
    "MiniCPM-V": partial(MiniCPM_V, model_path="openbmb/MiniCPM-V"),
    "MiniCPM-V-2": partial(MiniCPM_V, model_path="openbmb/MiniCPM-V-2"),
    "MiniCPM-Llama3-V-2_5": partial(
        MiniCPM_Llama3_V, model_path="openbmb/MiniCPM-Llama3-V-2_5"
    ),
    "MiniCPM-V-2_6": partial(MiniCPM_V_2_6, model_path="openbmb/MiniCPM-V-2_6"),
    "MiniCPM-o-2_6": partial(MiniCPM_o_2_6, model_path="openbmb/MiniCPM-o-2_6"),
    "MiniCPM-V-4": partial(MiniCPM_V_4, model_path="openbmb/MiniCPM-V-4"),
    "MiniCPM-V-4_5": partial(MiniCPM_V_4_5, model_path="openbmb/MiniCPM-V-4_5"),
}

xtuner_series = {
    "llava-internlm2-7b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm2-chat-7b",
        llava_path="xtuner/llava-internlm2-7b",
        visual_select_layer=-2,
        prompt_template="internlm2_chat",
    ),
    "llava-internlm2-20b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm2-chat-20b",
        llava_path="xtuner/llava-internlm2-20b",
        visual_select_layer=-2,
        prompt_template="internlm2_chat",
    ),
    "llava-internlm-7b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm-chat-7b",
        llava_path="xtuner/llava-internlm-7b",
        visual_select_layer=-2,
        prompt_template="internlm_chat",
    ),
    "llava-v1.5-7b-xtuner": partial(
        LLaVA_XTuner,
        llm_path="lmsys/vicuna-7b-v1.5",
        llava_path="xtuner/llava-v1.5-7b-xtuner",
        visual_select_layer=-2,
        prompt_template="vicuna",
    ),
    "llava-v1.5-13b-xtuner": partial(
        LLaVA_XTuner,
        llm_path="lmsys/vicuna-13b-v1.5",
        llava_path="xtuner/llava-v1.5-13b-xtuner",
        visual_select_layer=-2,
        prompt_template="vicuna",
    ),
    "llava-llama-3-8b": partial(
        LLaVA_XTuner,
        llm_path="xtuner/llava-llama-3-8b-v1_1",
        llava_path="xtuner/llava-llama-3-8b-v1_1",
        visual_select_layer=-2,
        prompt_template="llama3_chat",
    ),
}

qwen_series = {
    "qwen_base": partial(QwenVL, model_path="Qwen/Qwen-VL"),
    "qwen_chat": partial(QwenVLChat, model_path="Qwen/Qwen-VL-Chat"),
    "monkey": partial(Monkey, model_path="echo840/Monkey"),
    "monkey-chat": partial(MonkeyChat, model_path="echo840/Monkey-Chat"),
    "minimonkey": partial(MiniMonkey, model_path="mx262/MiniMonkey"),
}

thyme_series = {
    "Thyme-7B": partial(Thyme, model_path="Kwai-Keye/Thyme-RL")
}

llava_series = {
    "llava_v1.5_7b": partial(LLaVA, model_path="liuhaotian/llava-v1.5-7b"),
    "llava_v1.5_13b": partial(LLaVA, model_path="liuhaotian/llava-v1.5-13b"),
    "llava_v1_7b": partial(LLaVA, model_path=LLAVA_V1_7B_MODEL_PTH),
    "sharegpt4v_7b": partial(LLaVA, model_path="Lin-Chen/ShareGPT4V-7B"),
    "sharegpt4v_13b": partial(LLaVA, model_path="Lin-Chen/ShareGPT4V-13B"),
    "llava_next_vicuna_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-7b-hf"
    ),
    "llava_next_vicuna_13b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-13b-hf"
    ),
    "llava_next_mistral_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-mistral-7b-hf"
    ),
    "llava_next_yi_34b": partial(LLaVA_Next, model_path="llava-hf/llava-v1.6-34b-hf"),
    "llava_next_llama3": partial(
        LLaVA_Next, model_path="llava-hf/llama3-llava-next-8b-hf"
    ),
    "llava_next_72b": partial(LLaVA_Next, model_path="llava-hf/llava-next-72b-hf"),
    "llava_next_110b": partial(LLaVA_Next, model_path="llava-hf/llava-next-110b-hf"),
    "llava_next_qwen_32b": partial(
        LLaVA_Next2, model_path="lmms-lab/llava-next-qwen-32b"
    ),
    "llava_next_interleave_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-hf"
    ),
    "llava_next_interleave_7b_dpo": partial(
        LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-dpo-hf"
    ),
    "llava-onevision-qwen2-0.5b-ov-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    ),
    "llava-onevision-qwen2-0.5b-si-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    ),
    "llava-onevision-qwen2-7b-ov-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-ov-hf"
    ),
    "llava-onevision-qwen2-7b-si-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-si-hf"
    ),
    "llava_onevision_qwen2_0.5b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-si"
    ),
    "llava_onevision_qwen2_7b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-si"
    ),
    "llava_onevision_qwen2_72b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-si"
    ),
    "llava_onevision_qwen2_0.5b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-ov"
    ),
    "llava_onevision_qwen2_7b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-ov"
    ),
    "llava_onevision_qwen2_72b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-ov-sft"
    ),
    "Aquila-VL-2B": partial(LLaVA_OneVision, model_path="BAAI/Aquila-VL-2B-llava-qwen"),
    "llava_video_qwen2_7b": partial(
        LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
    ),
    "llava_video_qwen2_72b": partial(
        LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-72B-Qwen2"
    ),
}

varco_vision_series = {
    "varco-vision-hf": partial(
        LLaVA_OneVision_HF, model_path="NCSOFT/VARCO-VISION-14B-HF"
    ),
    "varco-vision-2-1.7b": partial(
        VarcoVision, model_path="NCSOFT/VARCO-VISION-2.0-1.7B"
    ),
    "varco-vision-2-14b": partial(
        VarcoVision, model_path="NCSOFT/VARCO-VISION-2.0-14B"
    ),
}

vita_series = {
    "vita": partial(VITA, model_path="VITA-MLLM/VITA", root=VITA_ROOT),
    "vita_qwen2": partial(VITAQwen2, model_path="VITA-MLLM/VITA-1.5", root=VITA_ROOT),
}

long_vita_series = {
    "Long-VITA-16K": partial(
        LongVITA, model_path="VITA-MLLM/Long-VITA-16K_HF", max_num_frame=128
    ),
    "Long-VITA-128K": partial(
        LongVITA, model_path="VITA-MLLM/Long-VITA-128K_HF", max_num_frame=256
    ),
    "Long-VITA-1M": partial(
        LongVITA, model_path="VITA-MLLM/Long-VITA-1M_HF", max_num_frame=256
    ),
}

interns1_mini = {
    "Intern-S1-mini": partial(
        InternS1Chat, model_path="/mnt/shared-storage-user/mllm/lijinsong/models/Intern-S1-mini/"
    ),
}

internvl = {
    "InternVL-Chat-V1-1": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-1", version="V1.1"
    ),
    "InternVL-Chat-V1-2": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-2", version="V1.2"
    ),
    "InternVL-Chat-V1-2-Plus": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-2-Plus", version="V1.2"
    ),
    "InternVL-Chat-V1-5": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL-Chat-V1-5",
        version="V1.5",
    )
}

mini_internvl = {
    "Mini-InternVL-Chat-2B-V1-5": partial(
        InternVLChat, model_path="OpenGVLab/Mini-InternVL-Chat-2B-V1-5", version="V1.5"
    ),
    "Mini-InternVL-Chat-4B-V1-5": partial(
        InternVLChat, model_path="OpenGVLab/Mini-InternVL-Chat-4B-V1-5", version="V1.5"
    ),
}

internvl2 = {
    "InternVL2-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-1B", version="V2.0"
    ),
    "InternVL2-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-2B", version="V2.0"
    ),
    "InternVL2-4B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-4B", version="V2.0"
    ),
    "InternVL2-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-8B", version="V2.0"
    ),
    "InternVL2-26B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-26B", version="V2.0"
    ),
    "InternVL2-40B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-40B", version="V2.0"
    ),
    "InternVL2-76B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-Llama3-76B", version="V2.0"
    ),
    "InternVL2-8B-MPO": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-8B-MPO", version="V2.0"
    ),
    "InternVL2-8B-MPO-CoT": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
}

internvl2_5 = {
    "InternVL2_5-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-1B", version="V2.0"
    ),
    "InternVL2_5-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-2B", version="V2.0"
    ),
    "QTuneVL1-2B": partial(
        InternVLChat, model_path="hanchaow/QTuneVL1-2B", version="V2.0"
    ),
    "InternVL2_5-4B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-4B", version="V2.0"
    ),
    "InternVL2_5-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0"
    ),
    "InternVL2_5-26B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-26B", version="V2.0"
    ),
    "InternVL2_5-38B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-38B", version="V2.0"
    ),
    "InternVL2_5-78B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-78B", version="V2.0"
    ),
    # InternVL2.5 series with Best-of-N evaluation
    "InternVL2_5-8B-BoN-8": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0",
        best_of_n=8, reward_model_path="OpenGVLab/VisualPRM-8B",
    ),
}

internvl2_5_mpo = {
    "InternVL2_5-1B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-1B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-2B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-2B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-4B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-4B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-26B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-26B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-38B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-38B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-78B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-78B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-GUI": partial(
        InternVLChat,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/models/internvl2_5_8b_internlm2_5_7b_dynamic_res_stage1", 
        version="V2.0", 
        max_new_tokens=512,
        screen_parse=False,
    ),
     "InternVL3-7B-GUI": partial(
        InternVLChat,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/GUI/checkpoints/internvl3_7b_dynamic_res_stage1_56/", 
        version="V2.0", 
        max_new_tokens=512,
        screen_parse=False,
    ),
}

internvl3 = {
    "InternVL3-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-1B", version="V2.0"
    ),
    "InternVL3-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-2B", version="V2.0"
    ),
    "InternVL3-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-8B", version="V2.0",
    ),
    "InternVL3-9B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-9B", version="V2.0"
    ),
    "InternVL3-14B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-14B", version="V2.0"
    ),
    "InternVL3-38B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-38B", version="V2.0"
    ),
    "InternVL3-78B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-78B", version="V2.0"
    ),
}

internvl3_5 = {
    "InternVL3_5-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-1B", version="V2.0"
    ),
    "InternVL3_5-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-2B", version="V2.0"
    ),
    "InternVL3_5-4B": partial(
        InternVLChat,
        model_path="/data/s3/videogpu/zhuyuhan/checkpoints/InternVL3_5-4B/",
        version="V2.0"
    ),
    "InternVL3_5-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-8B", version="V2.0"
    ),
    "InternVL3_5-14B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-14B", version="V2.0"
    ),
    "InternVL3_5-GPT-OSS-20B-A4B-Preview": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview", version="V2.0"
    ),
    "InternVL3_5-30B-A3B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-30B-A3B", version="V2.0"
    ),
    "InternVL3_5-38B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-38B", version="V2.0"
    ),
    "InternVL3_5-241B-A28B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-241B-A28B", version="V2.0"
    ),

    "InternVL3_5-1B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-1B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-2B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-2B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-4B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-4B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-8B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-8B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-14B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-14B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-GPT-OSS-20B-A4B-Preview-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-30B-A3B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-30B-A3B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-38B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-38B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
    "InternVL3_5-241B-A28B-Thinking": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3_5-241B-A28B", use_lmdeploy=True,
        max_new_tokens=2**16, cot_prompt_version="r1", do_sample=True, version="V2.0"
    ),
}

qwen3vl_series = {
    "Qwen3-VL-235B-A22B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-235B-A22B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7, 
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-235B-A22B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-235B-A22B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-30B-A3B-Instruct": partial(
        Qwen3VLChat,
        model_path="/data/s3/videogpu/zhuyuhan/checkpoints/Qwen3-VL-30B-A3B-Instruct/snapshots/4b184fbdab8886057d8d80c09f35bcfc65fe640e",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-30B-A3B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-30B-A3B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-8B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-8B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-4B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-4B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-8B-Instruct": partial(
        Qwen3VLChat,
        model_path="/data/s3/videogpu/zhuyuhan/checkpoints/Qwen3-VL-8B-Instruct/snapshots/cadac78306af287f801b75a5565ede58f323f472",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B-Instruct": partial(
        Qwen3VLChat,
        model_path="/data/s3/videogpu/zhuyuhan/checkpoints/Qwen3-VL-4B-Instruct",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-2B-Instruct": partial(
        Qwen3VLChat,
        model_path="/data/s3/videogpu/zhuyuhan/checkpoints/Qwen3-VL-2B-Instruct",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-32B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-32B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.7, 
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20

    ),
    "Qwen3-VL-2B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-2B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-VL-32B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-VL-32B-Thinking",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=1.0, 
        max_new_tokens=40960,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        top_p=0.95,
        top_k=20
    ),
    "Qwen3-Omni-30B-A3B-Instruct": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
    "Qwen3-Omni-30B-A3B-Thinking": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-Omni-30B-A3B-Thinking",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
    "Qwen3-Omni-30B-A3B-Captioner": partial(
        Qwen3VLChat,
        model_path="Qwen/Qwen3-Omni-30B-A3B-Captioner",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_new_tokens=16384,
    ),
    
}

sail_series = {
    "SAIL-VL-2B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-2B"),
    "SAIL-VL-1.5-2B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d5-2B", use_msac = True),
    "SAIL-VL-1.5-8B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d5-8B", use_msac = True),
    "SAIL-VL-1.6-8B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d6-8B", use_msac = True),
    "SAIL-VL-1.7-Thinking-2B-2507": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d7-Thinking-2B-2507", use_msac = True, use_cot=True, max_new_tokens=4096),
    "SAIL-VL-1.7-Thinking-8B-2507": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL-1d7-Thinking-8B-2507", use_msac = True, use_cot=True, max_new_tokens=4096),
    "SAIL-VL2-2B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL2-2B", use_msac = True),
    "SAIL-VL2-8B": partial(SailVL, model_path="BytedanceDouyinContent/SAIL-VL2-8B", use_msac = True),
}

ristretto_series = {
    "Ristretto-3B": partial(Ristretto, model_path="LiAutoAD/Ristretto-3B"),
}

yivl_series = {
    "Yi_VL_6B": partial(Yi_VL, model_path="01-ai/Yi-VL-6B", root=Yi_ROOT),
    "Yi_VL_34B": partial(Yi_VL, model_path="01-ai/Yi-VL-34B", root=Yi_ROOT),
}

xcomposer_series = {
    "XComposer": partial(XComposer, model_path="internlm/internlm-xcomposer-vl-7b"),
    "sharecaptioner": partial(ShareCaptioner, model_path="Lin-Chen/ShareCaptioner"),
    "XComposer2": partial(XComposer2, model_path="internlm/internlm-xcomposer2-vl-7b"),
    "XComposer2_1.8b": partial(
        XComposer2, model_path="internlm/internlm-xcomposer2-vl-1_8b"
    ),
    "XComposer2_4KHD": partial(
        XComposer2_4KHD, model_path="internlm/internlm-xcomposer2-4khd-7b"
    ),
    "XComposer2d5": partial(
        XComposer2d5, model_path="internlm/internlm-xcomposer2d5-7b"
    ),
}

minigpt4_series = {
    "MiniGPT-4-v2": partial(MiniGPT4, mode="v2", root=MiniGPT4_ROOT),
    "MiniGPT-4-v1-7B": partial(MiniGPT4, mode="v1_7b", root=MiniGPT4_ROOT),
    "MiniGPT-4-v1-13B": partial(MiniGPT4, mode="v1_13b", root=MiniGPT4_ROOT),
}

idefics_series = {
    "idefics_9b_instruct": partial(
        IDEFICS, model_path="HuggingFaceM4/idefics-9b-instruct"
    ),
    "idefics_80b_instruct": partial(
        IDEFICS, model_path="HuggingFaceM4/idefics-80b-instruct"
    ),
    "idefics2_8b": partial(IDEFICS2, model_path="HuggingFaceM4/idefics2-8b"),
    # Idefics3 follows Idefics2 Pattern
    "Idefics3-8B-Llama3": partial(
        IDEFICS2, model_path="HuggingFaceM4/Idefics3-8B-Llama3"
    ),
}

smolvlm_series = {
    "SmolVLM-256M": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-256M-Instruct"),
    "SmolVLM-500M": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-500M-Instruct"),
    "SmolVLM": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-Instruct"),
    "SmolVLM-DPO": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-Instruct-DPO"),
    "SmolVLM-Synthetic": partial(SmolVLM, model_path="HuggingFaceTB/SmolVLM-Synthetic"),
    "SmolVLM2-256M": partial(
        SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    ),
    "SmolVLM2-500M": partial(
        SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    ),
    "SmolVLM2": partial(SmolVLM2, model_path="HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
}

instructblip_series = {
    "instructblip_7b": partial(InstructBLIP, name="instructblip_7b"),
    "instructblip_13b": partial(InstructBLIP, name="instructblip_13b"),
}

deepseekvl_series = {
    "deepseek_vl_7b": partial(DeepSeekVL, model_path="deepseek-ai/deepseek-vl-7b-chat"),
    "deepseek_vl_1.3b": partial(
        DeepSeekVL, model_path="deepseek-ai/deepseek-vl-1.3b-chat"
    ),
}

deepseekvl2_series = {
    "deepseek_vl2_tiny": partial(
        DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2-tiny"
    ),
    "deepseek_vl2_small": partial(
        DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2-small"
    ),
    "deepseek_vl2": partial(DeepSeekVL2, model_path="deepseek-ai/deepseek-vl2"),
}

janus_series = {
    "Janus-1.3B": partial(Janus, model_path="deepseek-ai/Janus-1.3B"),
    "Janus-Pro-1B": partial(Janus, model_path="deepseek-ai/Janus-Pro-1B"),
    "Janus-Pro-7B": partial(Janus, model_path="deepseek-ai/Janus-Pro-7B"),
}

cogvlm_series = {
    "cogvlm-grounding-generalist": partial(
        CogVlm,
        model_path="THUDM/cogvlm-grounding-generalist-hf",
        tokenizer_name="lmsys/vicuna-7b-v1.5",
    ),
    "cogvlm-chat": partial(
        CogVlm, model_path="THUDM/cogvlm-chat-hf", tokenizer_name="lmsys/vicuna-7b-v1.5"
    ),
    "cogvlm2-llama3-chat-19B": partial(
        CogVlm, model_path="THUDM/cogvlm2-llama3-chat-19B"
    ),
    "glm-4v-9b": partial(GLM4v, model_path="THUDM/glm-4v-9b"),
    "GLM4_1VThinking-9b": partial(GLMThinking, model_path="THUDM/GLM-4.1V-9B-Thinking"),
    "GLM4_5V": partial(GLMThinking, model_path="THUDM/GLM-4.5V"),
}

wemm_series = {
    "WeMM": partial(WeMM, model_path="feipengma/WeMM"),
}

cambrian_series = {
    "cambrian_8b": partial(Cambrian, model_path="nyu-visionx/cambrian-8b"),
    "cambrian_13b": partial(Cambrian, model_path="nyu-visionx/cambrian-13b"),
    "cambrian_34b": partial(Cambrian, model_path="nyu-visionx/cambrian-34b"),
}

chameleon_series = {
    "chameleon_7b": partial(Chameleon, model_path="facebook/chameleon-7b"),
    "chameleon_30b": partial(Chameleon, model_path="facebook/chameleon-30b"),
}

vila_series = {
    "VILA1.5-3b": partial(VILA, model_path="Efficient-Large-Model/VILA1.5-3b"),
    "Llama-3-VILA1.5-8b": partial(
        VILA, model_path="Efficient-Large-Model/Llama-3-VILA1.5-8b"
    ),
    "VILA1.5-13b": partial(VILA, model_path="Efficient-Large-Model/VILA1.5-13b"),
    "VILA1.5-40b": partial(VILA, model_path="Efficient-Large-Model/VILA1.5-40b"),
    "NVILA-8B": partial(NVILA, model_path="Efficient-Large-Model/NVILA-8B"),
    "NVILA-15B": partial(NVILA, model_path="Efficient-Large-Model/NVILA-15B"),
}

ovis_series = {
    "Ovis1.5-Llama3-8B": partial(Ovis, model_path="AIDC-AI/Ovis1.5-Llama3-8B"),
    "Ovis1.5-Gemma2-9B": partial(Ovis, model_path="AIDC-AI/Ovis1.5-Gemma2-9B"),
    "Ovis1.6-Gemma2-9B": partial(Ovis1_6, model_path="AIDC-AI/Ovis1.6-Gemma2-9B"),
    "Ovis1.6-Llama3.2-3B": partial(Ovis1_6, model_path="AIDC-AI/Ovis1.6-Llama3.2-3B"),
    "Ovis1.6-Gemma2-27B": partial(
        Ovis1_6_Plus, model_path="AIDC-AI/Ovis1.6-Gemma2-27B"
    ),
    "Ovis2-1B": partial(Ovis2, model_path="AIDC-AI/Ovis2-1B"),
    "Ovis2-2B": partial(Ovis2, model_path="AIDC-AI/Ovis2-2B"),
    "Ovis2-4B": partial(Ovis2, model_path="AIDC-AI/Ovis2-4B"),
    "Ovis2-8B": partial(Ovis2, model_path="AIDC-AI/Ovis2-8B"),
    "Ovis2-16B": partial(Ovis2, model_path="AIDC-AI/Ovis2-16B"),
    "Ovis2-34B": partial(Ovis2, model_path="AIDC-AI/Ovis2-34B"),
    "Ovis-U1-3B": partial(OvisU1, model_path="AIDC-AI/Ovis-U1-3B"),
    "Ovis2.5-2B": partial(Ovis2_5, model_path="AIDC-AI/Ovis2.5-2B"),
    "Ovis2.5-9B": partial(Ovis2_5, model_path="AIDC-AI/Ovis2.5-9B")
}

mantis_series = {
    "Mantis-8B-siglip-llama3": partial(
        Mantis, model_path="TIGER-Lab/Mantis-8B-siglip-llama3"
    ),
    "Mantis-8B-clip-llama3": partial(
        Mantis, model_path="TIGER-Lab/Mantis-8B-clip-llama3"
    ),
    "Mantis-8B-Idefics2": partial(Mantis, model_path="TIGER-Lab/Mantis-8B-Idefics2"),
    "Mantis-8B-Fuyu": partial(Mantis, model_path="TIGER-Lab/Mantis-8B-Fuyu"),
}

phi3_series = {
    "Phi-3-Vision": partial(
        Phi3Vision, model_path="microsoft/Phi-3-vision-128k-instruct"
    ),
    "Phi-3.5-Vision": partial(
        Phi3_5Vision, model_path="microsoft/Phi-3.5-vision-instruct"
    ),
}

phi4_series = {
    'Phi-4-Vision': partial(Phi4Multimodal, model_path='microsoft/Phi-4-multimodal-instruct'),
}

xgen_mm_series = {
    "xgen-mm-phi3-interleave-r-v1.5": partial(
        XGenMM, model_path="Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
    ),
    "xgen-mm-phi3-dpo-r-v1.5": partial(
        XGenMM, model_path="Salesforce/xgen-mm-phi3-mini-instruct-dpo-r-v1.5"
    ),
}

hawkvl_series = {
    "HawkVL-2B": partial(
        HawkVL,
        model_path="xjtupanda/HawkVL-2B",
        min_pixels=4 * 28 * 28,
        max_pixels=6800 * 28 * 28,
        use_custom_prompt=True
    )
}

qwen2vl_series = {
    "Qwen-VL-Max-20250813": partial(
        Qwen2VLAPI,
        model="qwen-vl-max-2025-08-13",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        max_length=8192,
    ),
    "Qwen-VL-Max-0809": partial(
        Qwen2VLAPI,
        model="qwen-vl-max-0809",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen-VL-Plus-0809": partial(
        Qwen2VLAPI,
        model="qwen-vl-plus-0809",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "QVQ-72B-Preview": partial(
        Qwen2VLChat,
        model_path="Qwen/QVQ-72B-Preview",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        system_prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
        max_new_tokens=8192,
        post_process=False,
    ),
    "Qwen2-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int4": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int8": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int4": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int8": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "XinYuan-VL-2B-Instruct": partial(
        Qwen2VLChat,
        model_path="Cylingo/Xinyuan-VL-2B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2.5-VL-3B-Instruct": partial(
        Qwen2VLChat,
        model_path="/data/s3/videogpu/zhuyuhan/checkpoints/Qwen2.5-VL-3B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-3B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
        use_vllm=True,
    ),
    "Qwen2.5-VL-7B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-32B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-32B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "MiMo-VL-7B-SFT": partial(
        Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True
    ),
    "MiMo-VL-7B-RL": partial(
        Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-RL",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
        use_lmdeploy=True
    ),
    "Qwen2.5-VL-72B-Instruct-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-Omni-7B-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
        use_audio_in_video=True, # set use audio in video
    ),
    "Qwen2.5-Omni-7B": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    'VLM-R1': partial(
        VLMR1Chat, 
        model_path='omlab/VLM-R1-Qwen2.5VL-3B-Math-0305', 
        min_pixels=1280*28*28, 
        max_pixels=16384*28*28, 
        use_custom_prompt=False),
    'VLAA-Thinker-Qwen2.5VL-3B': partial(
        VLAAThinkerChat, 
        model_path='UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-3B', 
        min_pixels=1280*28*28, 
        max_pixels=16384*28*28, 
        use_custom_prompt=False,
        post_process=True, # post processing for evaluation
        system_prompt=(''
                    "You are VL-Thinking🤔, a helpful assistant with excellent reasoning ability."
                    " A user asks you a question, and you should try to solve it."
                    " You should first think about the reasoning process in the mind and then provides the user with the answer."
                    " The reasoning process and answer are enclosed within <think> </think> and"
                    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
                    "<answer> answer here </answer>"
                ),
    ),
    'VLAA-Thinker-Qwen2.5VL-7B': partial(
        VLAAThinkerChat, 
        model_path='UCSC-VLAA/VLAA-Thinker-Qwen2.5VL-7B', 
        min_pixels=1280*28*28, 
        max_pixels=16384*28*28, 
        use_custom_prompt=False,
        post_process=True, # post processing for evaluation
        system_prompt=(''
                    "You are VL-Thinking🤔, a helpful assistant with excellent reasoning ability."
                    " A user asks you a question, and you should try to solve it."
                    " You should first think about the reasoning process in the mind and then provides the user with the answer."
                    " The reasoning process and answer are enclosed within <think> </think> and"
                    "<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>"
                    "<answer> answer here </answer>"
                ),
    ),
    'WeThink-Qwen2.5VL-7B': partial(
        WeThinkVL, 
        model_path='yangjie-cv/WeThink-Qwen2.5VL-7B', 
        min_pixels=1280*28*28, 
        max_pixels=16384*28*28, 
        use_custom_prompt=False,
        system_prompt=("You FIRST think about the reasoning process as an internal monologue and then provide the final answer.\nThe reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE enclosed within <answer> </answer> tags."
        ),
    ),
}

slime_series = {
    "Slime-7B": partial(SliME, model_path="yifanzhang114/SliME-vicuna-7B"),
    "Slime-8B": partial(SliME, model_path="yifanzhang114/SliME-Llama3-8B"),
    "Slime-13B": partial(SliME, model_path="yifanzhang114/SliME-vicuna-13B"),
}

eagle_series = {
    "Eagle-X4-8B-Plus": partial(Eagle, model_path="NVEagle/Eagle-X4-8B-Plus"),
    "Eagle-X4-13B-Plus": partial(Eagle, model_path="NVEagle/Eagle-X4-13B-Plus"),
    "Eagle-X5-7B": partial(Eagle, model_path="NVEagle/Eagle-X5-7B"),
    "Eagle-X5-13B": partial(Eagle, model_path="NVEagle/Eagle-X5-13B"),
    "Eagle-X5-13B-Chat": partial(Eagle, model_path="NVEagle/Eagle-X5-13B-Chat"),
    "Eagle-X5-34B-Chat": partial(Eagle, model_path="NVEagle/Eagle-X5-34B-Chat"),
    "Eagle-X5-34B-Plus": partial(Eagle, model_path="NVEagle/Eagle-X5-34B-Plus"),
}

moondream_series = {
    "Moondream1": partial(Moondream1, model_path="vikhyatk/moondream1"),
    "Moondream2": partial(Moondream2, model_path="vikhyatk/moondream2"),
}

llama_series = {
    "Llama-3.2-11B-Vision-Instruct": partial(
        llama_vision, model_path="meta-llama/Llama-3.2-11B-Vision-Instruct"
    ),
    "LLaVA-CoT": partial(llama_vision, model_path="Xkev/Llama-3.2V-11B-cot"),
    "Llama-3.2-90B-Vision-Instruct": partial(
        llama_vision, model_path="meta-llama/Llama-3.2-90B-Vision-Instruct"
    ),
    "Llama-4-Scout-17B-16E-Instruct": partial(
        llama4, model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct", use_vllm=True
    ),
}


molmo2_series = {
    "Molmo2-4B": partial(Molmo2, model_path="/data/s3/videogpu/zhuyuhan/checkpoints/Molmo2-4B"),
}

molmo_series = {
    "molmoE-1B-0924": partial(molmo, model_path="allenai/MolmoE-1B-0924"),
    "molmo-7B-D-0924": partial(molmo, model_path="allenai/Molmo-7B-D-0924"),
    "molmo-7B-O-0924": partial(molmo, model_path="allenai/Molmo-7B-O-0924"),
    "molmo-72B-0924": partial(molmo, model_path="allenai/Molmo-72B-0924"),
}

kosmos_series = {
    "Kosmos2": partial(Kosmos2, model_path="microsoft/kosmos-2-patch14-224")
}

points_series = {
    "POINTS-Yi-1.5-9B-Chat": partial(
        POINTS, model_path="WePOINTS/POINTS-Yi-1-5-9B-Chat"
    ),
    "POINTS-Qwen-2.5-7B-Chat": partial(
        POINTS, model_path="WePOINTS/POINTS-Qwen-2-5-7B-Chat"
    ),
    "POINTSV15-Qwen-2.5-7B-Chat": partial(
        POINTSV15, model_path="WePOINTS/POINTS-1-5-Qwen-2-5-7B-Chat"
    ),
}

nvlm_series = {
    "NVLM": partial(NVLM, model_path="nvidia/NVLM-D-72B"),
}

vintern_series = {
    "Vintern-3B-beta": partial(VinternChat, model_path="5CD-AI/Vintern-3B-beta"),
    "Vintern-1B-v2": partial(VinternChat, model_path="5CD-AI/Vintern-1B-v2"),
}

aria_series = {"Aria": partial(Aria, model_path="rhymes-ai/Aria")}

h2ovl_series = {
    "h2ovl-mississippi-2b": partial(H2OVLChat, model_path="h2oai/h2ovl-mississippi-2b"),
    "h2ovl-mississippi-1b": partial(
        H2OVLChat, model_path="h2oai/h2ovl-mississippi-800m"
    ),
}

valley_series = {
    "valley2": partial(
        Valley2Chat, model_path="bytedance-research/Valley-Eagle-7B"
    ),
    "valley2_dpo": partial(
        Valley2Chat, model_path="bytedance-research/Valley2-DPO"
    ),
    "valley3": partial(
        Valley3Chat, use_gthinker_thinking=True, model_path="bytedance-research/Valley3"
    ),
}

ola_series = {
    "ola": partial(Ola, model_path="THUdyh/Ola-7b"),
}

xvl_series = {
    "X-VL-4B": partial(X_VL_HF, model_path="YannQi/X-VL-4B", temperature=0, retry=10),
}

ross_series = {
    "ross-qwen2-7b": partial(Ross, model_path="HaochenWang/ross-qwen2-7b"),
}

ursa_series = {
    "URSA-8B": partial(UrsaChat, model_path="URSA-MATH/URSA-8B"),
    "URSA-8B-PS-GRPO": partial(UrsaChat, model_path="URSA-MATH/URSA-8B-PS-GRPO")    
}

gemma_series = {
    "paligemma-3b-mix-448": partial(
        PaliGemma, model_path="google/paligemma-3b-mix-448"
    ),
    
    # 3B
    "paligemma2-3b-pt-224":  partial(PaliGemma, model_path="google/paligemma2-3b-pt-224"),
    "paligemma2-3b-pt-448":  partial(PaliGemma, model_path="google/paligemma2-3b-pt-448"),
    "paligemma2-3b-mix-224": partial(PaliGemma, model_path="google/paligemma2-3b-mix-224"),
    "paligemma2-3b-mix-448": partial(PaliGemma, model_path="google/paligemma2-3b-mix-448"),

    # 10B
    "paligemma2-10b-pt-224":  partial(PaliGemma, model_path="google/paligemma2-10b-pt-224"),
    "paligemma2-10b-pt-448":  partial(PaliGemma, model_path="google/paligemma2-10b-pt-448"),
    "paligemma2-10b-mix-224": partial(PaliGemma, model_path="google/paligemma2-10b-mix-224"),
    "paligemma2-10b-mix-448": partial(PaliGemma, model_path="google/paligemma2-10b-mix-448"),

    # 28B
    "paligemma2-28b-pt-224":  partial(PaliGemma, model_path="google/paligemma2-28b-pt-224"),
    "paligemma2-28b-pt-448":  partial(PaliGemma, model_path="google/paligemma2-28b-pt-448"),
    "paligemma2-28b-mix-224": partial(PaliGemma, model_path="google/paligemma2-28b-mix-224"),
    "paligemma2-28b-mix-448": partial(PaliGemma, model_path="google/paligemma2-28b-mix-448"),

    'Gemma3-4B': partial(Gemma3, model_path='google/gemma-3-4b-it'),
    'Gemma3-12B': partial(Gemma3, model_path='google/gemma-3-12b-it'),
    'Gemma3-27B': partial(Gemma3, model_path='google/gemma-3-27b-it')
}

aguvis_series = {
    "aguvis_7b": partial(
        Qwen2VLChatAguvis,
        model_path=os.getenv(
            "EVAL_MODEL",
            "xlangai/Aguvis-7B-720P",
        ),
        min_pixels=256 * 28 * 28,
        max_pixels=46 * 26 * 28 * 28,
        use_custom_prompt=False,
        mode='grounding',
    )
}

kimi_series = {
    'Kimi-VL-A3B-Thinking': partial(KimiVL, model_path='moonshotai/Kimi-VL-A3B-Thinking'),
    'Kimi-VL-A3B-Instruct': partial(KimiVL, model_path='moonshotai/Kimi-VL-A3B-Instruct'),
    'Kimi-VL-A3B-Thinking-2506': partial(KimiVL, model_path='moonshotai/Kimi-VL-A3B-Thinking-2506', temperature=0.8, max_tokens=32768, extract_summary=True)
}

flash_vl = {
    'Flash-VL-2B-Dynamic-ISS': partial(FlashVL, model_path='FlashVL/FlashVL-2B-Dynamic-ISS')
}


oryx_series = {
    'oryx': partial(Oryx, model_path="THUdyh/Oryx-1.5-7B"),
}

# recommend: vllm serve moonshotai/Kimi-VL-A3B-Thinking-2506 
# --served-model-name api-kimi-vl-thinking-2506 --trust-remote-code
# --tensor-parallel-size 2 --max-num-batched-tokens 131072 
# --max-model-len 131072 --limit-mm-per-prompt image=256
kimi_vllm_series = {
    "api-kimi-vl-thinking-2506": partial(
        KimiVLAPI,
        model="api-kimi-vl-thinking-2506",
    ),
    "api-kimi-vl-thinking": partial(
        KimiVLAPI,
        model="api-kimi-vl-thinking",
    ),
    "api-kimi-vl": partial(
        KimiVLAPI,
        model="api-kimi-vl",
        max_new_tokens=2048,
        temperature=0,
    ),
}


treevgr_series = {
    'TreeVGR-7B': partial(
        TreeVGR, 
        model_path='HaochenWang/TreeVGR-7B',
        min_pixels=1280*28*28, max_pixels=16384*28*28,
    ),
}

# QTuneVL series
qtunevl_series = {
    "QTuneVL1_5-2B": partial(
        QTuneVLChat, model_path="hanchaow/QTuneVL1_5-2B", version="V1.5"
    ),

    "QTuneVL1_5-3B": partial(
        QTuneVL,
        model_path="hanchaow/QTuneVL1_5-3B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=True,
        post_process=True
    ),
}

# RbdashMM series via lmdeploy API
rbdashmm_api_series_lmdeploy = {
    "rbdashmm3_DPO_38B_api": partial(
        RBdashMMChat3_API,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=600
    ),
    "rbdashmm3_5_DPO_38B_api": partial(
        RBdashChat3_5_API,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=600
    ),
    "rbdashmm3_5_38B_api": partial(
        RBdashMMChat3_5_38B_API,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=600
    ),
    "rbdashmm3_78B_api": partial(
        RBdashMMChat3_78B_API,
        api_base="http://0.0.0.0:23333/v1/chat/completions",
        temperature=0,
        retry=3,
        timeout=600
    )
}

logics_series = {
    "Logics-Thinking-8B": partial(Logics_Thinking,model_path='Logics-MLLM/Logics-Thinking-8B'),
    "Logics-Thinking-32B": partial(Logics_Thinking,model_path='Logics-MLLM/Logics-Thinking-32B'),
}

insight_v_series = {
    "insightv": partial(InsightV, pretrained_reason="THUdyh/Insight-V-Reason-LLaMA3", pretrained_summary="THUdyh/Insight-V-Summary-LLaMA3"),
}

cosmos_series = {
    'Cosmos-Reason1-7B': partial(Cosmos, model_path='nvidia/Cosmos-Reason1-7B', use_vllm=True),
}

keye_series = {
    "Keye-VL-1.5-8B-auto":partial(KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B"),
    "Keye-VL-1.5-8B-think":partial(KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B", think=True),
    "Keye-VL-1.5-8B-nothink":partial(KeyeChat, model_path="Kwai-Keye/Keye-VL-1_5-8B", no_think=True), 
    "Keye-VL-8B-Preview-think":partial(KeyeChat, model_path="Kwai-Keye/Keye-VL-8B-Preview", think=True), 
}

qianfanvl_series = {
    'Qianfan-VL-3B': partial(Qianfan_VL, model_path='baidu/Qianfan-VL-3B'),
    'Qianfan-VL-8B': partial(Qianfan_VL, model_path='baidu/Qianfan-VL-8B'),
    'Qianfan-VL-70B': partial(Qianfan_VL, model_path='baidu/Qianfan-VL-70B'),
}

lfm2vl_series = {
    "LFM2-VL-450M": partial(LFM2VL, model_path="LiquidAI/LFM2-VL-450M"),
    "LFM2-VL-1.6B": partial(LFM2VL, model_path="LiquidAI/LFM2-VL-1.6B"),
    "LFM2-VL-3B": partial(LFM2VL, model_path="LiquidAI/LFM2-VL-3B"),
}

internvl_groups = [
    internvl, internvl2, internvl2_5, mini_internvl, internvl2_5_mpo, 
    internvl3, internvl3_5
]
internvl_series = {}
for group in internvl_groups:
    internvl_series.update(group)

interns1_groups = [
    interns1_mini
]
interns1_series = {}
for group in interns1_groups:
    interns1_series.update(group)

videochat_o3_series = {
    "VideoChat_o3_7B_sft_600": partial(
        Qwen2VLChat,
        model_path="/data/s3/videogpu/videochat-o3/ckpt/SFT-600",
        use_custom_prompt=False,
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=16384 * 28 * 28,
        use_vllm=False
    ),
    "VideoChat_o3_7B_sft_1466": partial(
        Qwen2VLChat,
        model_path="/data/s3/videogpu/videochat-o3/ckpt/SFT-1466",
        use_custom_prompt=False,
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=16384 * 28 * 28,
        use_vllm=False
    ),
    "VideoChat_o3_7B_general_clue_sft_3913": partial(
        Qwen2VLChat,
        model_path="/data/s3/videogpu/videochat-o3/ckpt/general_clue-SFT-3913",
        use_custom_prompt=False,
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=16384 * 28 * 28,
        use_vllm=False
    ),
}

videochat_xl_series = {
    "TimeLens-8B": partial(
        Qwen3VLChat,
        model_path="/data/s3/videogpu/zhuyuhan/checkpoints/Timelens-8B",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_grounding_sft_v0": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_grounding_sft_v0/20260203203124/hf-75",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_grounding_timelens_qwen_iou_085_newqa": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_grounding_timelens_qwen_iou_085_newqa/20260207013901/hf-30",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_grounding_sft_timelens_qwen_dynamic_iou_t0.95_k5.0_newqa": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_grounding_sft_timelens_qwen_dynamic_iou_t0.95_k5.0_newqa/20260207021338/hf-31",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_grounding_vidi_iou0.6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_grounding_vidi_iou0.6/20260207043048/hf-46",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k/20260212184351/hf-113",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_format2": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_format2/20260315053136/hf-114",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_format2_lr2e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_format2_lr2e-6/20260315222258/hf-114",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_0.3_format2_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_0.3_format2_lr5e-6/20260316010126/hf-34",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_format2_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_format2_lr5e-6/20260315153837/hf-114",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_0.3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_0.3/20260213204334/hf-34",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_0.3_freeze_vit": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_0.3_freeze_vit/20260312163639/hf-34",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_0.3_decay0.1": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_0.3_decay0.1/20260312173747/hf-34",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_grounding_vidi_iou0.6_ctx64k": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_grounding_vidi_iou0.6_ctx64k/20260215154912/hf-45",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_0.3_format2": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_0.3_format2/20260315024004/hf-34",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_format2_lr1e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_format2_lr1e-6/20260316172345/hf-114",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_0.3_format2_lr1e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_0.3_format2_lr1e-6/20260316160051/hf-34",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_timelens_100k_0.3_format2_lr2e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens_100k_0.3_format2_lr2e-6/20260316160041/hf-34",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "TimeLens_4B_grpo_temp0.7_rep1.0_pres1.5_topp0.8_topk20": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/TimeLens-4B/grpo/grpo-20260322-1046_MAXFRAMES-131072_FPS-2_TOTALtokens-65536_MINtokens-1_n4x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_8k_timelens_anno": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_8k_timelens_anno/20260323032437/hf-52",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_8k_timelens_anno_0.5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_8k_timelens_anno_0.5/20260323163303/hf-26",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_8k_qwen3vl_anno": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_8k_qwen3vl_anno/20260323151509/hf-53",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_8k_timelens_anno_0.25": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_8k_timelens_anno_0.25/20260323234043/hf-14",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_8k_timelens_anno_sim_ge0.4_p0.51": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_8k_timelens_anno_sim_ge0.4_p0.51/20260324024736/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_8k_timelens_anno_sim_ge0.3_p0.5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_8k_timelens_anno_sim_ge0.3_p0.5/20260324171141/hf-26",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_8k_qwen3vl_anno_0.5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_8k_qwen3vl_anno_0.5/20260326162117/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.7": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.7/20260331215345/hf-28",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.8": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.8/20260401104231/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9/20260401102051/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.3/20260406004344/hf-26",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.4": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.4/20260406004557/hf-26",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5/20260406004627/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_bs256": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_bs256/20260408061708/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full/20260412043121/hf-78",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_freezeVit": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_freezeVit/20260430193334/hf-78",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_mergeGap0.5-1.0": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_mergeGap0.5-1.0/20260412084722/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_mergeGap0.5-3.0": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_mergeGap0.5-3.0/20260412061217/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_mergeGap0.5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_mergeGap0.5/20260415012017/hf-27",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_mergeGap0.5_dyfps": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_mergeGap0.5_dyfps/20260418005612/hf-28",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_grpo_16k_2fps_768f": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f/grpo-20260419-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260419-0621_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_grpo_16k_2fps_768f_PrecisionWeightIoU_0.8": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.8/grpo-20260419-1547_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step250": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step350": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-350",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step400": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-400",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step450": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-450",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step500": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260428-0609_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step250": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step350": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-350",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step400": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-400",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step450": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-450",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6_step500": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/full_sft_grpo_16k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260420-0237_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k0.3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k0.3/20260421055611/hf-86",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k/20260502000634/hf-104",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v2": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v2/20260507211301/hf-107",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_eventcap0.3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_eventcap0.3/20260421054324/hf-121",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_eventcap": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_eventcap/20260421054320/hf-222",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step250": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-250", 
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step350": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-350",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step400": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-400",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step450": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8/checkpoint-450",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_step500": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_ensemble500": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8_ensemble",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_8k_2fps_384f_PrecisionWeightIoU_0.6_ensemble100-300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_8k_2fps_384f_PrecisionWeightIoU_0.6/grpo-20260424-2119_MAXFRAMES-384_FPS-2_TOTALtokens-8192_MINtokens-1_n2x8_ensemble100-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step250": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step350": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-350",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step400": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-400",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step450": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8/checkpoint-450",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_step500": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_ensemble500": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8_ensemble",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_32k_2fps_768f_PrecisionWeightIoU_0.6_ensemble100-300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_32k_2fps_768f_PrecisionWeightIoU_0.6/grpo-20260424-2124_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n2x8_ensemble100-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6/grpo-20260427-1920_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6/grpo-20260427-1920_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6/grpo-20260427-1920_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6/grpo-20260427-1920_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6_step250": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6/grpo-20260427-1920_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6_step300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6/grpo-20260427-1920_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6_step350": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6/grpo-20260427-1920_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6_ensemble150-350": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_DualAnnoPrecisionWeightIoU_0.6/grpo-20260427-1920_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8_ensemble150-350",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step250": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step350": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-350",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step400": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-400",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step450": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-450",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step500": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_ensemble100-200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260430-1922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8_ensemble100-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260502-1714_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260502-1714_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260502-1714_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260502-1714_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_ensemble100-200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260502-1714_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8_ensemble100-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2/grpo-20260503-0142_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2/grpo-20260503-0142_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2/grpo-20260503-0142_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2/grpo-20260503-0142_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2_ensemble100-200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.2/grpo-20260503-0142_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8_ensemble100-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0138_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0138_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0138_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0138_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_ensemble100-200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0138_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8_ensemble100-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_bs128_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0140_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_bs128_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0140_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_bs128_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0140_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_bs128_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0140_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4_bs128_ensemble100-200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.4/grpo-20260503-0140_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8_ensemble100-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260503-0136_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260503-0136_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260503-0136_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260503-0136_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_ensemble100-200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260503-0136_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8_ensemble100-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8/grpo-20260503-0922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8/grpo-20260503-0922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8/grpo-20260503-0922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8/grpo-20260503-0922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8_ensemble100-200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_16k_2fps_768f_rollout_PrecisionWeightIoU_0.8/grpo-20260503-0922_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n2x8_ensemble100-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260507-1144_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260507-1144_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260507-1144_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260507-1144_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6_ensemble100-200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/sft_lvdb_timelens_full_grpo_lvdb_timelens_16k_2fps_768f_rollout_PrecisionWeightIoU_0.6/grpo-20260507-1144_MAXFRAMES-768_FPS-2_TOTALtokens-16384_MINtokens-1_n4x8_ensemble100-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3/20260509034643/hf-92",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v4": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v4/20260509234057/hf-108",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v5/20260510221703/hf-126",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v6/20260511022054/hf-121",
        use_custom_prompt=False, use_vllm=False, temperature=0.7, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.8, top_k=20
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3_videopoint0.3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3_videopoint0.3/20260609134411/hf-134",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3_seeker": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3_seeker/20260609135330/hf-110",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3_ego4d": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3_ego4d/20260609151340/hf-96",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3_hacs0.5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_full_timelens100k_v3_hacs0.5/20260609163636/hf-101",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3/20260611224055/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_ego4d-mo": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_ego4d-mo/20260613023826/hf-51",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_ego4d-fho": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_ego4d-fho/20260613024249/hf-47",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_refine_half_timelens100k_half_v3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_refine_half_timelens100k_half_v3/20260613104409/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_v3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_v3/20260613185636/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_rextime": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_rextime/20260614045814/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_nextgqa": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_nextgqa/20260614203720/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_tvqa0.5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_tvqa0.5/20260614203806/hf-58",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_half_timelens100k_half_v3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_half_timelens100k_half_v3/20260613185736/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_egotimeqa": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_egotimeqa/20260614220234/hf-62",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_ego4dx2": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_ego4dx2/20260615094712/hf-54",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_filter_easy_v3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_filter_easy/20260615160014/hf-45",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_formatv3_v3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_formatv3_v3/20260616171753/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_halfv2_v3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_halfv2_v3/20260617065920/hf-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_lr5e-6/20260617065920/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_ego4d_tgqa0.3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_ego4d_tgqa0.3/20260617173436/hf-73",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_lr1e-5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_half_timelens100k_half_v3_lr1e-5/20260618133705/hf-46",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_v3_ego4d_nlq_refined": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_v3_ego4d_nlq_refined/20260619005358/hf-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_v3_hacs_refined0.5": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_v3_hacs_refined0.5/20260619005407/hf-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_v3_seeker_refined": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_half_v3_seeker_refined/20260619001927/hf-64",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_halfv2_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_half_timelens100k_halfv2_v3_lr5e-6/20260619141753/hf-48",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6/20260619142738/hf-95",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq/20260620003320/hf-99",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_seeker": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_seeker/20260620003704/hf-118",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_seeker_hacs": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_seeker_hacs/20260620004607/hf-126",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_seekerv2": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_seekerv2/20260620135527/hf-117",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-8B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-8B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq/20260621005312/hf-99",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_baseline_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_baseline/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_baseline_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_baseline/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_baseline_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_baseline/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_baseline_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_baseline/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_baseline_step250": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_baseline/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_baseline_step300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_baseline/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6_step250": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6_step300": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_lvdb_timelens_16k_2fps_512f_rollout_PrecisionWeightIoU_0.6/grpo-20260621-1324_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-300",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_timelens2_16k_2fps_512f_rollout_tiou": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_timelens2_16k_2fps_512f_rollout_tiou/grpo-20260622-0354_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_timelens2_16k_2fps_512f_rollout_tgiou": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_timelens2_16k_2fps_512f_rollout_tgiou/grpo-20260622-0354_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_timelens2_16k_2fps_512f_rollout_tiou_lvdb": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_timelens2_16k_2fps_512f_rollout_tiou_lvdb/grpo-20260622-0633_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_timelens2_16k_2fps_512f_rollout_tiou_lvdb_bs128": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_timelens2_16k_2fps_512f_rollout_tiou_lvdb_bs128/grpo-20260622-0647_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_timelens2_32k_2fps_768f_rollout_tiou_lvdb": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_timelens2_32k_2fps_768f_rollout_tiou_lvdb/grpo-20260623-1439_MAXFRAMES-768_FPS-2_TOTALtokens-32768_MINtokens-1_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_timelens2_64k_2fps_1024f_rollout_tiou_lvdb": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/grpo_timelens2_64k_2fps_1024f_rollout_tiou_lvdb/grpo-20260623-2250_MAXFRAMES-1024_FPS-2_TOTALtokens-65536_MINtokens-1_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "halfsft_grpo_timelens2_16k_2fps_512f_rollout_tiou_lvdb": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/halfsft_grpo_timelens2_16k_2fps_512f_rollout_tiou_lvdb/grpo-20260624-0935_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "halfsft_grpo_timelens2_16k_2fps_512f_rollout_tiou_lvdb_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens/output/Qwen3-VL-4B/halfsft_grpo_timelens2_16k_2fps_512f_rollout_tiou_lvdb/grpo-20260624-0935_MAXFRAMES-512_FPS-2_TOTALtokens-16384_MINtokens-1_n8x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "gpt55_codex_v0": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_gpt55_codex/grpo-20260624-1842_F2_MAXF512_T16384_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_other0.3": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_other0.3/20260625002313/hf-33",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb/grpo-20260625-1900_F2_MAXF512_T16384_n8x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb/grpo-20260625-1900_F2_MAXF512_T16384_n8x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb/grpo-20260625-1900_F2_MAXF512_T16384_n8x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb/grpo-20260625-1900_F2_MAXF512_T16384_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_numgen16_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb/grpo-20260626-0124_F2_MAXF512_T16384_n8x8_numgen16/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_numgen16_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb/grpo-20260626-0124_F2_MAXF512_T16384_n8x8_numgen16/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_numgen16_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb/grpo-20260626-0124_F2_MAXF512_T16384_n8x8_numgen16/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_numgen16_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb/grpo-20260626-0124_F2_MAXF512_T16384_n8x8_numgen16",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_samplev2_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb_samplev2/grpo-20260626-0656_F2_MAXF512_T16384_n8x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_samplev2_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb_samplev2/grpo-20260626-0656_F2_MAXF512_T16384_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_samplev2_22_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb_samplev2_22/grpo-20260626-0658_F2_MAXF512_T16384_n8x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_samplev2_22_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb_samplev2_22/grpo-20260626-0658_F2_MAXF512_T16384_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_cot_step50": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb_cot/grpo-20260626-1918_F2_MAXF512_T16384_n8x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_cot_step100": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb_cot/grpo-20260626-1918_F2_MAXF512_T16384_n8x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_cot_step150": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb_cot/grpo-20260626-1918_F2_MAXF512_T16384_n8x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "other0.3_grpo_lvdb_cot_step200": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_lvdb_cot/grpo-20260626-1918_F2_MAXF512_T16384_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_other0.3_gpt55": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_other0.3_gpt55/20260626091251/hf-33",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_timelens_lvdb_split1": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_timelens_lvdb_split1/grpo-20260627-2327_F2_MAXF512_T16384_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "grpo_timelens_lvdb_split2": partial(
        Qwen3VLChat,
        model_path="/mnt/petrelfs/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/grpo_timelens_lvdb_split2/grpo-20260628-0309_F2_MAXF512_T16384_n8x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl/grpo-20260630-1812_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl/grpo-20260630-1812_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_r2twass1_tiou_parse_penalty_kl_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_r2twass1_tiou_parse_penalty_kl/grpo-20260701-0427_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_r2twass1_tiou_parse_penalty_kl": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_r2twass1_tiou_parse_penalty_kl/grpo-20260701-0427_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260701-2146_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_step100": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260701-2146_F2_MAXF512_T16384_n1x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_step150": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260701-2146_F2_MAXF512_T16384_n1x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_step200": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260701-2146_F2_MAXF512_T16384_n1x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_step250": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260701-2146_F2_MAXF512_T16384_n1x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260701-2146_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_parse_penalty_kl_rollhard_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_parse_penalty_kl_rollhard/grpo-20260702-1111_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_parse_penalty_kl_rollhard_step100": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_parse_penalty_kl_rollhard/grpo-20260702-1111_F2_MAXF512_T16384_n1x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_parse_penalty_kl_rollhard_step150": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_parse_penalty_kl_rollhard/grpo-20260702-1111_F2_MAXF512_T16384_n1x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_parse_penalty_kl_rollhard_step200": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_parse_penalty_kl_rollhard/grpo-20260702-1111_F2_MAXF512_T16384_n1x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_parse_penalty_kl_rollhard_step250": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_parse_penalty_kl_rollhard/grpo-20260702-1111_F2_MAXF512_T16384_n1x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_parse_penalty_kl_rollhard": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_parse_penalty_kl_rollhard/grpo-20260702-1111_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_len_ratio_penalty_kl_rollhard_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_len_ratio_penalty_kl_rollhard/grpo-20260703-0438_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_len_ratio_penalty_kl_rollhard_step100": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_len_ratio_penalty_kl_rollhard/grpo-20260703-0438_F2_MAXF512_T16384_n1x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_len_ratio_penalty_kl_rollhard_step150": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_len_ratio_penalty_kl_rollhard/grpo-20260703-0438_F2_MAXF512_T16384_n1x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_len_ratio_penalty_kl_rollhard": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_twass1_parse_penalty_len_ratio_penalty_kl_rollhard/grpo-20260703-0438_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tdirac1_parse_penalty_kl_rollhard_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tdirac1_parse_penalty_kl_rollhard/grpo-20260703-1656_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tdirac1_parse_penalty_kl_rollhard_step100": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tdirac1_parse_penalty_kl_rollhard/grpo-20260703-1656_F2_MAXF512_T16384_n1x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tdirac1_parse_penalty_kl_rollhard_step150": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tdirac1_parse_penalty_kl_rollhard/grpo-20260703-1656_F2_MAXF512_T16384_n1x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tdirac1_parse_penalty_kl_rollhard": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tdirac1_parse_penalty_kl_rollhard/grpo-20260703-1656_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tcgauss1a0.5_parse_penalty_kl_rollhard_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tcgauss1a0.5_parse_penalty_kl_rollhard/grpo-20260703-1702_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tcgauss1a0.5_parse_penalty_kl_rollhard_step100": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tcgauss1a0.5_parse_penalty_kl_rollhard/grpo-20260703-1702_F2_MAXF512_T16384_n1x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tcgauss1a0.5_parse_penalty_kl_rollhard_step150": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tcgauss1a0.5_parse_penalty_kl_rollhard/grpo-20260703-1702_F2_MAXF512_T16384_n1x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tcgauss1a0.5_parse_penalty_kl_rollhard": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tcgauss1a0.5_parse_penalty_kl_rollhard/grpo-20260703-1702_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tbgauss1s0.08_parse_penalty_kl_rollhard_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tbgauss1s0.08_parse_penalty_kl_rollhard/grpo-20260703-1702_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tbgauss1s0.08_parse_penalty_kl_rollhard_step100": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tbgauss1s0.08_parse_penalty_kl_rollhard/grpo-20260703-1702_F2_MAXF512_T16384_n1x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tbgauss1s0.08_parse_penalty_kl_rollhard_step150": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tbgauss1s0.08_parse_penalty_kl_rollhard/grpo-20260703-1702_F2_MAXF512_T16384_n1x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_tbgauss1s0.08_parse_penalty_kl_rollhard": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_tbgauss1s0.08_parse_penalty_kl_rollhard/grpo-20260703-1702_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_8b_step50": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-8b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260705-1000_F2_MAXF512_T16384_n1x8/checkpoint-50",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_8b_step100": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-8b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260705-1000_F2_MAXF512_T16384_n1x8/checkpoint-100",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_8b_step150": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-8b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260705-1000_F2_MAXF512_T16384_n1x8/checkpoint-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_8b_step200": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-8b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260705-1000_F2_MAXF512_T16384_n1x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_8b_step250": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-8b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260705-1000_F2_MAXF512_T16384_n1x8/checkpoint-250",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard_8b": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-8b/train_timelens_lvdb_tiou_twass1_parse_penalty_kl_rollhard/grpo-20260705-1000_F2_MAXF512_T16384_n1x8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb5pct_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb5pct_v3_lr5e-6/20260707172115/hf-8",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb20pct_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb20pct_v3_lr5e-6/20260707173841/hf-29",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb50pct_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb50pct_v3_lr5e-6/20260708021844/hf-37",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb100pct_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb100pct_v3_lr5e-6/20260707183909/hf-72",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_timelens100k_100pct_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_timelens100k_100pct_v3_lr5e-6/20260708030058/hf-24",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb50pct_v3_lr5e-6_gbs128": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb50pct_v3_lr5e-6_gbs128/20260708152817/hf-73",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_ctx16k": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_ctx16k/20260709021301/hf-133",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_ctx32k": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_ctx32k/20260709021303/hf-150",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_ctx64k": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_timelens100k_v3_lr5e-6_ego4dnlq_ctx64k/20260709021308/hf-121",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_timelens_8b_anno_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_timelens_8b_anno_v3_lr5e-6/20260709215117/hf-106",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen3vl_30bA3b_anno_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen3vl_30bA3b_anno_v3_lr5e-6/20260709215124/hf-91",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_v3_lr5e-6/20260711133003/hf-81",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
     "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_v3_lr5e-6/20260711133003/hf-72",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb100pct_ablation_query1_answer3_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb100pct_ablation_query1_answer3_v3_lr5e-6/20260712062050/hf-72",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb100pct_ablation_query1_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb100pct_ablation_query1_v3_lr5e-6/20260712001148/hf-72",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb100pct_ablation_answer3_v3_lr5e-6": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/xtuner_longvideo/work_dir/Qwen3-VL-4B_lvdb_35k_qwen_anno_drop0_crossIoUGe0.9_embedSimGe0.5_guardedRefine_merge1s_lvdb100pct_ablation_answer3_v3_lr5e-6/20260712105052/hf-72",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
    "train_timelens_lvdb_tiou_ngiou_parse_penalty_kl_rollhard_step200": partial(
        Qwen3VLChat,
        model_path="/data/videop1-shared/zhuyuhan/workspace/TimeLens2/output/qwen3-vl-4b/train_timelens_lvdb_tiou_ngiou_parse_penalty_kl_rollhard/grpo-20260714-0149_F2_MAXF512_T16384_n2x8/checkpoint-200",
        use_custom_prompt=False, use_vllm=False, temperature=0.01, max_new_tokens=4096,
        repetition_penalty=1.0, presence_penalty=1.5, top_p=0.001, top_k=1
    ),
}

videochat3_series = {
    "VideoChat3_4B_train_stage2_llava_video_academic": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic/20251203152844/hf-245",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_new": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_new/20251206232852/hf-245",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_new_cc3m": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_new_cc3m/20251210004257/hf-245",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_new_caprl2mrecap": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_new_caprl2mrecap/20251212232552/hf-245",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216/20251216180246/hf-258",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_t1": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_t1/20251221144855/hf-232",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_t1_fps1": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_t1_fps1/20251221012714/hf-609",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_t1": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_t1/20251221225003/hf-897",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_caprl2mrecap2": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_caprl2mrecap2/20251223004602/hf-245/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp/20251221232401/hf-909",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_mixcap": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_mixcap/20251226010459/hf-354",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_smit": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_smit/20251224020852/hf-310",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_vript": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_vript/20251225014437/hf-360",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video/20251225003059/hf-1443",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl2": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl2/20251228211244/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl2",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_cc3m_v2": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_cc3m_v2/20251229031009/hf-909",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl2_vtlr": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl2_vtlr/20251230022622/hf-909",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_no_stage1-2": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_no_stage1-2/20260103015002/hf-909/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl_vtlr_2": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_llava_video_academic_shortcotqa20251216_bee_image_temp_caprl_vtlr_2/20260103014008/hf-909",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v1_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v1_lr8e-5_vtlr2e-5/20260113001002/hf-324",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_beedata_minisft_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_beedata_minisft_lr8e-5_vtlr2e-5/20260112231757/hf-198",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_bee_image_minisft_lr5e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_bee_image_minisft_lr5e-5_vtlr2e-5/20260112233326/hf-198",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v2_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v2_lr8e-5_vtlr2e-5/20260113115745/hf-318",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v1_image_video_minisft_v1_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v1_image_video_minisft_v1_lr8e-5_vtlr2e-5/20260115151743/hf-324",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v1_image_video_minisft_v3_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v1_image_video_minisft_v3_lr8e-5_vtlr2e-5/20260116010903/hf-343/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v3_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v3_lr8e-5_vtlr2e-5/20260115215050/hf-343",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v4_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v4_lr8e-5_vtlr2e-5/20260116175518/hf-534",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_debugbase2_bee_image_minisft_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_debugbase2_bee_image_minisft_lr8e-5_vtlr2e-5/20260117150041/hf-198",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_debugbase1_bee_image_minisft_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_debugbase1_bee_image_minisft_lr8e-5_vtlr2e-5/20260117145748/hf-198",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_debugbase3_bee_image_minisft_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_debugbase3_bee_image_minisft_lr8e-5_vtlr2e-5/20260118003313/hf-198",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v5_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v5_lr8e-5_vtlr2e-5/20260118014953/hf-515",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v6_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v6_lr8e-5_vtlr2e-5/20260118161523/hf-489",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_debugbase4x2_beedata_minisft_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_debugbase4x2_beedata_minisft_lr8e-5_vtlr2e-5/20260119210436/hf-198",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v7_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v7_lr8e-5_vtlr2e-5/20260121131143/hf-1435",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v8_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v8_lr8e-5_vtlr2e-5/20260125163147/hf-500",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v9_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v9_lr8e-5_vtlr2e-5/20260127014326/hf-531",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_debugbase6_bee_image_minisft_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_debugbase6_bee_image_minisft_lr8e-5_vtlr2e-5/20260128005951/hf-198",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v10_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v10_lr8e-5_vtlr2e-5/20260127015141/hf-1539",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v10new_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v10new_lr8e-5_vtlr2e-5/20260129004042/hf-1539",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v11_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v11_lr8e-5_vtlr2e-5/20260128034915/hf-2942/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v13_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v13_lr8e-5_vtlr2e-5/20260201005226/hf-2749/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v13_lr5e-5_vtlr1e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v13_lr5e-5_vtlr1e-5/20260203002805/hf-2749",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_k25_minisft_v10_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_k25_minisft_v10_lr8e-5_vtlr2e-5/20260204002857/hf-1539/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v16_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/pnorm2/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v16_lr8e-5_vtlr2e-5/20260207010129/hf-1632/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v14_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v14_lr8e-5_vtlr2e-5/20260204142900/hf-6384",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_a1_lr2e-5_vtlr5e-6": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_a1_lr2e-5_vtlr5e-6/20260208125300/hf-480",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5/20260207001816/hf-2008",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v15_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v15_lr8e-5_vtlr2e-5/20260209001238/hf-1740",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_a2_lr2e-5_vtlr5e-6": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_a2_lr2e-5_vtlr5e-6/20260210142200/hf-533",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v17_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v17_lr8e-5_vtlr2e-5/20260210233805/hf-1729",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_8k_pt_minisft_image_ab0d2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_8k_pt_minisft_image_ab0d2/20260212140246/hf-1251",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v18_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v18_lr8e-5_vtlr2e-5/20260212024649/hf-3190",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_32k_pt_minisft_image_ab0d2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_32k_pt_minisft_image_ab0d2",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage1-2_sft": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage1-2_sft/20260217012450/hf-1241",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v20_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v20_lr8e-5_vtlr2e-5/20260217012302/hf-1994",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage1-2_sft2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage1-2_sft2/20260217153404/hf-1151",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage1-2_sft4": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage1-2_sft4/20260218134852/hf-1151",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage1-2_sft3": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage1-2_sft3/20260218213251/hf-1151",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage1-2_sohav2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage1-2_sohav2/20260217041339/hf-1461",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v19_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v19_lr8e-5_vtlr2e-5/20260216005332/hf-2106",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5_sohav2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5_sohav2/20260219230239/hf-2008",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5_ptl": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5_ptl/20260220002833/hf-2008",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage1-2_sft5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage1-2_sft5/20260222143606/hf-1151",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_pt4_minisft_v15_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_pt4_minisft_v15_lr8e-5_vtlr2e-5/20260222181114/hf-1740",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v15_lr2e-4_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v15_lr2e-4_vtlr2e-5/20260222000157/hf-2008",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage1-2_sft6": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage1-2_sft6/20260225012109/hf-1151",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5_2epoch": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5_2epoch/20260223002314/hf-4016",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5_pt5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v15_lr8e-5_vtlr2e-5_pt5/20260226015854/hf-2008",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v21_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v21_lr8e-5_vtlr2e-5/20260226012545/hf-5382",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v23_lr8e-5_vtlr2e-5_sohav2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v23_lr8e-5_vtlr2e-5_sohav2/20260228234310/hf-2209",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v22_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v22_lr8e-5_vtlr2e-5/20260303002143/hf-2243",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v24_lr8e-5_vtlr2e-5_sohav2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v24_lr8e-5_vtlr2e-5_sohav2/20260306014821/hf-2306",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_v2_minisft_v212_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_v2_minisft_v212_lr8e-5_vtlr2e-5/20260304231318/hf-4791",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_v23_lr8e-5_vtlr2e-5_sohav2_v21_long": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_v23_lr8e-5_vtlr2e-5_sohav2_v21_long/20260310174645/hf-2384",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_final_minisft_v15_lr8e-5_vtlr2e-5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_final_minisft_v15_lr8e-5_vtlr2e-5/20260315025040/hf-1740",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v25_lr8e-5_vtlr2e-5_sohav2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v25_lr8e-5_vtlr2e-5_sohav2/20260307023820/hf-2755",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_8k_pt-image_minisft_image_ab0d2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_8k_pt-image_minisft_image_ab0d2/20260215002648/hf-1251",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_minisft_0d2_lr8e-5_vtlr2e-5_oryx": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_minisft_0d2_lr8e-5_vtlr2e-5_oryx/20260319064811/hf-1106/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_minisft_0d2_lr8e-5_vtlr2e-5_sohav2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_minisft_0d2_lr8e-5_vtlr2e-5_sohav2/20260318000915/hf-1305",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_v27_lr8e-5_vtlr2e-5_sohav2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_v27_lr8e-5_vtlr2e-5_sohav2/20260314224859/hf-2559",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_v21base_lr8e-5_vtlr2e-5_v21_long": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_v21base_lr8e-5_vtlr2e-5_v21_long/20260317004502/hf-2384",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_v21base_lr8e-5_vtlr2e-5_long_v2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_v21base_lr8e-5_vtlr2e-5_long_v2/20260321003308/hf-3359",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_image_video_minisft_final_lr8e-5_vtlr2e-5_sohav2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_image_video_minisft_final_lr8e-5_vtlr2e-5_sohav2/20260322024754/hf-6060",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_v21_lr8e-5_vtlr2e-5_sohav2_long_v4": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_v21_lr8e-5_vtlr2e-5_sohav2_long_v4/20260326011735/hf-1251",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_v21_lr8e-5_vtlr2e-5_sohav2_long_v5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_v21_lr8e-5_vtlr2e-5_sohav2_long_v5/20260331111908/hf-893",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_v21base_lr8e-5_vtlr2e-5_long_v1_light": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_v21base_lr8e-5_vtlr2e-5_long_v1_light/20260327114335/hf-2177",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_final_lr8e-5_vtlr2e-5_sohav2_long_v6": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_final_lr8e-5_vtlr2e-5_sohav2_long_v6/20260401113048/hf-1661",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v6": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v6/20260403003502/hf-1717",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v8": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v8/20260405041848/hf-1189",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_final_lr8e-5_vtlr2e-5_sohav2_long_v7": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_final_lr8e-5_vtlr2e-5_sohav2_long_v7/20260404223924/hf-1719",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v9": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v9/20260407024302/hf-1314",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v10": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v10/20260409231400/hf-1246",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v1": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v1/20260411115016/hf-1284",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v11": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v11/20260412022912/hf-1225",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v12": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v12/20260414011034/hf-1010",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v3": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v3/20260413221651/hf-938",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v2/20260413201843/hf-1116",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v4": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v4/20260416110702/hf-858/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v13": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v13/20260417115115/hf-1411",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v14": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v14/20260420004208/hf-1246",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v5/20260418060208/hf-911",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v6": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v6/20260419062108/hf-908",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v7": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v7/20260420084946/hf-907",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v15": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v15/20260421180328/hf-989/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v16": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v16/20260422144816/hf-874",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v17": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v17/20260424125621/hf-1220",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v18": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v18/20260427002311/hf-1257",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v19": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v19/20260428130233/hf-1299",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v20": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v20/20260430040451/hf-1470",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v21": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_v21/20260502012404/hf-1353",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v8": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v8/20260422105827/hf-888",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_v11_online_v1": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_v11_online_v1/20260424040321/hf-925",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_v11_online_v2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_v11_online_v2/20260424072628/hf-992",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_v11_online_v3": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_v11_online_v3/20260426193826/hf-970",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_v11_online_v4": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_v11_online_v4/20260428061201/hf-977",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_v11_online_v5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_v11_online_v5/20260429131219/hf-833",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_s3_v11_ol_v5": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3_online/VideoChat3_4B_train_s3_v11_ol_v5/20260502204453/hf-1445",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_s3_v11_ol_v7": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3_online/VideoChat3_4B_train_s3_v11_ol_v7/20260504213620/hf-1666",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_final260503": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_final260503/20260503140113/hf-6241",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase2_lr8e-5_vtlr2e-5_long_v11": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage3_minisft_finalbase2_lr8e-5_vtlr2e-5_long_v11/20260511002919/hf-1225",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase2_lr8e-5_vtlr2e-5_long_v11clean": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3/VideoChat3_4B_train_stage3_minisft_finalbase2_lr8e-5_vtlr2e-5_long_v11clean/20260513012610/hf-1218",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_s3_v11_ol_v8": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3_online/VideoChat3_4B_train_s3_v11_ol_v8/20260508105753/hf-1555/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_s3_v11_ol_v9": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3_online/VideoChat3_4B_train_s3_v11_ol_v9/20260513162223/hf-1666",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_ab_v15p_llava_video": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_ab_v15p_llava_video/20260514122034/hf-2615",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_ab_v15p_llava_videocot": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_ab_v15p_llava_videocot/20260519004455/hf-2691",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_soha_ab_v15p_llava_videocot_t1": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2/VideoChat3_4B_train_stage2_soha_ab_v15p_llava_videocot_t1/20260522002801/hf-2625",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_minisft_final_lr8e-5_vtlr2e-5_sohav2_ol_v1": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2_new/VideoChat3_4B_train_stage2_minisft_final_lr8e-5_vtlr2e-5_sohav2_ol_v1/20260520195334/hf-6608",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_96k_v11": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3/VideoChat3_4B_train_stage3_minisft_finalbase_lr8e-5_vtlr2e-5_long_96k_v11/20260528035043/hf-868",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_s3new_v11_ol_v1": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3_new/VideoChat3_4B_train_s3new_v11_ol_v1/20260526041946/hf-1547",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_v11base_lr2e-5_seq112k_v11": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4/VideoChat3_4B_train_stage4_v11base_lr2e-5_seq112k_v11/20260529012739/hf-759",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_128k_long_v1": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4/VideoChat3_4B_train_stage4_128k_long_v1/20260601013906/hf-285",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v11base_lr2e-5_v11_OL_v1": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v11base_lr2e-5_v11_OL_v1/20260530042229/hf-751",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_tool_v11base_lr2e-5_v11_seeker": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_tool/VideoChat3_4B_train_stage4_tool_v11base_lr2e-5_v11_seeker/20260529023634/hf-516",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage4_online_v11base_lr2e-5_v11_OL_v2": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage4_online/VideoChat3_4B_train_stage4_online_v11base_lr2e-5_v11_OL_v2/20260531090851/hf-751",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.7, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20
    ),
    "VideoChat3_4B_train_stage2_minisft_final_lr8e-5_vtlr2e-5_sohav2_0609": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage2_new/VideoChat3_4B_train_stage2_minisft_final_lr8e-5_vtlr2e-5_sohav2_0609/20260610171912/hf-6225",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.01, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.001,
        top_k=1
    ),
    "VideoChat3_4B_train_stage3new_lr2e-5_96k_v11newmotion": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3_new/VideoChat3_4B_train_stage3new_lr2e-5_96k_v11newmotion/20260614025125/hf-867",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.01, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.001,
        top_k=1
    ),
    "VideoChat3_4B_train_stage3_minisft_finalbase_lr2e-5_long_96k_v11_internvideo3": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3/VideoChat3_4B_train_stage3_minisft_finalbase_lr2e-5_long_96k_v11_internvideo3/20260615222516/hf-1287/",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.01, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.001,
        top_k=1
    ),
    "VideoChat3_4B": partial(
        VideoChat3,
        model_path="/data/s3/videogpu/zhuyuhan/videochat3/checkpoints/stage3/VideoChat3_4B_train_stage3_minisft_finalbase_lr2e-5_long_96k_v11_SFv2_Streamo_Seeker/20260606011035/hf-1099",
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.01, 
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.001,
        top_k=1
    )
}
    
timelens2_release_series = {
    "TimeLens2-4B": partial(
        Qwen3VLChat,
        model_path=os.environ.get("TIMELENS2_4B_MODEL", "MCG-NJU/TimeLens2-4B"),
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.01,
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.001,
        top_k=1,
    ),
    "TimeLens2-8B": partial(
        Qwen3VLChat,
        model_path=os.environ.get("TIMELENS2_8B_MODEL", "MCG-NJU/TimeLens2-8B"),
        use_custom_prompt=False,
        use_vllm=False,
        temperature=0.01,
        max_new_tokens=4096,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.001,
        top_k=1,
    ),
}

supported_VLM = {}

model_groups = [
    timelens2_release_series,
    ungrouped, o1_apis, api_models, xtuner_series, qwen_series, llava_series, granite_vision_series,
    internvl_series, yivl_series, xcomposer_series, minigpt4_series, 
    idefics_series, instructblip_series, deepseekvl_series, deepseekvl2_series, 
    janus_series, minicpm_series, cogvlm_series, wemm_series, cambrian_series, 
    chameleon_series, video_models, ovis_series, vila_series, mantis_series,
    mmalaya_series, phi3_series, phi4_series, xgen_mm_series, qwen2vl_series,qwen3vl_series,
    slime_series, eagle_series, moondream_series, llama_series, molmo_series, molmo2_series,
    kosmos_series, points_series, nvlm_series, vintern_series, h2ovl_series,
    aria_series, smolvlm_series, sail_series, valley_series, vita_series,
    ross_series, emu_series, ola_series, ursa_series, gemma_series,
    long_vita_series, ristretto_series, kimi_series, aguvis_series, hawkvl_series,
    flash_vl, kimi_vllm_series, oryx_series, treevgr_series, varco_vision_series, qtunevl_series, 
    xvl_series, thyme_series, logics_series, cosmos_series, keye_series, qianfanvl_series, 
    lfm2vl_series, rbdashmm_api_series_lmdeploy, interns1_series, insight_v_series, videochat3_series, videochat_o3_series, videochat_xl_series
]

for grp in model_groups:
    supported_VLM.update(grp)
