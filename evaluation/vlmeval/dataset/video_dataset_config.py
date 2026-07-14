from vlmeval.dataset import *
from functools import partial

vcrbench_dataset = {
    'VCRBench_8frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=8, pack=False),
    'VCRBench_16frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=16, pack=False),
    'VCRBench_32frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=32, pack=False),
    'VCRBench_64frame_nopack': partial(VCRBench, dataset='VCR-Bench', nframe=64, pack=False),
    'VCRBench_1fps_nopack': partial(VCRBench, dataset='VCR-Bench', fps=1.0, pack=False)
}

mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_64frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=False),
    'MMBench_Video_64frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=64, pack=True),
    'MMBench_Video_1fps_nopack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True)
}

mvbench_dataset = {
    'MVBench_8frame': partial(MVBench, dataset='MVBench', nframe=8),
    'MVBench_64frame': partial(MVBench, dataset='MVBench', nframe=64),
    # MVBench not support fps, but MVBench_MP4 does
    'MVBench_MP4_8frame': partial(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': partial(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

tamperbench_dataset = {
    'MVTamperBench_8frame': partial(MVTamperBench, dataset='MVTamperBench', nframe=8),
    'MVTamperBenchStart_8frame': partial(MVTamperBench, dataset='MVTamperBenchStart', nframe=8),
    'MVTamperBenchEnd_8frame': partial(MVTamperBench, dataset='MVTamperBenchEnd', nframe=8),
}

videomme_dataset = {
    'Video-MME_2fps_limit_768': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768),
    'Video-MME_short_2fps_limit_768': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='short'),
    'Video-MME_medium_2fps_limit_768': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='medium'),
    'Video-MME_long_2fps_limit_768': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='long'),
    'Video-MME_2fps_limit_512': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=512),
    'Video-MME_short_2fps_limit_512': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=512, duration_filter='short'),
    'Video-MME_medium_2fps_limit_512': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=512, duration_filter='medium'),
    'Video-MME_long_2fps_limit_512': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=512, duration_filter='long'),
    'Video-MME_4fps_limit_512': partial(VideoMME, dataset='Video-MME', fps=4.0, frames_limit=512),
    'Video-MME_short_1fps_limit_512': partial(VideoMME, dataset='Video-MME', fps=1.0, frames_limit=512, duration_filter='short'),
    'Video-MME_short_4fps_limit_512': partial(VideoMME, dataset='Video-MME', fps=4.0, frames_limit=512, duration_filter='short'),
    'Video-MME_2fps': partial(VideoMME, dataset='Video-MME', fps=2.0),
    'Video-MME_long_2fps_limit_1024': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1024, duration_filter='long'),
    'Video-MME_long_2fps_limit_1536': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1536, duration_filter='long'),
    'Video-MME_short_2fps': partial(VideoMME, dataset='Video-MME', fps=2.0, duration_filter='short'),
    'Video-MME_medium_2fps': partial(VideoMME, dataset='Video-MME', fps=2.0, duration_filter='medium'),
    'Video-MME_long_2fps': partial(VideoMME, dataset='Video-MME', fps=2.0, duration_filter='long'),
    'Video-MME_0.5fps_subs': partial(VideoMME, dataset='Video-MME', fps=0.5, use_subtitle=True),
    'Video-MME_long_2fps_limit_768_448px_48kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=96000*4*14*14),
    'Video-MME_long_2fps_limit_1024_448px_48kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1024, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=96000*4*14*14),
    'Video-MME_long_2fps_limit_1536_448px_48kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1536, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=96000*4*14*14),
    'Video-MME_long_2fps_limit_768_448px_96kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=96000*2*4*14*14),
    'Video-MME_short_2fps_limit_768_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='short', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_medium_2fps_limit_768_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='medium', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_long_2fps_limit_768_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_short_2fps_limit_1024_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1024, duration_filter='short', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_medium_2fps_limit_1024_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1024, duration_filter='medium', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_long_2fps_limit_1024_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1024, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_short_2fps_limit_2048_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=2048, duration_filter='short', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_medium_2fps_limit_2048_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=2048, duration_filter='medium', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_long_2fps_limit_2048_448px_80kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=2048, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Video-MME_long_2fps_limit_1024_448px_96kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1024, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=96000*2*4*14*14),
    'Video-MME_long_2fps_limit_1024_448px_128kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=1024, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=128000*2*4*14*14),
    'Video-MME_long_2fps_limit_2048_448px_64kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=2048, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=64000*2*4*14*14),
    'Video-MME_short_1fps_limit_16': partial(VideoMME, dataset='Video-MME', fps=1.0, frames_limit=16, duration_filter='short'),
    'Video-MME_medium_1fps_limit_16': partial(VideoMME, dataset='Video-MME', fps=1.0, frames_limit=16, duration_filter='medium'),
    'Video-MME_long_1fps_limit_16': partial(VideoMME, dataset='Video-MME', fps=1.0, frames_limit=16, duration_filter='long'),
    'Video-MME_long_2fps_limit_768_448px_100kctx': partial(VideoMME, dataset='Video-MME', fps=2.0, frames_limit=768, duration_filter='long', min_pixels=28*28, max_pixels=448*448, total_pixels=100000*2*4*14*14),
}

videommev2_dataset = {
    'Video-MME-v2_64frame': partial(VideoMMEv2, dataset='Video-MME-v2', nframe=64),
    'Video-MME-v2_1fps_limit_512': partial(VideoMMEv2, dataset='Video-MME-v2', fps=1.0, frames_limit=512),
    'Video-MME-v2_64frame_resize': partial(VideoMMEv2, dataset='Video-MME-v2', nframe=64, resize_target_area=448 * 448),
    'Video-MME-v2_1fps_limit_512_resize': partial(VideoMMEv2, dataset='Video-MME-v2', fps=1.0, frames_limit=512, resize_target_area=448 * 448),
    'Video-MME-v2_2fps_limit_512_resize': partial(VideoMMEv2, dataset='Video-MME-v2', fps=2.0, frames_limit=512, resize_target_area=448 * 448),
    'Video-MME-v2_1fps_limit_512_448px_80kctx': partial(VideoMMEv2, dataset='Video-MME-v2', fps=1.0, frames_limit=512, resize_target_area=448 * 448, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

videommmu_dataset = {
    'VideoMMMU_8frame': partial(VideoMMMU, dataset='VideoMMMU', nframe=8),
    'VideoMMMU_64frame': partial(VideoMMMU, dataset='VideoMMMU', nframe=64),
    'VideoMMMU_2fps_limit_768': partial(VideoMMMU, dataset='VideoMMMU', fps=2.0, frames_limit=768),
    'VideoMMMU_2fps_limit_512': partial(VideoMMMU, dataset='VideoMMMU', fps=2.0, frames_limit=512),
    'VideoMMMU_2fps': partial(VideoMMMU, dataset='VideoMMMU', fps=2.0),
    'VideoMMMU_1fps': partial(VideoMMMU, dataset='VideoMMMU', fps=1.0),
    'VideoMMMU_0.5fps': partial(VideoMMMU, dataset='VideoMMMU', fps=0.5),
    'VideoMMMU_1fps_limit_32': partial(VideoMMMU, dataset='VideoMMMU', fps=1.0, frames_limit=32),
    'VideoMMMU_2fps_limit_512_768px_80kctx': partial(VideoMMMU, dataset='VideoMMMU', fps=2.0, frames_limit=512, min_pixels=28*28, max_pixels=768*768, total_pixels=80000*2*4*14*14),
    'VideoMMMU_2fps_limit_2048_768px_80kctx': partial(VideoMMMU, dataset='VideoMMMU', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=768*768, total_pixels=80000*2*4*14*14),
}

longvideobench_dataset = {
    'LongVideoBench_8frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=8),
    'LongVideoBench_8frame_subs': partial(LongVideoBench, dataset='LongVideoBench', nframe=8, use_subtitle=True),
    'LongVideoBench_64frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=64),
    'LongVideoBench_2fps_limit_768': partial(LongVideoBench, dataset='LongVideoBench', fps=2.0, frames_limit=768),
    'LongVideoBench_2fps_limit_512': partial(LongVideoBench, dataset='LongVideoBench', fps=2.0, frames_limit=512),
    'LongVideoBench_2fps': partial(LongVideoBench, dataset='LongVideoBench', fps=2.0),
    'LongVideoBench_2fps_limit_768_448px_48kctx': partial(LongVideoBench, dataset='LongVideoBench', fps=2.0, frames_limit=768, min_pixels=28*28, max_pixels=448*448, total_pixels=48000*2*4*14*14),
    'LongVideoBench_2fps_limit_2048_448px_64kctx': partial(LongVideoBench, dataset='LongVideoBench', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=64000*2*4*14*14),
    'LongVideoBench_2fps_group3600': partial(LongVideoBench, dataset='LongVideoBench', fps=2.0, group=3600),
    'LongVideoBench_1fps': partial(LongVideoBench, dataset='LongVideoBench', fps=1.0),
    'LongVideoBench_0.5fps': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5),
    'LongVideoBench_0.5fps_subs': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5, use_subtitle=True),
    'LongVideoBench_1fps_limit_64': partial(LongVideoBench, dataset='LongVideoBench', fps=1.0, frames_limit=64),
}

videoevalpro_dataset = {
    'VideoEval-Pro_OpenEnded_2fps_limit_2048_448px_80kctx': partial(VideoEvalPro, dataset='VideoEval-Pro', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'VideoEval-Pro_MCQ_2fps_limit_2048_448px_80kctx': partial(VideoEvalProMCQ, dataset='VideoEval-Pro', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'VideoEval-Pro_OpenEnded_2fps_limit_2048_448px_64kctx': partial(VideoEvalPro, dataset='VideoEval-Pro', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=64000*2*4*14*14),
    'VideoEval-Pro_MCQ_2fps_limit_2048_448px_64kctx': partial(VideoEvalProMCQ, dataset='VideoEval-Pro', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=64000*2*4*14*14),
    'VideoEval-Pro_OpenEnded_2fps_limit_512_448px_32kctx': partial(VideoEvalPro, dataset='VideoEval-Pro', fps=2.0, frames_limit=512, min_pixels=28*28, max_pixels=448*448, total_pixels=32000*2*4*14*14),
    'VideoEval-Pro_MCQ_2fps_limit_512_448px_32kctx': partial(VideoEvalProMCQ, dataset='VideoEval-Pro', fps=2.0, frames_limit=512, min_pixels=28*28, max_pixels=448*448, total_pixels=32000*2*4*14*14),
    'VideoEval-Pro_OpenEnded_64frame': partial(VideoEvalPro, dataset='VideoEval-Pro', nframe=64),
    'VideoEval-Pro_MCQ_64frame': partial(VideoEvalProMCQ, dataset='VideoEval-Pro', nframe=64),
}

lvbench_dataset = {
    'LVBench_8frame': partial(LVBench, dataset='LVBench', nframe=8),
    'LVBench_64frame': partial(LVBench, dataset='LVBench', nframe=64),
    'LVBench_2fps_limit_768': partial(LVBench, dataset='LVBench', fps=2.0, frames_limit=768),
    'LVBench_2fps_limit_512': partial(LVBench, dataset='LVBench', fps=2.0, frames_limit=512),
    'LVBench_2fps_limit_1024': partial(LVBench, dataset='LVBench', fps=2.0, frames_limit=1024),
    'LVBench_2fps_limit_1536': partial(LVBench, dataset='LVBench', fps=2.0, frames_limit=1536),
    'LVBench_2fps': partial(LVBench, dataset='LVBench', fps=2.0),
    'LVBench_1fps': partial(LVBench, dataset='LVBench', fps=1.0),
    'LVBench_2fps_limit_2048_448px_64kctx': partial(LVBench, dataset='LVBench', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=64000*2*4*14*14),
    'LVBench_2fps_limit_2048_448px_100kctx': partial(LVBench, dataset='LVBench', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=100000*2*4*14*14),
    'LVBench_1fps_limit_64': partial(LVBench, dataset='LVBench', fps=1.0, frames_limit=64),
    'LVBench_2fps_limit_2048_448px_80kctx': partial(LVBench, dataset='LVBench', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

mlvu_dataset = {
    'MLVU_8frame': partial(MLVU, dataset='MLVU', nframe=8),
    'MLVU_64frame': partial(MLVU, dataset='MLVU', nframe=64),
    'MLVU_1fps': partial(MLVU, dataset='MLVU', fps=1.0)
}

tempcompass_dataset = {
    'TempCompass_8frame': partial(TempCompass, dataset='TempCompass', nframe=8),
    'TempCompass_64frame': partial(TempCompass, dataset='TempCompass', nframe=64),
    'TempCompass_1fps': partial(TempCompass, dataset='TempCompass', fps=1.0),
    'TempCompass_8fps': partial(TempCompass, dataset='TempCompass', fps=8.0),
    'TempCompass_0.5fps': partial(TempCompass, dataset='TempCompass', fps=0.5),
    'TempCompass_2fps_limit_512': partial(TempCompass, dataset='TempCompass', fps=2.0, frames_limit=512),
    'TempCompass_2fps_limit_768': partial(TempCompass, dataset='TempCompass', fps=2.0, frames_limit=768),
    'TempCompass_MCQ_2fps_limit_512': partial(TempCompass_MCQ, dataset='TempCompass_MCQ', fps=2.0, frames_limit=512),
    'TempCompass_Captioning_2fps_limit_512': partial(TempCompass_Captioning, dataset='TempCompass_Captioning', fps=2.0, frames_limit=512),
    'TempCompass_YorN_2fps_limit_512': partial(TempCompass_YorN, dataset='TempCompass_YorN', fps=2.0, frames_limit=512),
    'TempCompass_1fps_limit_32': partial(TempCompass, dataset='TempCompass', fps=1.0, frames_limit=32),
    'TempCompass_8fps_limit_2048_448px_80kctx': partial(TempCompass, dataset='TempCompass', fps=8.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

# In order to reproduce the experimental results in CGbench paper,
# use_subtitle, use_subtitle_time and use_frame_time need to be set to True.
# When measuring clue-related results, if the number of frames used is greater
# than 32, the frame capture limit will be set to 32.
# We implement the metrics long_acc, clue_acc, miou, CRR, acc@iou and rec@iou
# in the CGBench_MCQ_Grounding_Mini and CGBench_MCQ_Grounding datasets;
# the metric open-ended is implemented in the CGBench_OpenEnded_Mini and CGBench_OpenEnded datasets.
cgbench_dataset = {
    'CGBench_MCQ_Grounding_Mini_8frame_subs_subt': partial(
        CGBench_MCQ_Grounding_Mini,
        dataset='CG-Bench_MCQ_Grounding_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True
    ),
    'CGBench_OpenEnded_Mini_8frame_subs_subt_ft': partial(
        CGBench_OpenEnded_Mini,
        dataset='CG-Bench_OpenEnded_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_MCQ_Grounding_32frame_subs': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=32,
        use_subtitle=True
    ),
    'CGBench_OpenEnded_8frame': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=8
    ),
    'CGBench_MCQ_Grounding_16frame_subs_subt_ft': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_OpenEnded_16frame_subs_subt_ft': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    )
}

megabench_dataset = {
    'MEGABench_core_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="core"),
    'MEGABench_open_16frame': partial(MEGABench, dataset='MEGABench', nframe=16, subset_name="open"),
    'MEGABench_core_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="core"),
    'MEGABench_open_64frame': partial(MEGABench, dataset='MEGABench', nframe=64, subset_name="open")
}

moviechat1k_dataset = {
    'moviechat1k_breakpoint_8frame': partial(MovieChat1k, dataset='MovieChat1k', subset='breakpoint', nframe=8),
    'moviechat1k_global_14frame': partial(MovieChat1k, dataset='MovieChat1k', subset='global', nframe=14),
    'moviechat1k_global_8frame_limit0.01': partial(
        MovieChat1k, dataset='MovieChat1k', subset='global', nframe=8, limit=0.01
    )
}

vdc_dataset = {
    'VDC_8frame': partial(VDC, dataset='VDC', nframe=8),
    'VDC_1fps': partial(VDC, dataset='VDC', fps=1.0),
}

worldsense_dataset = {
    'WorldSense_8frame': partial(WorldSense, dataset='WorldSense', nframe=8),
    'WorldSense_8frame_subs': partial(WorldSense, dataset='WorldSense', nframe=8, use_subtitle=True),
    'WorldSense_8frame_audio': partial(WorldSense, dataset='WorldSense', nframe=8, use_audio=True),
    'WorldSense_32frame': partial(WorldSense, dataset='WorldSense', nframe=32),
    'WorldSense_32frame_subs': partial(WorldSense, dataset='WorldSense', nframe=32, use_subtitle=True),
    'WorldSense_32frame_audio': partial(WorldSense, dataset='WorldSense', nframe=32, use_audio=True),
    'WorldSense_1fps': partial(WorldSense, dataset='WorldSense', fps=1.0),
    'WorldSense_1fps_subs': partial(WorldSense, dataset='WorldSense', fps=1.0, use_subtitle=True),
    'WorldSense_1fps_audio': partial(WorldSense, dataset='WorldSense', fps=1.0, use_audio=True),
    'WorldSense_0.5fps': partial(WorldSense, dataset='WorldSense', fps=0.5),
    'WorldSense_0.5fps_subs': partial(WorldSense, dataset='WorldSense', fps=0.5, use_subtitle=True),
    'WorldSense_0.5fps_audio': partial(WorldSense, dataset='WorldSense', fps=0.5, use_audio=True)
}

qbench_video_dataset = {
    'QBench_Video_8frame': partial(QBench_Video, dataset='QBench_Video', nframe=8),
    'QBench_Video_16frame': partial(QBench_Video, dataset='QBench_Video', nframe=16),
}

video_mmlu_dataset = {
    'Video_MMLU_CAP_16frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=16),
    'Video_MMLU_CAP_64frame': partial(Video_MMLU_CAP, dataset='Video_MMLU_CAP', nframe=64),
    'Video_MMLU_QA_16frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=16),
    'Video_MMLU_QA_64frame': partial(Video_MMLU_QA, dataset='Video_MMLU_QA', nframe=64),
}

video_tt_dataset = {
    'Video_TT_16frame': partial(VideoTT, dataset='Video-TT', nframe=16),
    'Video_TT_32frame': partial(VideoTT, dataset='Video-TT', nframe=32),
    'Video_TT_64frame': partial(VideoTT, dataset='Video-TT', nframe=64),
}

video_holmes_dataset = {
    'Video_Holmes_32frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=32),
    'Video_Holmes_64frame': partial(Video_Holmes, dataset='Video_Holmes', nframe=64),
}

cg_av_counting_dataset = {
    'CG-AV-Counting_32frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=32, use_frame_time=False),
    'CG-AV-Counting_64frame': partial(CGAVCounting, dataset='CG-AV-Counting', nframe=64, use_frame_time=False)
}

egoexobench_dataset = {
    'EgoExoBench_64frame': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=False),  # noqa: E501
    'EgoExoBench_64frame_skip_EgoExo4D': partial(EgoExoBench_MCQ, dataset='EgoExoBench_MCQ', nframe=64, skip_EgoExo4D=True)  # noqa: E501

}

vsibench_dataset = {
    'vsibench_16frame': partial(VSIBench, dataset='VSIBench', nframe=16),
    'vsibench_32frame': partial(VSIBench, dataset='VSIBench', nframe=32),
    'vsibench_64frame': partial(VSIBench, dataset='VSIBench', nframe=64),
}

mmvu_dataset = {
    'MMVU_8frame': partial(MMVU, dataset='MMVU', nframe=8),
    'MMVU_64frame': partial(MMVU, dataset='MMVU', nframe=64),
    'MMVU_2fps_limit_768': partial(MMVU, dataset='MMVU', fps=2.0, frames_limit=768),
    'MMVU_2fps_limit_512': partial(MMVU, dataset='MMVU', fps=2.0, frames_limit=512),
    'MMVU_2fps': partial(MMVU, dataset='MMVU', fps=2.0),
    'MMVU_1fps': partial(MMVU, dataset='MMVU', fps=1.0),
    'MMVU_0.5fps': partial(MMVU, dataset='MMVU', fps=0.5),
    'MMVU_1fps_limit_32': partial(MMVU, dataset='MMVU', fps=1.0, frames_limit=32),
    'MMVU_2fps_limit_512_768px_80kctx': partial(MMVU, dataset='MMVU', fps=2.0, frames_limit=512, min_pixels=28*28, max_pixels=768*768, total_pixels=80000*2*4*14*14),
    'MMVU_2fps_limit_2048_768px_80kctx': partial(MMVU, dataset='MMVU', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=768*768, total_pixels=80000*2*4*14*14),
}

tomato_dataset = {
    'TOMATO_8frame': partial(TOMATO, dataset='TOMATO', nframe=8),
    'TOMATO_64frame': partial(TOMATO, dataset='TOMATO', nframe=64),
    'TOMATO_2fps_limit_768': partial(TOMATO, dataset='TOMATO', fps=2.0, frames_limit=768),
    'TOMATO_1fps_limit_512': partial(TOMATO, dataset='TOMATO', fps=1.0, frames_limit=512),
    'TOMATO_2fps_limit_512': partial(TOMATO, dataset='TOMATO', fps=2.0, frames_limit=512),
    'TOMATO_4fps_limit_512': partial(TOMATO, dataset='TOMATO', fps=4.0, frames_limit=512),
    'TOMATO_8fps': partial(TOMATO, dataset='TOMATO', fps=8.0),
    'TOMATO_2fps': partial(TOMATO, dataset='TOMATO', fps=2.0),
    'TOMATO_1fps': partial(TOMATO, dataset='TOMATO', fps=1.0),
    'TOMATO_0.5fps': partial(TOMATO, dataset='TOMATO', fps=0.5),
    'TOMATO_1fps_limit_32': partial(TOMATO, dataset='TOMATO', fps=1.0, frames_limit=32),
    'TOMATO_8fps_limit_2048_448px_80kctx': partial(TOMATO, dataset='TOMATO', fps=8.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

minerva_dataset = {
    'Minerva_8frame': partial(Minerva, dataset='Minerva', nframe=8),
    'Minerva_16frame': partial(Minerva, dataset='Minerva', nframe=16),
    'Minerva_1fps': partial(Minerva, dataset='Minerva', fps=1.0),
    'Minerva_2fps': partial(Minerva, dataset='Minerva', fps=2.0),
    'Minerva_2fps_limit_512': partial(Minerva, dataset='Minerva', fps=2.0, frames_limit=512),
    'Minerva_2fps_limit_768': partial(Minerva, dataset='Minerva', fps=2.0, frames_limit=768),
    'Minerva_1fps_limit_32': partial(Minerva, dataset='Minerva', fps=1.0, frames_limit=32),
    'Minerva_2fps_limit_512_448px_80kctx': partial(Minerva, dataset='Minerva', fps=2.0, frames_limit=512, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'Minerva_2fps_limit_2048_448px_80kctx': partial(Minerva, dataset='Minerva', fps=2.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

timelens_dataset = {
    'TimeLens_2fps': partial(TimeLens, dataset='TimeLens', fps=2.0),
    'TimeLens_1fps': partial(TimeLens, dataset='TimeLens', fps=1.0),
    'TimeLens_Charades_2fps': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'TimeLens_Charades_1fps': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'TimeLens_ActivityNet_2fps': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'TimeLens_ActivityNet_1fps': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'TimeLens_QVHighlights_2fps': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'TimeLens_QVHighlights_1fps': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'TimeLens_Charades_2fps_limit_768': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=2.0, frames_limit=768),
    'TimeLens_ActivityNet_2fps_limit_768': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=2.0, frames_limit=768),
    'TimeLens_QVHighlights_2fps_limit_768': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=2.0, frames_limit=768),
    'TimeLens_Charades_1fps_limit_512': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=1.0, frames_limit=512),
    'TimeLens_ActivityNet_1fps_limit_512': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=1.0, frames_limit=512),
    'TimeLens_QVHighlights_1fps_limit_512': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=1.0, frames_limit=512),
    'TimeLens_Charades_2fps_limit_512': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=2.0, frames_limit=512),
    'TimeLens_ActivityNet_2fps_limit_512': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=2.0, frames_limit=512),
    'TimeLens_QVHighlights_2fps_limit_512': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=2.0, frames_limit=512),
    'TimeLens_Charades_4fps_limit_512': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=4.0, frames_limit=512),
    'TimeLens_ActivityNet_4fps_limit_512': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=4.0, frames_limit=512),
    'TimeLens_QVHighlights_4fps_limit_512': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=4.0, frames_limit=512),
    'TimeLens_Charades_4fps': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32),
    'TimeLens_ActivityNet_4fps': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32),
    'TimeLens_QVHighlights_4fps': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32),
    'TimeLens_Charades_Reason_4fps': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32, reason=True),
    'TimeLens_ActivityNet_Reason_4fps': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32, reason=True),
    'TimeLens_QVHighlights_Reason_4fps': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=4.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=128000*32*32, reason=True),
    'TimeLens_Charades_2fps_limit_512_px640_ctx51.2k': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=2.0, frames_limit=512, min_pixels=32*32, max_pixels=640*640, total_pixels=51200*32*32),
    'TimeLens_ActivityNet_2fps_limit_512_px640_ctx51.2k': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=2.0, frames_limit=512, min_pixels=32*32, max_pixels=640*640, total_pixels=51200*32*32),
    'TimeLens_QVHighlights_2fps_limit_512_px640_ctx51.2k': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=2.0, frames_limit=512, min_pixels=32*32, max_pixels=640*640, total_pixels=51200*32*32),
    'TimeLens_Charades_1fps_limit_32': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=1.0, frames_limit=32),
    'TimeLens_ActivityNet_1fps_limit_32': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=1.0, frames_limit=32),
    'TimeLens_QVHighlights_1fps_limit_32': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=1.0, frames_limit=32),
    'TimeLens_Charades_4fps_limit_2048_px640_80kctx': partial(TimeLens_Charades, dataset='TimeLens_Charades', fps=4.0, frames_limit=2048, min_pixels=28*28, max_pixels=640*640, total_pixels=80000*2*4*14*14),
    'TimeLens_ActivityNet_2fps_limit_512_448px_80kctx': partial(TimeLens_ActivityNet, dataset='TimeLens_ActivityNet', fps=2.0, frames_limit=512, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
    'TimeLens_QVHighlights_2fps_limit_512_448px_80kctx': partial(TimeLens_QVHighlights, dataset='TimeLens_QVHighlights', fps=2.0, frames_limit=512, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

tvbench_dataset = {
    'TVBench_1fps_limit_512': partial(TVBench, dataset='TVBench', fps=1.0, frames_limit=512),
    'TVBench_2fps_limit_512': partial(TVBench, dataset='TVBench', fps=2.0, frames_limit=512),
    'TVBench_2fps_limit_768': partial(TVBench, dataset='TVBench', fps=2.0, frames_limit=768),
    'TVBench_4fps_limit_512': partial(TVBench, dataset='TVBench', fps=4.0, frames_limit=512),
    'TVBench_1fps_limit_32': partial(TVBench, dataset='TVBench', fps=1.0, frames_limit=32),
    'TVBench_8fps': partial(TVBench, dataset='TVBench', fps=8.0),
    'TVBench_8fps_limit_2048_448px_80kctx': partial(TVBench, dataset='TVBench', fps=8.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

motionbench_dataset = {
    'MotionBench_8frame': partial(MotionBench, dataset='MotionBench', nframe=8),
    'MotionBench_16frame': partial(MotionBench, dataset='MotionBench', nframe=16),
    'MotionBench_1fps': partial(MotionBench, dataset='MotionBench', fps=1.0),
    'MotionBench_2fps': partial(MotionBench, dataset='MotionBench', fps=2.0),
    'MotionBench_8fps': partial(MotionBench, dataset='MotionBench', fps=8.0),
    'MotionBench_2fps_limit_512': partial(MotionBench, dataset='MotionBench', fps=2.0, frames_limit=512),
    'MotionBench_2fps_limit_768': partial(MotionBench, dataset='MotionBench', fps=2.0, frames_limit=768),
    'MotionBench_1fps_limit_32': partial(MotionBench, dataset='MotionBench', fps=1.0, frames_limit=32),
    'MotionBench_8fps_limit_2048_448px_80kctx': partial(MotionBench, dataset='MotionBench', fps=8.0, frames_limit=2048, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

vue_tr_dataset = {
    'VUE_TR_1fps_limit_768': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=768),
    'VUE_TR_1fps_limit_512': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=512),
    'VUE_TR_1fps_limit_128_ctx12.8k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=128, total_pixels=12800*32*32),
    'VUE_TR_1fps': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'VUE_TR_2fps': partial(VUE_TR, dataset='VUE_TR', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'VUE_TR_1fps_limit_512_px480_ctx51.2k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=512, min_pixels=32*32, max_pixels=480*480, total_pixels=51200*32*32),
    'VUE_TR_1fps_limit_512_px480_ctx51.2k_fixprompt': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=512, min_pixels=32*32, max_pixels=480*480, total_pixels=51200*32*32),
    # Grid: {1,2}fps × {512,768} frames × {256,360,480,640} max side × {12800,25600,51200,76800}×32×32 total_pixels; min_pixels=32*32
    'VUE_TR_1fps_limit_768_px480_ctx76.8k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=768, min_pixels=32*32, max_pixels=480*480, total_pixels=76800*32*32),
    'VUE_TR_1fps_limit_768_px480_ctx76.8k_fixprompt': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=768, min_pixels=32*32, max_pixels=480*480, total_pixels=76800*32*32),
    'VUE_TR_1fps_limit_1024_px480_ctx76.8k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=1024, min_pixels=32*32, max_pixels=480*480, total_pixels=76800*32*32),
    'VUE_TR_1fps_limit_1024_px480_ctx102.4k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=1024, min_pixels=32*32, max_pixels=480*480, total_pixels=102400*32*32),
    'VUE_TR_1fps_limit_1024_px480_ctx102.4k_fixprompt': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=1024, min_pixels=32*32, max_pixels=480*480, total_pixels=102400*32*32),
    'VUE_TR_1fps_limit_1024_px480_ctx128k_fixprompt': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=1024, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
    'VUE_TR_1fps_limit_2048_px480_ctx204.8k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=204800*32*32),
    'VUE_TR_1fps_limit_2048_px480_ctx204.8k_fixprompt': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=204800*32*32),
    'VUE_TR_1fps_limit_2048_px480_ctx128k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
    'VUE_TR_1fps_limit_2048_px480_ctx128k_fixprompt': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
    'VUE_TR_Reason_1fps_limit_2048_px480_ctx128k': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32, reason=True),
    'VUE_TR_1fps_limit_512_448px_80kctx': partial(VUE_TR, dataset='VUE_TR', fps=1.0, frames_limit=512, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

moment_seeker_dataset = {
    'MomentSeeker_2fps_limit_768_px480_ctx80k': partial(MomentSeeker, dataset='MomentSeeker', fps=2.0, frames_limit=768, min_pixels=32*32, max_pixels=480*480, total_pixels=80000*2*28*28),
    'MomentSeeker_1fps_limit_128_px480_ctx12.8k': partial(MomentSeeker, dataset='MomentSeeker', fps=1.0, frames_limit=128, min_pixels=32*32, max_pixels=480*480, total_pixels=12800*32*32),
    'MomentSeeker_1fps_limit_512_px480_ctx16k': partial(MomentSeeker, dataset='MomentSeeker', fps=1.0, frames_limit=512, min_pixels=32*32, max_pixels=480*480, total_pixels=16000*32*32),
    'MomentSeeker_1fps_limit_512_px480_ctx51.2k': partial(MomentSeeker, dataset='MomentSeeker', fps=1.0, frames_limit=512, min_pixels=32*32, max_pixels=480*480, total_pixels=51200*32*32),
    'MomentSeeker_2fps_limit_512_px480_ctx51.2k': partial(MomentSeeker, dataset='MomentSeeker', fps=2.0, frames_limit=512, min_pixels=32*32, max_pixels=480*480, total_pixels=51200*32*32),
    'MomentSeeker_1fps_limit_768_px480_ctx76.8k': partial(MomentSeeker, dataset='MomentSeeker', fps=1.0, frames_limit=768, min_pixels=32*32, max_pixels=480*480, total_pixels=76800*32*32),
    'MomentSeeker_2fps_limit_768_px480_ctx76.8k': partial(MomentSeeker, dataset='MomentSeeker', fps=2.0, frames_limit=768, min_pixels=32*32, max_pixels=480*480, total_pixels=76800*32*32),
    'MomentSeeker_1fps_limit_1024_px480_ctx102.4k': partial(MomentSeeker, dataset='MomentSeeker', fps=1.0, frames_limit=1024, min_pixels=32*32, max_pixels=480*480, total_pixels=102400*32*32),
    'MomentSeeker_2fps_limit_1024_px480_ctx102.4k': partial(MomentSeeker, dataset='MomentSeeker', fps=2.0, frames_limit=1024, min_pixels=32*32, max_pixels=480*480, total_pixels=102400*32*32),
    'MomentSeeker_1fps_limit_2048_px480_ctx204.8k': partial(MomentSeeker, dataset='MomentSeeker', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=204800*32*32),
    'MomentSeeker_2fps_limit_2048_px480_ctx204.8k': partial(MomentSeeker, dataset='MomentSeeker', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=204800*32*32),
    'MomentSeeker_2fps_limit_2048_px480_ctx128k': partial(MomentSeeker, dataset='MomentSeeker', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
    'MomentSeeker_Reason_2fps_limit_2048_px480_ctx128k': partial(MomentSeeker, dataset='MomentSeeker', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32, reason=True),
}

ego4d_nlq_v2_dataset = {
    'Ego4D-NLQ-v2_1fps_limit_64_px480_ctx12.8k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=1.0, frames_limit=64, min_pixels=32*32, max_pixels=480*480, total_pixels=12800*32*32),
    'Ego4D-NLQ-v2_1fps_limit_768_px480_ctx76.8k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=1.0, frames_limit=768, min_pixels=32*32, max_pixels=480*480, total_pixels=76800*32*32),
    'Ego4D-NLQ-v2_2fps_limit_768_px480_ctx76.8k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=2.0, frames_limit=768, min_pixels=32*32, max_pixels=480*480, total_pixels=76800*32*32),
    'Ego4D-NLQ-v2_1fps_limit_1024_px480_ctx102.4k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=1.0, frames_limit=1024, min_pixels=32*32, max_pixels=480*480, total_pixels=102400*32*32),
    'Ego4D-NLQ-v2_2fps_limit_1024_px480_ctx102.4k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=2.0, frames_limit=1024, min_pixels=32*32, max_pixels=480*480, total_pixels=102400*32*32),
    'Ego4D-NLQ-v2_1fps_limit_2048_px480_ctx128k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
    'Ego4D-NLQ-v2_2fps_limit_2048_px480_ctx128k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
    'Ego4D-NLQ-v2_Reason_2fps_limit_2048_px480_ctx128k': partial(Ego4DNLQv2, dataset='Ego4D-NLQ-v2', fps=2.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32, reason=True),
}

vue_tr_v2_dataset = {
    'VUE_TR_V2_1fps_limit_128_px480_ctx12.8k': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=128, min_pixels=32*32, max_pixels=480*480, total_pixels=12800*32*32),
    'VUE_TR_V2_1fps_limit_768': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=768),
    'VUE_TR_V2_1fps_limit_512': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=512),    
    'VUE_TR_V2_1fps': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=640*640, total_pixels=96000*32*32),
    'VUE_TR_V2_1fps_limit_512_px480_ctx16k': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=512, min_pixels=32*32, max_pixels=480*480, total_pixels=16000*32*32),
    'VUE_TR_V2_1fps_limit_512_px480_ctx51.2k': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=512, min_pixels=32*32, max_pixels=480*480, total_pixels=51200*32*32),
    'VUE_TR_V2_1fps_limit_768_px480_ctx76.8k_fixprompt': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=768, min_pixels=32*32, max_pixels=480*480, total_pixels=76800*32*32),
    'VUE_TR_V2_1fps_limit_2048_px480_ctx128k': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32),
    'VUE_TR_V2_Reason_1fps_limit_2048_px480_ctx128k': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=2048, min_pixels=32*32, max_pixels=480*480, total_pixels=128000*32*32, reason=True),
    'VUE_TR_V2_1fps_limit_768_px480_80kctx': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=768, min_pixels=28*28, max_pixels=480*480, total_pixels=80000*2*4*14*14),
    'VUE_TR_V2_1fps_limit_768_448px_80kctx': partial(VUE_TR_V2, dataset='VUE_TR_V2', fps=1.0, frames_limit=768, min_pixels=28*28, max_pixels=448*448, total_pixels=80000*2*4*14*14),
}

dream_1k_dataset = {
    'DREAM-1K_8frame': partial(DREAM, dataset='DREAM-1K', nframe=8),
    'DREAM-1K_64frame': partial(DREAM, dataset='DREAM-1K', nframe=64),
    'DREAM-1K_2fps': partial(DREAM, dataset='DREAM-1K', fps=2.0),
    'DREAM-1K_1fps': partial(DREAM, dataset='DREAM-1K', fps=1.0),
    'DREAM-1K_0.5fps': partial(DREAM, dataset='DREAM-1K', fps=0.5),
}

supported_video_datasets = {}

dataset_groups = [
    mmbench_video_dataset, mvbench_dataset, videomme_dataset, videommev2_dataset, videommmu_dataset, longvideobench_dataset,
    videoevalpro_dataset, lvbench_dataset,
    mlvu_dataset, tempcompass_dataset, cgbench_dataset, worldsense_dataset, tamperbench_dataset,
    megabench_dataset, qbench_video_dataset, moviechat1k_dataset, vdc_dataset, video_holmes_dataset, vcrbench_dataset,
    cg_av_counting_dataset, video_mmlu_dataset, egoexobench_dataset, dream_1k_dataset, video_tt_dataset,
    vsibench_dataset, mmvu_dataset, tomato_dataset, minerva_dataset, timelens_dataset, motionbench_dataset, vue_tr_dataset, vue_tr_v2_dataset, moment_seeker_dataset, ego4d_nlq_v2_dataset, tvbench_dataset
]

for grp in dataset_groups:
    supported_video_datasets.update(grp)
