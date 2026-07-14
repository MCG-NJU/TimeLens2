import os
import warnings

from ..smp import load
from .utils.video_pyav import get_video_decode_backend
from .video_meta_cache import ensure_video_meta_json
from .vidi_vue_tr import VUE_TR, _VUE_TR_Paths, _ensure_vue_tr_tsv, _index_video_files


def _default_vue_tr_v2_root() -> str:
    return os.environ.get(
        'VUE_TR_V2_ROOT',
        'data/benchmarks/VUE_TR_V2',
    )


class VUE_TR_V2(VUE_TR):
    """
    VUE-TR v2: same temporal-retrieval protocol as VUE_TR, different annotation release.
    Ground truth: `VUE-TRv2_ground_truth.json` under `VUE_TR_V2_ROOT`.
    Videos: `<VUE_TR_V2_ROOT>/videos/`. Missing mp4 files are skipped when building the TSV.
    Evaluation metrics match `VUE_TR_V2/qa_eval.py` (same as V1 implementation in `vidi_vue_tr.py`).
    Only `query_modality == "vision"` samples are kept (same as VUE_TR).
    """

    def __init__(
        self,
        dataset='VUE_TR_V2',
        nframe=0,
        fps=-1,
        frames_limit=2048,
        min_pixels=28 * 28,
        max_pixels=448 * 448,
        total_pixels=32000 * 2 * 4 * 14 * 14,
        check_extracted_frames=True,
        reason=False,
    ):
        super().__init__(
            dataset=dataset,
            nframe=nframe,
            fps=fps,
            frames_limit=frames_limit,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            check_extracted_frames=check_extracted_frames,
            reason=reason,
        )

    @classmethod
    def supported_datasets(cls):
        return ['VUE_TR_V2']

    def prepare_dataset(self, dataset):
        root = _default_vue_tr_v2_root()
        videos_dir = os.path.join(root, 'videos')
        paths = _VUE_TR_Paths(
            root=root,
            videos_dir=videos_dir,
            gt_json=os.path.join(root, 'VUE-TRv2_ground_truth.json'),
            tsv_path=os.path.join(root, 'VUE_TR_V2.tsv'),
        )
        meta_path = paths.tsv_path.replace('.tsv', '_video_meta.json')
        self._video_meta_path = meta_path
        _ensure_vue_tr_tsv(paths)
        try:
            df = load(paths.tsv_path)
            vid_col = 'video_id' if 'video_id' in df.columns else 'video'
            video_ids = sorted(set(str(x) for x in df[vid_col].dropna().tolist()))
            video_map = _index_video_files(paths.videos_dir)
            ensure_video_meta_json(
                meta_path,
                video_ids,
                lambda video_id: os.path.join(paths.videos_dir, video_map[str(video_id)]),
                prefer_pyav=get_video_decode_backend() == 'pyav',
            )
        except Exception as e:
            warnings.warn(f'[VUE_TR_V2] failed to ensure video meta json: {e}')
        self._videos_dir = paths.videos_dir
        return dict(root=paths.videos_dir, data_file=paths.tsv_path)
