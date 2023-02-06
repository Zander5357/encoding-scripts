import vapoursynth as vs

from typing import Any, Dict, List, Tuple, Union
from vskernels import Bicubic, Catrom, ScalerT
from vsmask.edge import SobelStd
from vsmask.util import XxpandMode, expand, inpand
from vsmasktools import detail_mask_neo
from vsscale import SSIM
from vstools import depth, get_w, get_y, join, plane


core = vs.core


def angel_aa(clip: vs.VideoNode, descale_height: int = 720, descale_b: float = 0, descale_c: float = 1/2, mask: bool = True,
             rfactor: float = 2.0, sraa_width: int = 1920, sraa_height: int = 1080, kernel: ScalerT = Catrom,
             alpha: float = 0.25, beta: float = 0.5, gamma: float = 40, nrad: int = 2, mdis: int = 20, vcheck: int = 2, vthresh0: int = 12, vthresh1: int = 24, vthresh2: int = 4,
             rx: float = 1.8, ry: float = 1.8, darkstr: float = 0, brightstr: float = 1.0, thmi: int = 80, thma: int = 128, thlimi=50, thlima=100) -> vs.VideoNode:
    """A wrapper I made. Mostly stolen"""
    from vardefunc import nnedi3_upscale, merge_chroma
    from vsaa import upscaled_sraa, Nnedi3SS, Nnedi3, Eedi3SR
    from vsdehalo import fine_dehalo

    clip_y = get_y(clip)
    clip32_y = depth(clip_y, 32)

    descale = Bicubic(descale_b, descale_c).descale(clip32_y, get_w(descale_height), descale_height)
    upscale = nnedi3_upscale(descale, use_znedi=False, nsize=0, nns=4, qual=2)

    upscaled_sraaa = upscaled_sraa(
        upscale, rfactor=rfactor, width=sraa_width, height=sraa_height,
        ssfunc=Nnedi3SS(opencl=True, nsize=0, nns=4, qual=2),
        downscaler=SSIM(kernel=kernel),
        aafunc=Eedi3SR(opencl=True, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, hp=True, ucubic=True, cost3=True, 
                       sclip_aa=Nnedi3(nsize=0, nns=4, qual=2))
    )
    upscaled_sraaa = depth(upscaled_sraaa, 16)

    if mask:
        lmask = SobelStd().edgemask(depth(clip_y, 32)).akarin.Expr('x 0 1 clamp')
        lmask = expand(lmask, 2, 2, mode=XxpandMode.ELLIPSE)
        lmask = inpand(lmask, 1, 1, mode=XxpandMode.ELLIPSE)
        lmask = depth(lmask, 16)
        upscaled_sraaa = core.std.MaskedMerge(clip_y, upscaled_sraaa, lmask)

    fd = fine_dehalo(upscaled_sraaa, darkstr=darkstr, brightstr=brightstr, rx=rx, ry=ry, thmi=thmi, thma=thma, thlimi=thlimi, thlima=thlima)
    dehalo_min = core.std.Expr([upscaled_sraaa, fd], "x y min")

    if clip.format.num_planes == 1:
        return dehalo_min
    return merge_chroma(dehalo_min, clip)

def chroma_aa(clip: vs.VideoNode, transpose: bool = False, clamp_strength: float = 1.0, opencl: bool = False) -> vs.VideoNode:
    """An attempt to fix the chroma"""
    from vsaa import clamp_aa, Nnedi3, Eedi3
    from vskernels import BicubicDidee
    from xvs import WarpFixChromaBlend

    nnedi_u = Nnedi3(opencl=opencl, transpose_first=transpose).aa(plane(clip, 1))
    nnedi_u = Nnedi3(opencl=opencl, transpose_first=transpose).aa(nnedi_u)
    nnedi_u = BicubicDidee().scale(nnedi_u, 960, 540)
    
    nnedi_v = Nnedi3(opencl=opencl, transpose_first=transpose).aa(plane(clip, 2))
    nnedi_v = Nnedi3(opencl=opencl, transpose_first=transpose).aa(nnedi_v)
    nnedi_v = BicubicDidee().scale(nnedi_v, 960, 540)
    
    eedi_u = Eedi3(opencl=opencl, transpose_first=transpose).aa(plane(clip, 1))
    eedi_u = Eedi3(opencl=opencl, transpose_first=transpose).aa(eedi_u)
    eedi_u = BicubicDidee().scale(eedi_u, 960, 540)
    
    eedi_v = Eedi3(opencl=opencl, transpose_first=transpose).aa(plane(clip, 2))
    eedi_v = Eedi3(opencl=opencl, transpose_first=transpose).aa(eedi_v)
    eedi_v = BicubicDidee().scale(eedi_v, 960, 540)

    clamp_u = clamp_aa(plane(clip, 1), nnedi_u, eedi_u, strength=clamp_strength)
    clamp_v = clamp_aa(plane(clip, 2), nnedi_v, eedi_v, strength=clamp_strength)
    merge = join([plane(clip, 0), clamp_u, clamp_v], vs.YUV)
    return WarpFixChromaBlend(merge, thresh=72, blur=3, btype=1, depth=2)

def degrain(clip: vs.VideoNode, thSAD: int = 200, prefilter: vs.VideoNode = None) -> vs.VideoNode:
    """Stolen from IEW"""
    from vsdenoise import MVTools, Prefilter, SADMode, MotionMode, SearchMode
    
    mv = MVTools(
        clip,
        prefilter=prefilter if prefilter else Prefilter.NONE,
        params_curve=False,
    )
    mv.analyze(
        sharp=2,
        rfilter=4,
        block_size=32,
        overlap=16,
        thSAD=thSAD,
        sad_mode=SADMode.SPATIAL.same_recalc,
        motion=MotionMode.HIGH_SAD,
        search=SearchMode.DIAMOND.defaults,
    )
    return mv.degrain()

def masked_f3kdb(clip: vs.VideoNode,
                 rad: int = 16,
                 thr: Union[int, List[int]] = 24,
                 grain: Union[int, List[int]] = [12, 0],
                 mask_args: Dict[str, Any] = {}
                 ) -> vs.VideoNode:
    """Basic f3kdb debanding with detail mask"""
    from vsdeband.f3kdb import F3kdb

    deb_mask_args: Dict[str, Any] = dict(detail_brz=0.05, lines_brz=0.08)
    deb_mask_args |= mask_args

    deband_mask = detail_mask_neo(clip, **deb_mask_args)

    deband = F3kdb.deband(clip, radius=rad, thr=thr, grains=grain)
    return core.std.MaskedMerge(deband, clip, deband_mask)

def placebo_debander(clip: vs.VideoNode, grain: int = 4, **deband_args: Any) -> vs.VideoNode:
    return join([  # Still not sure why splitting it up into planes is faster, but hey!
        core.placebo.Deband(plane(clip, 0), grain=grain, **deband_args),
        core.placebo.Deband(plane(clip, 1), grain=0, **deband_args),
        core.placebo.Deband(plane(clip, 2), grain=0, **deband_args)
    ])

def masked_placebo(clip: vs.VideoNode,
                   rad: int = 12, thr: Union[int, List[int]] = 4,
                   itr: int = 2, grain: int = 2,
                   mask_args: Dict[str, Any] = {}
                   ) -> vs.VideoNode:
    """Basic placebo debanding with detail mask"""
    deb_mask_args: Dict[str, Any] = dict(detail_brz=0.05, lines_brz=0.08)
    deb_mask_args |= mask_args

    deband_mask = detail_mask_neo(clip, **deb_mask_args)

    deband = placebo_debander(clip, radius=rad, threshold=thr, grain=grain, iterations=itr)
    return core.std.MaskedMerge(deband, clip, deband_mask)

def placebo_debander(clip: vs.VideoNode, grain: int = 4, **deband_args: Any) -> vs.VideoNode:
    return join([  # Still not sure why splitting it up into planes is faster, but hey!
        core.placebo.Deband(plane(clip, 0), grain=grain, **deband_args),
        core.placebo.Deband(plane(clip, 1), grain=0, **deband_args),
        core.placebo.Deband(plane(clip, 2), grain=0, **deband_args)
    ])
