import vapoursynth as vs

from lvsfunc.kernels import Bicubic
from typing import Tuple, NamedTuple, Callable, Union, List, Dict, Any, Optional
from vsmask.better_vsutil import join
from vsmask.edge import SobelStd
from vsmask.util import XxpandMode, expand, inpand
from vsutil import depth, get_w, get_y, join, plane

core = vs.core

class Thr(NamedTuple):
    lo: float
    hi: float

class SSIMDownscaler(Bicubic):
    def scale(self, clip: vs.VideoNode, width: int, height: int, shift: Tuple[float, float] = (0, 0)) -> vs.VideoNode:

        from muvsfunc import SSIM_downsample

        return SSIM_downsample(clip, width, height, smooth=((3 ** 2 - 1) / 12) ** 0.5,
                                sigmoid=True, filter_param_a=self.b, filter_param_b=self.c, **self.kwargs)

def detail_mask(clip: vs.VideoNode, sigma: float = 1.0,
                detail_brz: int = 2500, lines_brz: int = 4500,
                blur_func: Callable[[vs.VideoNode, vs.VideoNode, float],
                                    vs.VideoNode] = core.bilateral.Bilateral,  # type: ignore
                edgemask_func: Callable[[vs.VideoNode], vs.VideoNode] = core.std.Prewitt,
                rg_mode: int = 17) -> vs.VideoNode:
    """
    A detail mask aimed at preserving as much detail as possible within darker areas,
    even if it winds up being mostly noise.
    Currently still in the beta stage.
    Please report any problems or feedback in the IEW Discord (link in the README).
    :param clip:            Input clip
    :param sigma:           Sigma for the detail mask.
                            Higher means more detail and noise will be caught.
    :param detail_brz:      Binarizing for the detail mask.
                            Default values assume a 16bit clip, so you may need to adjust it yourself.
                            Will not binarize if set to 0.
    :param lines_brz:       Binarizing for the prewitt mask.
                            Default values assume a 16bit clip, so you may need to adjust it yourself.
                            Will not binarize if set to 0.
    :param blur_func:       Blurring function used for the detail detection.
                            Must accept the following parameters: ``clip``, ``ref_clip``, ``sigma``.
    :param edgemask_func:   Edgemasking function used for the edge detection
    :param rg_mode:         Removegrain mode performed on the final output
    """
    import lvsfunc as lvf
    from vsutil import get_y, iterate

    if clip.format is None:
        raise ValueError("detail_mask: 'Variable-format clips not supported'")

    clip_y = get_y(clip)
    blur_pf = core.bilateral.Gaussian(clip_y, sigma=sigma / 4 * 3)

    blur_pref = blur_func(clip_y, blur_pf, sigma)
    blur_pref_diff = core.std.Expr([blur_pref, clip_y], "x y -").std.Deflate()
    blur_pref = iterate(blur_pref_diff, core.std.Inflate, 4)

    prew_mask = edgemask_func(clip_y).std.Deflate().std.Inflate()

    if detail_brz > 0:
        blur_pref = blur_pref.std.Binarize(detail_brz)
    if lines_brz > 0:
        prew_mask = prew_mask.std.Binarize(lines_brz)

    merged = core.std.Expr([blur_pref, prew_mask], "x y +")
    rm_grain = lvf.util.pick_removegrain(merged)(merged, rg_mode)
    return depth(rm_grain, clip.format.bits_per_sample)

def angel_aa(clip: vs.VideoNode, descale_height: int = 720, descale_b: float = 0, descale_c: float = 1/2, rep: Optional[int] = 13, contrasharp: Optional[int] = 20, mask: bool = True,
             rfactor: float = 2.0, sraa_width: int = 1920, sraa_height: int = 1080, sraa_b: int = 0, sraa_c: int = 1/2,
             alpha: float = 0.25, beta: float = 0.5, gamma: float = 40, nrad: int = 2, mdis: int = 20, vcheck: int = 2, vthresh0: int = 12, vthresh1: int = 24, vthresh2: int = 4) -> vs.VideoNode:
    """
    Not sure if i am memeing here. Feel free to bonk me on discord
    """
    from fine_dehalo import fine_dehalo
    from vardefunc import upscaled_sraa, Nnedi3SS, Eedi3SR, nnedi3_upscale, merge_chroma
    from havsfunc import LSFmod

    clip_y = get_y(clip)
    clip32_y = depth(clip_y, 32)

    descale = Bicubic(descale_b, descale_c).descale(clip32_y, get_w(descale_height), descale_height)
    upscale = nnedi3_upscale(descale, use_znedi=False, nsize=0, nns=4, qual=2)

    aaa = upscaled_sraa(
        upscale, rfactor=rfactor, width=sraa_width, height=sraa_height,
        supersampler=Nnedi3SS(opencl=True, nsize=0, nns=4, qual=2),
        downscaler=SSIMDownscaler(sraa_b, sraa_c),
        singlerater=Eedi3SR(eedi3cl=False, nnedi3cl=True, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, eedi3_args=dict(vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, hp=False, ucubic=True, cost3=True), nnedi3_args=dict(nsize=0, nns=4, qual=2))
    )
    aaa = depth(aaa, 16)
    aaa = aaa.rgvs.Repair(clip_y, rep) if rep else aaa
    aaa = LSFmod(aaa, strength=contrasharp, Smode=3, Lmode=1, edgemode=1, edgemaskHQ=True) if rep else aaa

    if mask:
        lmask = SobelStd().edgemask(depth(clip_y, 32)).akarin.Expr('x 0 1 clamp')
        lmask = expand(lmask, 2, 2, mode=XxpandMode.ELLIPSE)
        lmask = inpand(lmask, 1, 1, mode=XxpandMode.ELLIPSE)
        lmask = depth(lmask, 16)
        aaa = core.std.MaskedMerge(clip_y, aaa, lmask)

    fd = fine_dehalo(aaa, darkstr=0, brightstr=0.7, rx=2.0, ry=2.0)
    dehalo_min = core.std.Expr([aaa, fd], "x y min")

    if clip.format.num_planes == 1:
        return dehalo_min
    return merge_chroma(dehalo_min, clip)

def chroma_aa(clip: vs.VideoNode, rfactor: float = 2.0, width: int = 1920, height: int = 1080,
              alpha: float = 0.25, beta: float = 0.5, gamma: float = 40, nrad: int = 2, mdis: int = 20, vcheck: int = 2, vthresh0: int = 12, vthresh1: int = 24, vthresh2: int = 4) -> vs.VideoNode:
    from lvsfunc.aa import nnedi3, clamp_aa
    from vskernels import BicubicDidee
    from vardefunc import upscaled_sraa, Znedi3SS, Eedi3SR
    from xvs import WarpFixChromaBlend
    
    sraa_uv = upscaled_sraa(
        clip, rfactor=rfactor, width=width, height=height,
        supersampler=Znedi3SS(nsize=0, nns=4, qual=2),
        downscaler=None,
        singlerater=Eedi3SR(eedi3cl=False, nnedi3cl=True, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, eedi3_args=dict(vcheck=vcheck, vthresh0=vthresh0, vthresh1=vthresh1, vthresh2=vthresh2, hp=False, ucubic=True, cost3=True), nnedi3_args=dict(nsize=0, nns=4, qual=2))
    )
    sraa_u = BicubicDidee().scale(plane(sraa_uv.resize.Bicubic(src_left=0.5, src_top=0.5), 1), 960, 540)
    sraa_v = BicubicDidee().scale(plane(sraa_uv.resize.Bicubic(src_left=0.5, src_top=0.5), 2), 960, 540)

    nnedi_u = nnedi3(opencl=True)(plane(clip.std.Transpose(), 1))
    nnedi_u = nnedi3(opencl=True)(nnedi_u.std.Transpose()).resize.Bicubic(src_left=0.5, src_top=0.5)
    nnedi_u = BicubicDidee().scale(nnedi_u, 960, 540)

    nnedi_v = nnedi3(opencl=True)(plane(clip.std.Transpose(), 2))
    nnedi_v = nnedi3(opencl=True)(nnedi_v.std.Transpose()).resize.Bicubic(src_left=0.5, src_top=0.5)
    nnedi_v = BicubicDidee().scale(nnedi_v, 960, 540)

    clamp_u = clamp_aa(plane(clip, 1), nnedi_u, sraa_u, strength=2.0)
    clamp_v = clamp_aa(plane(clip, 2), nnedi_v, sraa_v, strength=2.0)
    merge = join([plane(clip, 0), clamp_u, clamp_v], vs.YUV)

    return WarpFixChromaBlend(merge, thresh=72, blur=3, btype=1, depth=2)

def masked_f3kdb(clip: vs.VideoNode,
                 rad: int = 16,
                 thr: Union[int, List[int]] = 24,
                 grain: Union[int, List[int]] = [12, 0],
                 mask_args: Dict[str, Any] = {}
                 ) -> vs.VideoNode:
    """Basic f3kdb debanding with detail mask """
    from debandshit import dumb3kdb

    deb_mask_args: Dict[str, Any] = dict(detail_brz=1500, lines_brz=1000)
    deb_mask_args |= mask_args

    bits, clip = _get_bits(clip)

    deband_mask = detail_mask(clip, **deb_mask_args)

    deband = dumb3kdb(clip, radius=rad, threshold=thr, grain=grain)
    deband_masked = core.std.MaskedMerge(deband, clip, deband_mask)
    return deband_masked if bits == 16 else depth(deband_masked, bits)

def masked_placebo(clip: vs.VideoNode,
                   rad: int = 12, thr: Union[int, List[int]] = 4,
                   itr: int = 2, grain: int = 2,
                   mask_args: Dict[str, Any] = {}
                   ) -> vs.VideoNode:
    """Basic placebo debanding with detail mask"""
    deb_mask_args: Dict[str, Any] = dict(detail_brz=1750, lines_brz=4000)
    deb_mask_args |= mask_args

    bits, clip = _get_bits(clip)

    deband_mask = detail_mask(clip, **deb_mask_args)

    deband = placebo_debander(clip, radius=rad, threshold=thr, grain=grain, iterations=itr)
    deband_masked = core.std.MaskedMerge(deband, clip, deband_mask)
    deband_masked = deband_masked if bits == 16 else depth(deband_masked, bits)
    return deband_masked

def placebo_debander(clip: vs.VideoNode, grain: int = 4, **deband_args: Any) -> vs.VideoNode:
    return join([  # Still not sure why splitting it up into planes is faster, but hey!
        core.placebo.Deband(plane(clip, 0), grain=grain, **deband_args),
        core.placebo.Deband(plane(clip, 1), grain=0, **deband_args),
        core.placebo.Deband(plane(clip, 2), grain=0, **deband_args)
    ])

# Helper
def _get_bits(clip: vs.VideoNode, expected_depth: int = 16) -> Tuple[int, vs.VideoNode]:
    from vsutil import get_depth

    bits = get_depth(clip)
    return bits, depth(clip, expected_depth) if bits != expected_depth else clip