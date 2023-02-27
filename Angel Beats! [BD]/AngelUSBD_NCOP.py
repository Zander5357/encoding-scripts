import vapoursynth as vs
import vsencode as vse

from typing import Any, Dict, Tuple
from vardefunc import initialise_input

from project_module import flt


ini = vse.generate.init_project("x265")

core = vse.util.get_vs_core(reserve_core=ini.reserve_core)


# Sources
US_BD = vse.FileInfo(f"{ini.bdmv_dir}/USBD/Angel Beats!/NCOP & NCED/00029.m2ts", (None, -24))
JP_BD = vse.FileInfo(f"{ini.bdmv_dir}/JPBD/Angel Beats!/NCOP & NCED/00001.m2ts", (None, -24))


zones: Dict[Tuple[int, int], Dict[str, Any]] = {  # Zoning for the encoder
}


run_script: bool = __name__ == '__main__'


@initialise_input()
def filterchain(src: vs.VideoNode = US_BD.clip_cut
                ) -> vs.VideoNode | Tuple[vs.VideoNode, ...]:
    """Main VapourSynth filterchain"""
    from awsmfunc import bbmod
    from lvsfunc.deblock import dpir
    from rekt.rektlvls import rektlvls
    from vardefunc import Graigasm, AddGrain, to_444
    from vsdenoise.bm3d import Profile, BM3DCudaRTC
    from vstools import depth
    
    from vstools import plane, get_y, join
       

    #----- Importing source -----#
    src = US_BD.clip_cut
    # src2 = JP_BD.clip_cut 
    # The brightness between USBD & JPBD is different and I have no idea how to fix it.
    # Let me know in Discord if you have any idea on how to fix it, cheers.


    #-------- Edge fixing -------#
    rkt = rektlvls(src, rownum=[0, 1] + [1078, 1079], rowval=[7, -7] + [-7, 7], colnum=[0, 1] + [1918, 1919], colval=[7, -7] + [-7, 7])
    ef = bbmod(rkt, left=2, top=2, right=2, bottom=2, y=True, u=False, v=False)
    ef = bbmod(ef, left=2, right=2, y=False, u=True, v=True)
    ef = depth(ef, 16)


    #--------- Rescaling --------#
    rescale = flt.angel_aa(ef, descale_height=720, descale_b=0, descale_c=1/2, mask=True, rfactor=1.2, alpha=0.25, beta=0.5, gamma=40, nrad=2, mdis=20,
                           rx=2.0, ry=2.0, darkstr=0, brightstr=0.7)
    # chroma = flt.chroma_aa(rescale, transpose=True, clamp_strength=2.0, opencl=True) # I gave up on this


    #-------- Deblocking & Denoising --------#
    # degrain = flt.degrain(chroma, thSAD=75) # The source is pretty grainy and dpir can't reduce the grain much thus make it harder to deband.
    # rescale32 = depth(chroma, 32)
    # rescale_444= to_444(rescale32, 1920, 1080, znedi=False, join_planes=True)
    # deblock_444 = dpir(rescale_444, strength=20, mode="deblock", matrix=1, cuda=True, i444=True) # > strength=20. yes, the blocking is real.
    # deblock_420 = core.fmtc.resample(deblock_444, css="420")
    # deblock = depth(deblock_420, 16)

    rescale32 = depth(rescale, 32)
    rescale444 = to_444(rescale32, 1920, 1080, znedi=False, join_planes=True)
    deblock_y = dpir(get_y(rescale444), strength=50, mode="deblock", matrix=1, cuda=False, i444=False)
    deblock_uv = dpir(rescale444, strength=50, mode="deblock", matrix=1, cuda=True, i444=True)
    deblock_uv = core.fmtc.resample(deblock_uv, css="420")
    deblock = join(deblock_y, deblock_uv)
    deblock = depth(deblock, 16)
    denoise = BM3DCudaRTC(deblock, sigma=[1.25, 0], radius=1, profile=Profile.LOW_COMPLEXITY, matrix=1).clip


    #--------- Debanding --------# 
    deband = core.average.Mean([
        flt.masked_f3kdb(denoise, rad=16, thr=[40, 24], grain=[24, 12], mask_args={'detail_brz': 0.007, 'lines_brz': 0.02}),
        flt.masked_f3kdb(denoise, rad=20, thr=[48, 24], grain=[24, 12], mask_args={'detail_brz': 0.007, 'lines_brz': 0.02}),
        flt.masked_placebo(denoise, rad=16, thr=4.0, itr=2, grain=4, mask_args={'detail_brz': 0.007, 'lines_brz': 0.02})
    ])
    

    #--------- Graining ---------#
    grain = Graigasm(
        thrs=[x << 8 for x in (32, 80, 128, 176)],
        strengths=[(0.6, 0.0), (0.4, 0.0), (0.25, 0.0), (0.15, 0.0)],
        sizes=(1.20, 1.15, 1.10, 1),
        sharps=(80, 70, 60, 50),
        grainers=[
            AddGrain(seed=69420, constant=True),
            AddGrain(seed=69420, constant=False),
            AddGrain(seed=69420, constant=False)
        ]).graining(deband)

    return grain
    

FILTERED = filterchain()


if __name__ == '__main__':
    vse.EncodeRunner(US_BD, FILTERED).video('x265', '_settings/x265_settings', zones=zones) \
        .audio('FLAC').mux('Zander5357').run()
elif __name__ == '__vapoursynth__':
    if not isinstance(FILTERED, vs.VideoNode):
        raise vs.Error(f"Input clip has multiple output nodes ({len(FILTERED)})! Please output a single clip")
    else:
        vse.video.finalize_clip(FILTERED).set_output(0)
else:
    US_BD.clip_cut.set_output(0)

    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(1)

    for i, audio_node in enumerate(US_BD.audios_cut, start=10):
        if audio_node.bits_per_sample == 32:
            audio_node.set_output(i)
