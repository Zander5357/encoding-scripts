import vapoursynth as vs
import vsencode as vse

from lvsfunc.types import Range
from vardefunc import initialise_input
from typing import Any, Dict, List, Tuple

from project_module import flt


shader = vse.get_shader(r"C:/Program Files (x86)/mpv/shaders/FSRCNNX_x2_56-16-4-1.glsl")

ini = vse.generate.init_project()

core = vse.util.get_vs_core(reserve_core=ini.reserve_core)


# Sources
JP_BD = vse.FileInfo(f"{ini.bdmv_dir}/USBD/Angel Beats!/NCOP & NCED/00029.m2ts", (0, -24))


NO_AA_RANGES: List[Range] = [  # Ranges for no AA

]

MID_AA_RANGES: List[Range] = [  # Ranges for mediocre AA

]

STRONG_AA_RANGES: List[Range] = [  # Ranges for strong AA
    
]

CHROMA_AA_RANGES: List[Range] = [  # Ranges for chroma AA
    
]

NO_CCD_RANGES: List[Range] = [  # Ranges for no CCD

]

zones: Dict[Tuple[int, int], Dict[str, Any]] = {  # Zoning for the encoder
}


run_script: bool = __name__ == '__main__'


@initialise_input()
def filterchain(src: vs.VideoNode = JP_BD.clip_cut
                ) -> vs.VideoNode | Tuple[vs.VideoNode, ...]:
    """Main VapourSynth filterchain"""
    from awsmfunc import bbmod
    from lvsfunc.deblock import vsdpir
    from rekt import rektlvls
    from vardefunc import Graigasm, AddGrain, to_444
    from vsutil import depth

    #----- Importing source -----#
    src = JP_BD.clip_cut


    #-------- Edge fixing -------#
    rkt = rektlvls(src, rownum=[0, 1] + [1078, 1079], rowval=[7, -7] + [-7, 7], colnum=[0, 1] + [1918, 1919], colval=[7, -7] + [-7, 7])
    ef = bbmod(rkt, left=2, top=2, right=2, bottom=2, y=True, u=False, v=False)
    ef = bbmod(ef, left=2, right=2, y=False, u=True, v=True)
    ef = depth(ef, 16)


    #--------- Rescaling --------#
    rescale = flt.angel_aa(
        ef, descale_height=720, descale_b=0, descale_c=1/2, rep=9, contrasharp=90, mask=True,
        rfactor=1.2, sraa_width=1920, sraa_height=1080, sraa_b=0, sraa_c=1/2,
        alpha=0.25, beta=0.5, gamma=40, nrad=2, mdis=20, vcheck=2, vthresh0=12, vthresh1=24, vthresh2=4)
    chroma_aa = flt.chroma_aa(rescale, rfactor=2.0, alpha=0.4, beta=0.2, gamma=20)  # I gave up on this
    rescale32 = depth(chroma_aa, 32)


    #-------- Deblocking --------#
    rescale_444= to_444(rescale32, 1920, 1080, znedi=False, join_planes=True)
    deblock_444 = vsdpir(rescale_444, strength=20, mode="deblock", matrix=1, cuda=False, i444=True)
    deblock_420 = core.fmtc.resample(deblock_444, css="420")
    deblock = depth(deblock_420, 16)


    #--------- Debanding --------#
    deband = core.average.Mean([
        flt.masked_placebo(deblock, rad=12, thr=2, itr=2, grain=4, mask_args={'detail_brz': 2250, 'lines_brz': 4500}),
        flt.masked_placebo(deblock, rad=16, thr=3, itr=2, grain=4, mask_args={'detail_brz': 2250, 'lines_brz': 4500}),
        flt.masked_placebo(deblock, rad=20, thr=4, itr=2, grain=4, mask_args={'detail_brz': 2250, 'lines_brz': 4500}),
        flt.masked_f3kdb(deblock, rad=16, thr=[32, 24], grain=[24, 12], mask_args={'detail_brz': 1000, 'lines_brz': 2750})
    ])


    #--------- Graining ---------#
    grain = Graigasm(
        thrs=[x << 8 for x in (32, 80, 128, 176)],
        strengths=[(0.4, 0.0), (0.2, 0.0), (0.15, 0.0), (0.0, 0.0)],
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
    vse.EncodeRunner(JP_BD, FILTERED).video('x265', '.settings/x265_settings', zones=zones) \
        .audio('FLAC').mux('Zander5357').run()
elif __name__ == '__vapoursynth__':
    if not isinstance(FILTERED, vs.VideoNode):
        raise vs.Error(f"Input clip has multiple output nodes ({len(FILTERED)})! Please output a single clip")
    else:
        vse.video.finalize_clip(FILTERED).set_output(0)
else:
    JP_BD.clip_cut.set_output(0)

    if not isinstance(FILTERED, vs.VideoNode):
        for i, clip_filtered in enumerate(FILTERED, start=1):
            clip_filtered.set_output(i)
    else:
        FILTERED.set_output(1)

    for i, audio_node in enumerate(JP_BD.audios_cut, start=10):
        if audio_node.bits_per_sample == 32:
            audio_node.set_output(i)