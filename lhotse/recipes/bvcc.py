import logging
from pathlib import Path
from typing import Dict, Optional, Union
import shutil
import tarfile

from lhotse import (
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.utils import Pathlike, urlretrieve_progress


def download_bvcc(
    target_dir: Pathlike = ".",
    force_download: bool = False,
    base_url: str = "https://zenodo.org/record/6572573/files/",
) -> None:
    main_tar_name = "main.tar.gz"
    ood_tar_name = "ood.tar.gz"

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    corpus_dir = target_dir / "BVCC"

    for tar_name in [ood_tar_name, main_tar_name]:
        tar_path = target_dir / tar_name
        extracted_dir = corpus_dir / tar_name[:-7]
        completed_detector = extracted_dir / ".completed"
        if completed_detector.is_file():
            logging.info(f"Skipping download of because {completed_detector} exists.")
            continue
        if force_download or not tar_path.is_file():
            urlretrieve_progress(
                f"{base_url}/{tar_name}",
                filename=tar_path,
                desc=f"Downloading {tar_name}",
            )
        shutil.rmtree(extracted_dir, ignore_errors=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=corpus_dir)
        completed_detector.touch()

    return corpus_dir


def prepare_bvcc(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)

    recs = {}
    for track in ["main", "ood"]:
        track_dir = (corpus_dir / track).resolve()
        assert track_dir.exists(), f"{track} track dir is missing {track_dir}"
        track_sets = track_dir / "DATA" / "sets"
        track_wav = track_dir / "DATA" / "wav"
        assert (
            track_sets.exists() and track_wav.exists()
        ), f"Have you run data preparation in {track_dir}?"

        # for split in ["test", "dev", "train"]:
        for split in ["dev", "train"]:
            splitp = track_sets / f"{split.upper()}SET"
            assert splitp.exists(), splitp
        recs[track] = RecordingSet.from_dir(
            track_wav, pattern="*.wav", num_jobs=num_jobs
        )

    ood_unlabeledp = (corpus_dir / "ood/DATA/sets/unlabeled_mos_list.txt").resolve()
    assert ood_unlabeledp.exists(), ood_unlabeledp

    manifests = {}
    for track in ["main", "ood"]:
        # for split in ["test", "dev", "train"]:
        for split in ["dev", "train"]:
            logging.info(f"Preparing {track}_{split}")
            track_splitp = track_sets / f"{split.upper()}SET"
            __import__("ipdb").set_trace()
            parse_line = parse_main_line if track == "main" else parse_ood_line
            track_split_sup = SupervisionSet.from_segments(
                gen_supervision_per_utt(
                    sorted(open(track_splitp).readlines()),
                    recs[track],
                    parse_line,
                )
            )
            track_split_recs = recs[track].filter(lambda rec: rec.id in track_split_sup)
            manifests[f"track_{split}"] = {
                "recordings": track_split_recs,
                "supervisions": track_split_sup,
            }

    # Add unlabeled OOD dev data
    ood_wav = (corpus_dir / "ood/DATA/wav").resolve()
    unlabeled_wavpaths = [
        ood_wav / name.strip() for name in open(ood_unlabeledp).readlines()
    ]
    manifests["ood_unlabeled"] = {
        "recordings": RecordingSet.from_recordings(
            Recording.from_file(p) for p in unlabeled_wavpaths
        )
    }

    # Optionally serializing to disc
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for part, d in manifests.items():
            d["recordings"].to_file(output_dir / f"bvcc_recordings_{part}.jsonl.gz")
            if "supervisions" in d:
                d["supervisions"].to_file(
                    output_dir / f"bvcc_supervisions_{part}.jsonl.gz"
                )

    return manifests


def parse_main_line(line):
    """
    For context see phase1-main/README:

    TRAINSET and DEVSET contain the individual ratings from each rater, along with
    some demographic information for the rater.
    The format is as follows:

      sysID,uttID,rating,ignore,listenerinfo

    The listener info is as follows:

      {}_AGERANGE_LISTENERID_GENDER_[ignore]_[ignore]_HEARINGIMPAIRMENT

    """
    sysid, uttid, rating, _ignore, listenerinfo = line.split(",")
    _, agerange, listenerid, listener_mf, _, _, haveimpairment = listenerinfo.split("_")

    assert listener_mf in ["Male", "Female", "Others", "na"], listener_mf
    if listener_mf == "Male":
        listener_mf = "M"
    elif listener_mf == "Female":
        listener_mf = "F"
    elif listener_mf == "Others":
        listener_mf = "O"
    elif listener_mf == "na":
        listener_mf = "na"
    else:
        ValueError(f"Unsupported value {listener_mf}")
    assert haveimpairment in ["EP", "ER", "EE", "Yes", "No"], haveimpairment
    haveimpairment = haveimpairment == "Yes"

    return (
        uttid,
        sysid,
        rating,
        {
            "id": listenerid,
            "M_F": listener_mf,
            "impairment": haveimpairment,
            "age": agerange,
        },
    )


def parse_ood_line(line):
    """
    For context see phase1-ood/README:

    TRAINSET and DEVSET contain the individual ratings from each rater, along with
    some demographic information for the rater.  (TRAINSET only contains information
    about the labeled training data, not for the unlabeled samples.)
    The format is as follows:

      sysID,uttID,rating,ignore,listenerinfo

    The listener info is as follows:

      {}_na_LISTENERID_na_na_na_LISTENERTYPE

    LISTENERTYPE may take the following values:
      EE: speech experts
      EP: paid listeners, native speakers of Chinese (any dialect)
      ER: voluntary listeners

    """
    sysid, uttid, rating, _ignore, listenerinfo = line.split(",")
    _, _, listenerid, _, _, _, listenertype = listenerinfo.split("_")

    assert listenertype in ["EE", "EP", "ER"]

    return (
        uttid,
        sysid,
        rating,
        {"id": listenerid, "type": listenertype},
    )


def gen_supervision_per_utt(lines, recordings, parse_line):
    prev_uttid, prev_sups = None, []
    for line in lines:
        line = line.strip()
        info = parse_line(line)
        uttid = info[0]
        if uttid != prev_uttid:
            yield from segment_from_run(prev_sups, recordings)
            prev_uttid, prev_sups = uttid, [info]
        else:
            prev_sups.append(info)
    if len(prev_sups) > 0:
        yield from segment_from_run(prev_sups, recordings)


def segment_from_run(infos, recordings):

    MOSd = {}
    LISTENERsd = {}
    uttidA, sysidA = None, None

    for uttid, sysid, rating, listenerd in infos:
        listenerid = listenerd.pop("id")
        MOSd[listenerid] = int(rating)
        LISTENERsd[listenerid] = listenerid

        if uttidA is None:
            uttidA = uttid
        else:
            assert uttid == uttidA, f"{uttid} vs {uttidA}"
        if sysidA is None:
            sysidA = sysid
        else:
            assert sysid == sysidA, f"{sysid} vs {sysidA}"
    __import__("ipdb").set_trace()
    if uttidA is not None:
        assert sysidA is not None and len(MOSd) > 0 and len(LISTENERsd) > 0
        if uttidA.endswith(".wav"):
            uttidA = uttidA[:-4]
        duration = recordings[uttidA].duration

        yield SupervisionSegment(
            id=uttidA,
            recording_id=uttidA,
            start=0,
            duration=duration,
            text=None,
            language=None,  # cloud be
            speaker=None,
            custom={"MOS": MOSd, "listeners": LISTENERsd},
        )
