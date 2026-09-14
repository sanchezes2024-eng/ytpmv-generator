import os
import argparse
import subprocess
import math
import random
import mido
import shutil
import re

import numpy as np
from scipy.io import wavfile
from PIL import Image


def verify_environment(ff, vid, mid_f):
    try:
        subprocess.run(
            [ff, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        raise FileNotFoundError(
            f"❌ FFmpeg ('{ff}') was not found on your PATH."
        )

    if not os.path.isfile(vid):
        raise FileNotFoundError(
            f"❌ Video file not found:\n{vid}"
        )

    if not os.path.isfile(mid_f):
        raise FileNotFoundError(
            f"❌ MIDI file not found:\n{mid_f}"
        )


def get_video_info(ff, vid):
    """
    Gets the source video's width, height, and duration.

    This uses FFmpeg itself instead of -show_entries, because
    -show_entries is an ffprobe option, not an FFmpeg option.
    """

    probe = subprocess.run(
        [
            ff,
            "-i",
            vid
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    output = probe.stderr

    # Find the video dimensions.
    #
    # Examples FFmpeg can report:
    #   1920x1080
    #   1280x720
    #   720x1280
    #
    # The first matching resolution after "Video:" is used.

    video_section = re.search(
        r"Video:.*",
        output
    )

    if not video_section:
        raise RuntimeError(
            "❌ FFmpeg could not find a video stream.\n\n"
            + output
        )

    dimensions = re.search(
        r"(\d{2,5})x(\d{2,5})",
        video_section.group(0)
    )

    if not dimensions:
        raise RuntimeError(
            "❌ Could not determine the video's dimensions.\n\n"
            "FFmpeg output:\n"
            + output
        )

    width = int(dimensions.group(1))
    height = int(dimensions.group(2))

    # ------------------------------------------------------------
    # Extract audio temporarily to determine duration.
    # ------------------------------------------------------------

    temp_audio = "t_probe.wav"

    try:
        subprocess.run(
            [
                ff,
                "-y",
                "-i",
                vid,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                temp_audio
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        sample_rate, audio_data = wavfile.read(
            temp_audio
        )

        duration = len(audio_data) / sample_rate

    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)

    return width, height, duration


def extract_pitch_audio(
    ff,
    vid,
    note,
    root,
    start,
    dur,
    out_wav
):
    """
    Extract a section of the source audio and pitch it
    according to the MIDI note.
    """

    multiplier = 2 ** (
        (note - root) / 12.0
    )

    extract_duration = (
        dur + 2.0
    ) * multiplier

    command = [
        ff,
        "-y",

        # Fixed source position.
        "-ss",
        str(start),

        "-t",
        str(extract_duration),

        "-i",
        vid,

        "-vn",

        "-filter:a",
        (
            f"asetrate={int(44100 * multiplier)},"
            f"aresample=44100"
        ),

        out_wav
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def extract_video_frames(
    ff,
    vid,
    start,
    dur,
    tw,
    th
):
    """
    Extract video frames without artificial panning.

    IMPORTANT:

    1. The source timestamp is fixed.
    2. There is no animated crop.
    3. There is no horizontal translation.
    4. There are no color filters.
    5. The source aspect ratio is preserved.
    6. Raw RGB is used instead of PPM.

    The video is scaled to fit inside the grid cell.
    Any unused area is black.
    """

    video_filter = (
        f"scale={tw}:{th}:"
        f"force_original_aspect_ratio=decrease,"
        f"pad={tw}:{th}:"
        f"(ow-iw)/2:"
        f"(oh-ih)/2:"
        f"black"
    )

    command = [
        ff,
        "-y",

        # Seek to one fixed location.
        "-ss",
        str(start),

        "-i",
        vid,

        "-t",
        str(dur),

        # Preserve aspect ratio.
        # NO crop movement.
        # NO color processing.
        "-vf",
        video_filter,

        # Force exact cell size.
        "-s",
        f"{tw}x{th}",

        # Raw RGB output.
        "-f",
        "rawvideo",

        "-pix_fmt",
        "rgb24",

        "-"
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    frame_size = (
        tw *
        th *
        3
    )

    frames = []

    while True:
        raw = process.stdout.read(
            frame_size
        )

        if len(raw) != frame_size:
            break

        frame = Image.frombytes(
            "RGB",
            (tw, th),
            raw
        )

        frames.append(frame)

    process.stdout.close()
    process.wait()

    return frames


def parse_midi(midi_path):
    """
    Read MIDI note events and convert them into
    start/duration information.
    """

    midi = mido.MidiFile(
        midi_path
    )

    notes = []

    current_time = 0.0

    active_notes = {}

    for message in midi:
        current_time += message.time

        channel = getattr(
            message,
            "channel",
            0
        )

        # Note started.
        if (
            message.type == "note_on"
            and message.velocity > 0
        ):
            key = (
                channel,
                message.note
            )

            active_notes.setdefault(
                key,
                []
            ).append(
                current_time
            )

        # Note ended.
        elif message.type in [
            "note_off",
            "note_on"
        ]:
            key = (
                channel,
                message.note
            )

            if (
                key in active_notes
                and active_notes[key]
            ):
                start = active_notes[key].pop(
                    0
                )

                duration = (
                    current_time -
                    start
                )

                if duration > 0:
                    notes.append(
                        {
                            "note": message.note,
                            "channel": channel,
                            "start": start,
                            "duration": duration
                        }
                    )

    return sorted(
        notes,
        key=lambda item: item["start"]
    )


def generate_ytpmv():
    parser = argparse.ArgumentParser(
        description="YTPMV generator"
    )

    parser.add_argument(
        "-v",
        "--video",
        default="source_clip.mp4"
    )

    parser.add_argument(
        "-m",
        "--midi",
        default="melody.mid"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="ytpmv_output.mp4"
    )

    parser.add_argument(
        "-r",
        "--root",
        type=int,
        default=60
    )

    parser.add_argument(
        "-f",
        "--fps",
        type=int,
        default=30
    )

    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # VERIFY
    # ------------------------------------------------------------

    verify_environment(
        args.ffmpeg,
        args.video,
        args.midi
    )

    # ------------------------------------------------------------
    # SOURCE VIDEO INFO
    # ------------------------------------------------------------

    print("🔍 Reading source video...")

    width, height, source_duration = get_video_info(
        args.ffmpeg,
        args.video
    )

    print(
        f"   Source: {width}x{height}"
    )

    print(
        f"   Duration: {source_duration:.2f}s"
    )

    # ------------------------------------------------------------
    # MIDI
    # ------------------------------------------------------------

    print("🎼 Reading MIDI...")

    notes = parse_midi(
        args.midi
    )

    if not notes:
        print("❌ No notes found in MIDI.")
        return

    total_duration = max(
        note["start"] +
        note["duration"]
        for note in notes
    )

    print(
        f"   Notes: {len(notes)}"
    )

    print(
        f"   Output duration: "
        f"{total_duration:.2f}s"
    )

    # ------------------------------------------------------------
    # INSTRUMENTS / MIDI CHANNELS
    # ------------------------------------------------------------

    unique_instruments = sorted(
        set(
            note["channel"]
            for note in notes
        )
    )

    instrument_count = len(
        unique_instruments
    )

    print(
        f"   MIDI channels: "
        f"{instrument_count}"
    )

    # ------------------------------------------------------------
    # GRID
    # ------------------------------------------------------------

    columns = math.ceil(
        math.sqrt(
            instrument_count
        )
    )

    rows = math.ceil(
        instrument_count /
        columns
    )

    cell_width = (
        width // columns
    )

    cell_height = (
        height // rows
    )

    print(
        f"🧱 Grid: "
        f"{columns} x {rows}"
    )

    print(
        f"   Cell size: "
        f"{cell_width}x{cell_height}"
    )

    # ------------------------------------------------------------
    # GRID POSITIONS
    # ------------------------------------------------------------

    slots = {}

    for index, instrument in enumerate(
        unique_instruments
    ):
        x = (
            index % columns
        ) * cell_width

        y = (
            index // columns
        ) * cell_height

        slots[instrument] = (
            x,
            y
        )

    # ------------------------------------------------------------
    # FIXED SOURCE SEGMENTS
    # ------------------------------------------------------------

    random.seed(42)

    instrument_segments = {}

    max_start = max(
        0,
        source_duration - 4.0
    )

    for instrument in unique_instruments:
        instrument_segments[instrument] = random.uniform(
            0,
            max_start
        )

        print(
            f"   Channel {instrument}: "
            f"source position "
            f"{instrument_segments[instrument]:.2f}s"
        )

    # ------------------------------------------------------------
    # AUDIO CACHE
    # ------------------------------------------------------------

    print()
    print(
        "🎵 Pre-rendering pitched audio..."
    )

    os.makedirs(
        "cache_audio",
        exist_ok=True
    )

    audio_length = (
        int(
            total_duration *
            44100
        )
        + 88200
    )

    mixed_audio = np.zeros(
        (
            audio_length,
            2
        ),
        dtype=np.float32
    )

    # ------------------------------------------------------------
    # VIDEO TIMELINE
    # ------------------------------------------------------------

    timeline_length = (
        int(
            total_duration *
            args.fps
        )
        + 2
    )

    video_timeline = [
        []
        for _ in range(
            timeline_length
        )
    ]

    video_name = os.path.splitext(
        os.path.basename(
            args.video
        )
    )[0]

    # ------------------------------------------------------------
    # EXTRACT UNIQUE PITCHED AUDIO
    # ------------------------------------------------------------

    for note in notes:
        wav_path = (
            f"cache_audio/"
            f"{video_name}_"
            f"n_{note['note']}_"
            f"{note['channel']}.wav"
        )

        if not os.path.exists(
            wav_path
        ):
            extract_pitch_audio(
                args.ffmpeg,
                args.video,
                note["note"],
                args.root,
                instrument_segments[
                    note["channel"]
                ],
                total_duration,
                wav_path
            )

    # ------------------------------------------------------------
    # BUILD AUDIO TIMELINE
    # ------------------------------------------------------------

    print(
        "🎛️ Structuring audio timeline..."
    )

    for note in notes:
        wav_path = (
            f"cache_audio/"
            f"{video_name}_"
            f"n_{note['note']}_"
            f"{note['channel']}.wav"
        )

        sample_rate, data = wavfile.read(
            wav_path
        )

        if data.ndim == 1:
            data = np.column_stack(
                (
                    data,
                    data
                )
            )

        multiplier = 2 ** (
            (
                note["note"] -
                args.root
            ) / 12.0
        )

        start_sample = int(
            note["start"] *
            44100
        )

        duration_samples = int(
            note["duration"] *
            44100 *
            multiplier
        )

        clip = data[
            :duration_samples
        ].astype(
            np.float32
        )

        normal_duration = int(
            note["duration"] *
            44100
        )

        if len(clip) > normal_duration:
            clip = clip[
                :normal_duration
            ]

        end_sample = (
            start_sample +
            len(clip)
        )

        if (
            start_sample >= 0
            and end_sample <=
            len(mixed_audio)
        ):
            mixed_audio[
                start_sample:end_sample
            ] += clip

        # --------------------------------------------------------
        # VIDEO TIMELINE
        # --------------------------------------------------------

        start_frame = int(
            note["start"] *
            args.fps
        )

        end_frame = int(
            (
                note["start"] +
                note["duration"]
            ) *
            args.fps
        )

        for frame_number in range(
            start_frame,
            min(
                end_frame,
                len(video_timeline)
            )
        ):
            frame_offset = (
                frame_number -
                start_frame
            )

            video_timeline[
                frame_number
            ].append(
                (
                    note["note"],
                    note["channel"],
                    frame_offset
                )
            )

    # ------------------------------------------------------------
    # NORMALIZE AUDIO
    # ------------------------------------------------------------

    print(
        "🔊 Normalizing audio..."
    )

    maximum = np.max(
        np.abs(
            mixed_audio
        )
    )

    if maximum > 0:
        mixed_audio = (
            mixed_audio /
            maximum *
            32767
        ).astype(
            np.int16
        )
    else:
        mixed_audio = (
            mixed_audio.astype(
                np.int16
            )
        )

    wavfile.write(
        "m_audio.wav",
        44100,
        mixed_audio
    )

    # ------------------------------------------------------------
    # CACHE VIDEO
    # ------------------------------------------------------------

    print()
    print(
        "🖼️ Caching video segments..."
    )

    video_cache = {}

    for instrument in unique_instruments:
        print(
            f"   Channel {instrument}: "
            f"extracting..."
        )

        video_cache[instrument] = (
            extract_video_frames(
                args.ffmpeg,
                args.video,
                instrument_segments[
                    instrument
                ],
                total_duration,
                cell_width,
                cell_height
            )
        )

        print(
            f"      "
            f"{len(video_cache[instrument])} "
            f"frames cached"
        )

    # ------------------------------------------------------------
    # OUTPUT VIDEO
    # ------------------------------------------------------------

    print()
    print(
        "🎬 Rendering master video..."
    )

    output_command = [
        args.ffmpeg,
        "-y",

        # Raw RGB input.
        "-f",
        "rawvideo",

        "-vcodec",
        "rawvideo",

        "-pix_fmt",
        "rgb24",

        "-s",
        f"{width}x{height}",

        "-r",
        str(args.fps),

        "-i",
        "-",

        # Audio.
        "-i",
        "m_audio.wav",

        # Video encoder.
        "-c:v",
        "libx264",

        # Audio encoder.
        "-c:a",
        "aac",

        # Compatible pixel format.
        "-pix_fmt",
        "yuv420p",

        # Make the output duration follow the video.
        "-shortest",

        args.output
    ]

    output_process = subprocess.Popen(
        output_command,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    # ------------------------------------------------------------
    # EMPTY CELL
    # ------------------------------------------------------------

    blank_cell = Image.new(
        "RGB",
        (
            cell_width,
            cell_height
        ),
        "black"
    )

    # ------------------------------------------------------------
    # RENDER
    # ------------------------------------------------------------

    total_frames = len(
        video_timeline
    )

    for frame_index, timeline_frame in enumerate(
        video_timeline
    ):
        canvas = Image.new(
            "RGB",
            (
                width,
                height
            ),
            "black"
        )

        # Fill every grid position.
        for position in slots.values():
            canvas.paste(
                blank_cell,
                position
            )

        # Draw active notes.
        for (
            note_number,
            instrument,
            frame_offset
        ) in timeline_frame:

            cache = video_cache.get(
                instrument,
                []
            )

            if not cache:
                continue

            # Loop through the fixed cached frames.
            frame = cache[
                frame_offset %
                len(cache)
            ]

            position = slots[
                instrument
            ]

            canvas.paste(
                frame,
                position
            )

        output_process.stdin.write(
            canvas.tobytes()
        )

        if (
            frame_index % args.fps
            == 0
        ):
            seconds = (
                frame_index /
                args.fps
            )

            print(
                f"\r   Rendering: "
                f"{seconds:.1f}s / "
                f"{total_duration:.1f}s",
                end="",
                flush=True
            )

    print()

    # ------------------------------------------------------------
    # FINISH FFMPEG
    # ------------------------------------------------------------

    output_process.stdin.close()

    output_process.wait()

    if output_process.returncode != 0:
        raise RuntimeError(
            "❌ FFmpeg failed while rendering the output video."
        )

    # ------------------------------------------------------------
    # CLEANUP
    # ------------------------------------------------------------

    print(
        "🧹 Cleaning temporary files..."
    )

    if os.path.exists(
        "m_audio.wav"
    ):
        os.remove(
            "m_audio.wav"
        )

    if os.path.exists(
        "cache_audio"
    ):
        shutil.rmtree(
            "cache_audio"
        )

    print()
    print(
        "🎉 Complete!"
    )

    print(
        f"📁 Output: {args.output}"
    )


if __name__ == "__main__":
    generate_ytpmv()