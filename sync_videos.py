import os, sys
import argparse
import logging
import json
import subprocess
import numpy as np
from datetime import datetime, time, timedelta
import re

if sys.platform == 'darwin':
    FFPROBE = '/opt/homebrew/bin/ffprobe'
elif sys.platform == 'win32':
    FFPROBE = ''
else:
    raise(OSError("Could not find ffprobe"))

if not os.path.exists(FFPROBE):
    raise(OSError("Could not find ffprobe at path {}".format(FFPROBE)))

def build_parser():
    parser = argparse.ArgumentParser(
                        prog='sync_videos',
                        description='Synchronize videos based on timecode and audio')
    
    parser.add_argument('-v', '--verbose', help="Show more details",
                        action='store_true')
    parser.add_argument('files', nargs='+', help="Input movie files to synchronize",
                        action='append')
    parser.add_argument('-od', '--output_dir', help="Directory to save the output files",
                        default='.')
    parser.add_argument('-s', '--output_suffix', help="Suffix to add to synchronized output videos",
                        default='-sync')
    
    return parser
    
def get_file_data(filename):
    cmd = [FFPROBE, '-print_format', 'json',
           '-show_format', '-show_streams',
            filename]
    logging.debug("Command: {}".format(' '.join(cmd)))
    r = subprocess.run(cmd, capture_output=True)

    dat = json.loads(r.stdout)
    return(dat)

def get_timecode(dat):
    for s in dat['streams']:
        if s['codec_type'] == 'video':
            creation_time = s['tags']['creation_time']
            frame_rate = s['avg_frame_rate']
            timecode = s['tags']['timecode']
            break
    else:
        raise KeyError("No video stream found")
    
    creation_time = datetime.fromisoformat(str(creation_time).replace('Z', '+00:00'))

    m = re.fullmatch('(\d+)/(\d+)', frame_rate)
    if m is None:
        raise ValueError("Could not parse frame rate {}".format(frame_rate))
    frame_rate = float(m.group(1)) / float(m.group(2))

    m = re.fullmatch('(\d{2})[:;.](\d{2})[:;.](\d{2})[:;.](\d+)', timecode)
    if m is None:
        raise ValueError("Could not parse timecode {}".format(timecode))

    timecodestr = timecode
    hmsf = [int(g) for g in m.groups()]

    us = int(float(hmsf[3]) / frame_rate * 1e6)
    timecode = time(hmsf[0], hmsf[1], hmsf[2], us)

    timecode = datetime(creation_time.year, creation_time.month, creation_time.day,
                        timecode.hour, timecode.minute, timecode.second, timecode.microsecond)

    return timecode, frame_rate

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = build_parser()
    args = parser.parse_args()

    dat = [get_file_data(f) for f in args.files[0]]
    tcfps = [get_timecode(d) for d in dat]
    tc = [t[0] for t in tcfps]
    fps = [t[1] for t in tcfps]

    if any([fps1 != fps[0] for fps1 in fps]):
        logging.error("All of the frame rates must be the same. Currently they are {}".format(fps))
        raise ValueError("All of the frame rates must be the same!")
    
    tc = sorted(tc)
    fps = fps[0]

    for f1, tc1 in zip(args.files[0], tc):
        off = tc1 - tc[0]
        offframes = off.total_seconds() * fps
        print("{}: {:%Y-%m-%d %H:%M:%S} + {}ms. Offset = {}ms ({:.3f} frames)"\
              .format(os.path.basename(f1), tc1, tc1.microsecond/1000,
                      off.total_seconds() * 1000, offframes))
    

if __name__ == '__main__':
    main()
