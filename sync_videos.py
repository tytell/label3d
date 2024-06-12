import os, sys
import argparse
import logging
import json
import subprocess
import numpy as np

if sys.platform == "darwin":
    FFPROBE = "/opt/homebrew/bin/ffprobe"
elif sys.platform == "win32":
    FFPROBE = ""
else:
    raise(OSError("Could not find ffprobe"))

if not os.path.exists(FFPROBE):
    raise(OSError("Could not find ffprobe at path {}".format(FFPROBE)))

def build_parser():
    parser = argparse.ArgumentParser(
                        prog='sync_videos',
                        description='Synchronize videos based on timecode and audio')
    
    parser.add_argument("-v", "--verbose", help="Show more details",
                        action="store_true")
    parser.add_argument("files", nargs="+", help="Input movie files to synchronize",
                        action="append")
    parser.add_argument("-od", "--output_dir", help="Directory to save the output files",
                        default=".")
    parser.add_argument("-s", "--output_suffix", help="Suffix to add to synchronized output videos",
                        default="-sync")
    
    return parser
    
def get_file_data(filename):
    cmd = [FFPROBE, "-print_format", "json",
           "-show_format", "-show_streams",
            filename]
    logging.debug("Command: {}".format(' '.join(cmd)))
    r = subprocess.run(cmd, capture_output=True)

    dat = json.loads(r.stdout)
    return(dat)

def get_timecode(dat):
    for s in dat['streams']:
        if s['codec_tag_string'] == "tmcd":
            timecode = s['tags']['timecode']
            break
    else:
        raise KeyError("No timecode stream found")
    
    return timecode

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = build_parser()
    args = parser.parse_args()

    dat = [get_file_data(f) for f in args.files[0]]
    for f, d in zip(args.files[0], dat):
        tc = get_timecode(d)
        print("{}: {}".format(os.path.basename(f), tc))
        
if __name__ == "__main__":
    main()
