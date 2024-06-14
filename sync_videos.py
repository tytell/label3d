import os, sys
import argparse
import logging
import json
import subprocess
import numpy as np
from datetime import datetime, time, timedelta
import re
from functools import total_ordering

if sys.platform == 'darwin':
    FFPROBE = '/opt/homebrew/bin/ffprobe'
    FFMPEG = '/opt/homebrew/bin/ffmpeg'
elif sys.platform == 'win32':
    FFPROBE = ''
    FFMPEG = ''
else:
    raise(OSError("Could not find ffprobe or ffmpeg"))

if not os.path.exists(FFPROBE):
    raise(OSError("Could not find ffprobe at path {}".format(FFPROBE)))
if not os.path.exists(FFMPEG):
    raise(OSError("Could not find ffmpeg at path {}".format(FFMPEG)))

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
    parser.add_argument('-y', '--overwrite', help="Automatically overwrite existing output files",
                        default=False, action='store_true')
    parser.add_argument('-c:v', '--codec_video', help="Video codec for reencoding",
                        default="libx265")
    parser.add_argument('-c:a', '--codec_audio', help="Audio codec for reencoding",
                        default="aac")
    parser.add_argument('-preset', '--preset', help="FFmpeg encoding preset",
                        default="slow")
    parser.add_argument('--nframes', help="Number of frames to encode",
                        default="span")
    parser.add_argument('-n', '--dry_run', help="Just display information but don't do the encoding",
                        default=False, action='store_true')
    return parser

@total_ordering
class VideoTimeData:
    def __init__(self, filename):
        self.filename = filename
        self.file_data = self._get_file_data(filename)    

        self.video_stream = self._get_video_stream()
        self._parse_timecode()

        self.nframes = int(self.video_stream['nb_frames'])
        self.bitrate = float(self.video_stream['bit_rate'])

        self.audio_stream = self._get_audio_stream()
        if self.audio_stream is not None:
            self.audiobitrate = float(self.audio_stream['bit_rate'])

    def _get_file_data(self, filename):
        cmd = [FFPROBE, '-print_format', 'json',
            '-show_format', '-show_streams',
                filename]
        logging.debug("Command: {}".format(' '.join(cmd)))
        r = subprocess.run(cmd, capture_output=True)

        dat = json.loads(r.stdout)
        return(dat)

    def _get_video_stream(self):
        for s in self.file_data['streams']:
            if s['codec_type'] == 'video':
                break
        else:
            raise KeyError("No video stream found")
        
        return s
        
    def _get_audio_stream(self):
        for s in self.file_data['streams']:
            if s['codec_type'] == 'audio':
                break
        else:
            return None
        
        return s

    def _parse_timecode(self):
        creation_time = self.video_stream['tags']['creation_time']
        frame_rate = self.video_stream['avg_frame_rate']
        timecode = self.video_stream['tags']['timecode']
        
        creation_time = datetime.fromisoformat(str(creation_time).replace('Z', '+00:00'))

        m = re.fullmatch('(\d+)/(\d+)', frame_rate)
        if m is None:
            raise ValueError("Could not parse frame rate {}".format(frame_rate))
        frame_rate = float(m.group(1)) / float(m.group(2))

        m = re.fullmatch('(\d{2})[:;.](\d{2})[:;.](\d{2})[:;.](\d+)', timecode)
        if m is None:
            raise ValueError("Could not parse timecode {}".format(timecode))

        self.timecodestr = timecode
        hmsf = [int(g) for g in m.groups()]

        us = int(float(hmsf[3]) / frame_rate * 1e6)
        timecode = time(hmsf[0], hmsf[1], hmsf[2], us)

        timecode = datetime(creation_time.year, creation_time.month, creation_time.day,
                            timecode.hour, timecode.minute, timecode.second, timecode.microsecond)

        self.timecode = timecode
        self.frame_rate = frame_rate

    def __lt__(self, other):
        return (self.timecode < other.timecode)
    def __gt__(self, other):
        return (self.timecode > other.timecode)
    def __le__(self, other):
        return (self.timecode <= other.timecode)
    def __ge__(self, other):
        return (self.timecode >= other.timecode)
    def __eq__(self, other):
        return (self.timecode == other.timecode)
    def __ne__(self, other):
        return (self.timecode != other.timecode)

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = build_parser()
    args = parser.parse_args()

    timedata = [VideoTimeData(f) for f in args.files[0]]
    timedata = sorted(timedata, reverse=True)

    tc = np.array([td.timecode for td in timedata])
    fps = np.array([td.frame_rate for td in timedata])
    nfr = np.array([td.nframes for td in timedata])

    if np.any(fps != fps[0]):
        logging.error("All of the frame rates must be the same. Currently they are {}".format(fps))
        raise ValueError("All of the frame rates must be the same!")
    
    fps = fps[0]

    startfr = np.array([(tc1 - tc[0]).total_seconds() * fps for tc1 in tc])
    endfr = np.array([nfr1 + startfr1 for nfr1, startfr1 in zip(nfr, startfr)])

    minframes = np.min(endfr)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for f1, td1, tc1, endfr1 in zip(args.files[0], timedata, tc, endfr):
        off = tc1 - tc[0]
        offframes = off.total_seconds() * fps

        offend = minframes - endfr1

        print("{}: {:%Y-%m-%d %H:%M:%S} + {}ms. Offset = {}ms ({:.3f}fr). End offset = {:.3f}fr"\
              .format(os.path.basename(f1), tc1, tc1.microsecond/1000,
                      off.total_seconds() * 1000, offframes,
                      offend))

        if ((off.total_seconds() != 0) or (offend != 0)):
            nm, ext = os.path.splitext(f1)
            outf1 = ''.join([nm, args.output_suffix, ext])
            if outf1 == f1:
                logging.error("Cannot overwrite the input file {}".format(f1))
                raise ValueError("Cannot overwrite the input file")
            elif os.path.exists(outf1) and not args.overwrite:
                logging.warning("Output file {} exists. Stopping".format(outf1))
                break

            cmd = [FFMPEG]

            if args.overwrite:
                cmd.append('-y')
            else:
                cmd.append('-n')

            off = -off + timedelta(seconds=1/fps)
            hours, rem = divmod(off.seconds, 3600)
            minutes, seconds = divmod(rem, 60)
            # If you want to take into account fractions of a second
            msec = int(off.microseconds / 1e3)

            cmd.extend(['-ss', '{}:{}:{}.{:03d}'.format(hours, minutes, seconds, msec),
                   '-i', f1, 
                   '-c:v', args.codec_video,
                   '-preset', args.preset,
                   '-b:v', str(td1.bitrate)])
            
            if args.nframes == 'span':
                cmd.extend(['-vframes', str(int(minframes))])
            elif args.nframes == 'all':
                pass
            else:
                try:
                    nfr = int(args.nframes)
                    cmd.extend(['-vframes', args.nframes])
                except ValueError:
                    logging.error("Cannot parse number of frames option {}".format(args.nframes))
                    raise ValueError("Cannot parse number of frames option {}".format(args.nframes))
            
            if args.codec_video == "libx265":
                cmd.extend(['-vtag', 'hvc1'])

            if td1.audio_stream is not None:
                cmd.extend(['-c:a', args.codec_audio,
                            '-b:a', str(td1.audiobitrate)])
            else:
                cmd.extend(['-an'])
            
            cmd.append(outf1)

            logging.debug("Command: {}".format(' '.join(cmd)))
            if not args.dry_run:
                r = subprocess.run(cmd)


if __name__ == '__main__':
    main()
