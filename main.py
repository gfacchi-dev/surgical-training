from detector import Detector
import argparse


def main(args):
    detector = Detector()
    detector.detect_and_draw(args.video_path, args.interactive, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("-i", "--interactive", action="store_true", help="Show the output in a window", default=False)
    parser.add_argument("-s", "--save_path", type=str, required=False, help="Path to save the output frames, with scissors and hands detected")
    args = parser.parse_args()
    main(args)
