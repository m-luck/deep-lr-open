import sys

import progressbar as pb


def get_adaptive_progressbar(max_val: int) -> pb.ProgressBar:
    if sys.stdout.isatty():
        ProgressBar = pb.ProgressBar
    else:
        ProgressBar = pb.NullBar

    print("max val {}, type {}".format(max_val, type(max_val)))

    widgets = [pb.Percentage(), ' ', pb.Bar(), ' ', pb.AdaptiveETA(), ', ', pb.Timer()]
    bar = ProgressBar(max_val, widgets)
    return bar
