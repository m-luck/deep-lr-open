import sys

import progressbar as pb


def get_adaptive_progressbar(max_val: int) -> pb.ProgressBar:
    if sys.stdout.isatty():  # real terminal
        ProgressBar = pb.ProgressBar
    else:
        ProgressBar = pb.NullBar

    widgets = [pb.Percentage(), ' ', pb.Bar(), ' ', pb.AdaptiveETA(), ', ', pb.Timer()]
    bar = ProgressBar(max_val, widgets)
    return bar
