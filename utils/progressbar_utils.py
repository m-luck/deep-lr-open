import sys

import progressbar as pb


def get_adaptive_progressbar(max_val: int) -> pb.ProgressBar:
    if sys.stdout.isatty():
        ProgressBar = pb.ProgressBar
    else:
        ProgressBar = pb.NullBar

    widgets = [pb.Percentage(), ' ', pb.Bar(), ' ', pb.AdaptiveETA(), ', ', pb.Timer()]
    bar = ProgressBar(max_value=max_val, widgets=widgets, term_width=80)
    return bar
