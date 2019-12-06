import progressbar as pb


def get_adaptive_progressbar(max_val: int) -> pb.ProgressBar:
    widgets = [pb.Percentage(), ' ', pb.Bar(), ' ', pb.AdaptiveETA(), ', ', pb.Timer()]
    bar = pb.ProgressBar(max_val, widgets)
    return bar
