
def keep_only_digits(txt):
    return ''.join(filter(lambda c: c.isdigit(), txt))


def secs_to_hours(secs):
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)
