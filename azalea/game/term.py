

purple, blue, gray, red, green = "#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#2ecc71"
white = '#ffffff'


class Color:
    def __init__(self, code=None):
        self.r = self.g = self.b = 0
        if code:
            rgb = int(code[1:], 16)
            self.r = (rgb >> 16) & 0xFF
            self.g = (rgb >> 8) & 0xFF
            self.b = rgb & 0xFF

    def scale(self, level, scale_min=0.0):
        level = scale_min + max(0.0, level - scale_min) / (1.0 - scale_min)

        def rnd(x):
            return min(max(int(round(x)), 0), 255)
        res = Color()
        res.r = rnd(level * self.r)
        res.g = rnd(level * self.g)
        res.b = rnd(level * self.b)
        return res


def termcolor256(txt, fg=None, bg=None):
    ## 256 colors ##
    # \x1b[38;5;#m foreground, # = 0 - 255
    # \x1b[48;5;#m background, # = 0 - 255
    tc = ''
    if fg:
        tc += '\x1b[38;5;{}m'.format(fg)
    if bg:
        tc += '\x1b[48;5;{}m'.format(bg)
    if tc:
        return tc + txt + '\x1b[0m'
    return txt


def termcolor(txt, fg=None, bg=None):
    ## True Color ##
    # \x1b[38;2;r;g;bm r = red, g = green, b = blue foreground
    # \x1b[48;2;r;g;bm r = red, g = green, b = blue background
    tc = ''
    if fg:
        tc += '\x1b[38;2;{};{};{}m'.format(fg.r, fg.g, fg.b)
    if bg:
        tc += '\x1b[48;2;{};{};{}m'.format(bg.r, bg.g, bg.b)
    if tc:
        return tc + txt + '\x1b[0m'
    return txt
