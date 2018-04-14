import mss
import mss.tools


sct = mss.mss()
monitor = {
    'top': 173,
    'left': 428,
    'width': 508,
    'height': 442,
}
output = 'sct-mon.png'
sct_img = sct.grab(monitor)
mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
