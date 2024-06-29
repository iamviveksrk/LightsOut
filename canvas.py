from ipycanvas import Canvas, hold_canvas

width, height = 500, 500

canvas = Canvas(width=width, height=height, sync_image_data=True)
canvas.fill_style = "#DA7297"
canvas.fill_rect(int(0.95*width), 0, width/20, height)
canvas.fill_style = "#667BC6"

drawing = False
marker_size = 25

def myround(x, base=marker_size):
    return base * round((x-base//2)/base)

def on_mouse_down(x, y):
    global drawing
    drawing = True

    if x >= int(0.95*width):
        canvas.clear_rect(0, 0, int(0.95*width), height)

def on_mouse_move(x, y):
    global drawing
    if not drawing:
        return

    with hold_canvas():
        canvas.fill_rect(myround(x), myround(y), marker_size)

def on_mouse_up(x, y):
    global drawing
    drawing = False


canvas.on_mouse_down(on_mouse_down)
canvas.on_mouse_move(on_mouse_move)
canvas.on_mouse_up(on_mouse_up)