import matplotlib.pyplot as plt

def show_canvas_part(pixels, ax=None):
    '''
    Plots 2D pixels array

    '''
    if ax == None:
        plt.figure(origin='upper')
        plt.imshow(pixels, origin='upper')
    else:
        ax.imshow(pixels, origin='upper')
