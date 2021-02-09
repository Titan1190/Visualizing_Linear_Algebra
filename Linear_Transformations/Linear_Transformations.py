import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy

print('matplotlib: {}'.format(mpl.__version__))

# Generates list from range (-4 --> 4) with 9 values
xvals = np.linspace(-4,4,9)
yvals = np.linspace(-3, 3, 7)
xygrid = np.column_stack([[x,y] for x in xvals for y in yvals])


# Apply Linear Transform
#a = np.column_stack([[2,1],[-1,1]])
#uvgrid = np.dot(a, xygrid)


# Colors every point in grid by its xy position
def colorizer(x,y):
    r = min(1, 1-y/3)
    g = min(1, 1 + y / 3)
    b = 1/4 + x/16
    return (r,g,b)

colors = list(map(colorizer, xygrid[0], xygrid[1]))

'''
plt.figure(figsize = (4,4), facecolor = "w")
plt.scatter(xygrid[0], xygrid[1], s=144, c=colors, edgecolor = "none")

# Sets axis limits

plt.grid(True)
plt.axis("equal")
plt.title("Original grid in x-y space")

plt.show()



# Plot transformed grid points
plt.figure(figsize = (4,4), facecolor = "w")
plt.scatter(uvgrid[0], uvgrid[1], s = 144, c = colors, edgecolor = "none")
plt.grid(True)
plt.axis("equal")
plt.title("Transformed grid in u-v space")

plt.show()
'''


def intermediate_transforms(a, points, nsteps = 30):
    # Makes nsteps + 1 grids of zeros in the shape of points
    transgrid = np.zeros((nsteps + 1,) + np.shape(points))
    for j in range(nsteps + 1):
        # Ai = I + j/n * (A - I) --> transgrid[i] == Linear transformation of intermediate matrix & points in question
        intermediate = np.eye(2) + (j / nsteps) * (a - np.eye(2))
        transgrid[j] = np.dot(intermediate, points)

    return transgrid



def make_plots(transarray, color, numTransform, maxval, outdir= "png-frames", figuresize = (4,4), figuredpi=150):
    nsteps = transarray.shape[0]
    ndigits = len(str(nsteps))
    #maxval = np.abs(transarray.max())

    import os

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Saves images and compiles into directory

    plt.ioff()
    fig = plt.figure(figsize=figuresize, facecolor = "w")
    for j in range(nsteps):
        plt.cla()
        plt.scatter(transarray[j,0], transarray[j,1], s=36, c=color, edgecolor = "none")
        plt.xlim(1.1*np.array([-maxval, maxval]))
        plt.ylim(1.1*np.array([-maxval, maxval]))
        plt.grid(True)
        plt.draw()
        # save as png
        outfile = os.path.join(outdir, "frame-" + str(numTransform) + "-" + str(j+1).zfill(ndigits) + ".png")
        fig.savefig(outfile, dpi=figuredpi)
    plt.ion()


if __name__ == "__main__":

    theta = np.pi / 6
    num_transformations = int(input("Please input the number of Linear Transformations you'd like to perform: "))
    transformations = [np.eye(2) for x in range(num_transformations)]
    const = num_transformations

    while num_transformations > 0:
        for x in range(2):
            for y in range(2):
                transformations[num_transformations - 1][x][y] = float(input(f"Please input value ({x},{y}) in A{const - num_transformations} > "))

        num_transformations -= 1

    # Find max values
    maxval = 0
    steps = 30
    tmp_xygrid = copy.deepcopy(xygrid)

    for j in range(len(transformations)):
        transform = intermediate_transforms(transformations[-(j + 1)], tmp_xygrid, nsteps = steps)
        tmp = np.abs(transform.max())
        if tmp > maxval:
            maxval = tmp
        tmp_xygrid = transform[-1]


    for i in range(len(transformations)):
        transform = intermediate_transforms(transformations[-(i + 1)], xygrid, nsteps = steps)
        make_plots(transform, colors, i, maxval, outdir = "tmp")
        xygrid = transform[-1]

    from subprocess import call
    call("cd tmp && magick convert -delay 10 frame-*.png ../animation.gif && explorer .", shell=True)
