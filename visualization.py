import pylab
import matplotlib.gridspec as gridspec
import numpy

from logger import setup_main_logger


def display_model_state(model, filename: str = None) -> None:
    logger = setup_main_logger()
    # find the longest number of projections
    max_num_in_projections = 0
    for s in model.sheets:
        max_num_in_projections = max(max_num_in_projections, len(s.in_projections))
    pylab.subplot(max_num_in_projections + 1, len(model.sheets), 1)

    logger.info(f"model sheets {str(model.sheets)}")

    for i in range(len(model.sheets)):
        logger.info(f"max_num_in_projections + 1, len(model.sheets), i + 1 {max_num_in_projections + 1, len(model.sheets), i + 1}")
        pylab.subplot(max_num_in_projections + 1, len(model.sheets), i + 1)
        pylab.title(model.sheets[i].name)
        logger.info(f"name {model.sheets[i].name}")
        logger.info(f"model.sheets get activity {str(model.sheets[i].get_activity(model.dt))}")
        im = pylab.imshow(
            model.sheets[i].get_activity(model.dt), cmap="gray", interpolation="nearest"
        )
        pylab.colorbar(im, fraction=0.046, pad=0.04)

        for j in range(len(model.sheets[i].in_projections)):
            pylab.subplot(
                max_num_in_projections + 1,
                len(model.sheets),
                len(model.sheets) * (j + 1) + i + 1,
            )
            pylab.title(model.sheets[i].in_projections[j].name)
            im = pylab.imshow(
                model.sheets[i].in_projections[j].activity,
                cmap="gray",
                interpolation="nearest",
            )
            pylab.colorbar(im, fraction=0.046, pad=0.04)
            pylab.axis("off")

    if filename is not None:
        pylab.savefig(filename, dpi=200)


def plot_projection(
    projection,
    downsample: float = 0.2,
    filename: str = None,
) -> None:
    """
    Plots the connection fields in the projection.
    Only *downsample* fraction (evenly spaced) of the connection fields will be shown
    """

    size = int(numpy.floor(numpy.floor(projection.target.unit_diameter * downsample)))

    step = int(numpy.round(1 / downsample))

    gs = gridspec.GridSpec(int(size), int(size))
    gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.1)

    for i in range(size):
        for j in range(size):
            pylab.subplot(gs[int(i), int(j)])
            pylab.imshow(
                projection.get_cf(i * step, j * step),
                interpolation="nearest",
                cmap="gray",
            )
            pylab.axis("off")

    if filename is not None:
        pylab.savefig(filename, dpi=200)
