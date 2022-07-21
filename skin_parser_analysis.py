"""
This file contains analysis protocols.
"""
import matplotlib.gridspec as gridspec
import numpy
import pylab
import pickle

from imagen import SineGrating
from typing import List

from logger import setup_main_logger
from rate_model import Model, InputSheet, HomeostaticSheet


def fullfield_sine_grating_orientation_tuning_protocol(
    model: Model,
    retina: InputSheet,
    skin: InputSheet,
    sheets: List[HomeostaticSheet] = None,
    num_orientation: int = 8,
    num_phase: int = 10,
    duration: float = 0.04,
    frequency: float = 2.4,
    scale: float = 1.0,
    filename: str = None,
    plot: bool = False,
    load: bool = False,
    reset: bool = False,
):
    """
    This analysis will present fullfield sine grating orientation tuning protocol to the model *model*.

    It will collect the final activation of neurons in sheets *sheets* (if None all sheets in the model are analyzed)
    and compute orientation tuning preference and
    selectivity for the neurons in these sheets.

    If filename is not None it will save the collected data to file, filename
    """
    logger = setup_main_logger()
    if not load:
        responses = {}

        if sheets is None:
            sheets = model.sheets

        # initialize the
        for sheet in sheets:
            responses[sheet.name] = numpy.zeros(
                (num_orientation, num_phase, sheet.unit_diameter ** 2)
            )

        # present the stimulation protocol and collect data
        for i in range(num_orientation):
            for j in range(num_phase):
                if reset:
                    for sheet in model.sheets:
                        sheet.reset()

                stim = SineGrating(
                    orientation=numpy.pi / num_orientation * i,
                    phase=numpy.pi * 2 / num_phase * j,
                    xdensity=retina.unit_diameter,
                    ydensity=retina.unit_diameter,
                    frequency=frequency,
                    scale=scale,
                )()
                retina.set_activity(stim)
                model.run(duration)
                for sheet in sheets:
                    responses[sheet.name][i, j, :] = (
                        sheet.get_activity(model.dt).copy().ravel()
                    )

        logger.info(str(responses))

    # lets calculate the orientation preference and selectivity
    angles = [2 * numpy.pi / num_orientation * i for i in range(num_orientation)]
    angles_as_complex_numbers = numpy.cos(angles) + numpy.sin(angles) * 1j

    orientation_preference_maps = {}
    orientation_selectivity_maps = {}

    pylab.figure()
    for sheet in responses.keys():
        # first let's select the response to each orientation as the maximum response across phases
        response = numpy.max(responses[sheet], axis=1)

        sheet_size = numpy.sqrt(len(response[0]))
        for i in range(len(angles)):
            if i > 0:
                pylab.subplot(1, len(angles), i)
                pylab.imshow(
                    numpy.reshape(response[i], (int(sheet_size), int(sheet_size))),
                    interpolation="none",
                    cmap="gray",
                )
                pylab.colorbar()

        # calculate selectivity and preference as the angle and magnitude of the mean of the responses projected
        # into polar coordinates based on the orientation angle
        # to which they were collected
        polar_resp = response * numpy.array(angles_as_complex_numbers)[:, numpy.newaxis]
        polar_mean_resp = numpy.mean(polar_resp, axis=0)

        orientation_preference_maps[sheet] = (
            numpy.angle(polar_mean_resp) + 4 * numpy.pi
        ) % (numpy.pi * 2)
        orientation_selectivity_maps[sheet] = numpy.absolute(polar_mean_resp)

    pylab.savefig(filename + "_or_resps.png", dpi=200)

    pylab.figure()
    # make a plot of the maps if asked for
    if plot:
        gs = gridspec.GridSpec(2, len(responses.keys()))
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.1)

        for i, sheet in enumerate(responses.keys()):
            sheet_size = numpy.sqrt(len(orientation_preference_maps[sheet]))
            pylab.subplot(gs[0, i])
            im = pylab.imshow(
                numpy.resize(
                    orientation_preference_maps[sheet],
                    (int(sheet_size), int(sheet_size)),
                ),
                vmin=0,
                vmax=2 * numpy.pi,
                interpolation="nearest",
                cmap="hsv",
            )
            pylab.axis("off")
            pylab.colorbar(im, fraction=0.046, pad=0.04)

            pylab.subplot(gs[1, i])
            im = pylab.imshow(
                numpy.resize(
                    orientation_selectivity_maps[sheet],
                    (int(sheet_size), int(sheet_size)),
                ),
                interpolation="nearest",
                cmap="gray",
            )
            pylab.axis("off")
            pylab.colorbar(im, fraction=0.046, pad=0.04)

        pylab.savefig(filename + "_maps.png", dpi=200)

    return orientation_preference_maps, orientation_selectivity_maps
