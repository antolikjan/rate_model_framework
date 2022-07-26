"""
This file contains analysis protocols.
"""
import os
import random
from typing import Dict, List

import matplotlib.gridspec as gridspec
import numpy
import numpy as np
from imagen import SineGrating
import pylab

from rate_model import NoTimeconstantSheet, InputSheet
from skin_parser.logger import setup_main_logger

TIMESTAMP_POSITION = 2
FIRST_TAXEL_POSITION = 3
FILEPATH = f"{os.getcwd()}\\data.log"

class InputSheetSkin(NoTimeconstantSheet):
#class InputSheetSkin():
    """
    Skin sheet.
    """

    skin_data: Dict[str, List[str]]

    def __init__(
        self, filepath,
    ):
        self.filepath = filepath
        self.skin_data = {}

    def load_data(self):
        with open(self.filepath) as data_file:
            for line in data_file.readlines():
                line = line.split()
                no_of_data_points = len(line[FIRST_TAXEL_POSITION:])
                self.skin_data[line[TIMESTAMP_POSITION]] = np.array([float(i) for i in line[FIRST_TAXEL_POSITION:]])
                #self.skin_data[line[TIMESTAMP_POSITION]] = np.expand_dims(self.skin_data[line[TIMESTAMP_POSITION]], axis=0)

    def load_data_in_2D(self):
        with open(self.filepath) as data_file:
            for line in data_file.readlines():
                line = line.split()
                self.skin_data[line[TIMESTAMP_POSITION]] = self.chunks(line[FIRST_TAXEL_POSITION:], 28)

    def set_activity(self, activity):
        self.activities = activity
        self.tmp_changed = True
        self.changed = True

    def update(self):
        self.tmp_changed = False


def fullfieldSineGratingOrientationTuningProtocol(
    model,
    retina,
    skin: InputSheet,
    sheets=None,
    num_orientation=100,
    num_phase=100,
    duration=0.04,
    frequency=2.4,
    scale=1.0,
    filename=None,
    plot=False,
    load=False,
    reset=False,
):
    """
    This analysis will present fullfield sine grating orientation tuning protocol to the model *model*.

    It will collect the final activation of neurons in sheets *sheets* (if None all sheets in the model are analyzed) and compute orientation tuning preference and
    selectivity for the neurons in these sheets.

    If filename is not None it will save the collected data to file, filename
    """
    logger = setup_main_logger()
    logger.info(f"skin = {skin}")
    skin = InputSheetSkin(FILEPATH)
    skin.load_data()
    skin = list(skin.skin_data.values())
    #logger.info(f"defskin = {skin}")

    if not load:
        responses = {}

        if sheets == None:
            sheets = model.sheets

        # initialize the
        for s in sheets:
            responses[s.name] = numpy.zeros(
                (num_orientation, num_phase, (s.unit_diameter) ** 2)
            )

        # present the stimulation protocol and collect data
        for i in range(num_orientation):
            for j in range(num_phase):
                if reset:
                    for s in model.sheets:
                        s.reset()

                # stim = SineGrating(
                #     orientation=numpy.pi / num_orientation * i,
                #     phase=numpy.pi * 2 / num_phase * j,
                #     xdensity=retina.unit_diameter,
                #     ydensity=retina.unit_diameter,
                #     frequency=frequency,
                #     scale=scale,
                # )()
                stim_number = random.randint(0, len(skin))
                #logger.info(f"stimnumber = {stim_number}")
                stim = skin[stim_number]
                #logger.info(f"stim = {stim}")
                retina.set_activity(stim)
                #logger.info(f"retina activity {retina.get_activity(model.dt)}")
                model.run(duration)
                for sheet in sheets:
                    logger.info(f"sheets {sheets}")
                    responses[sheet.name][i, j, :] = (
                        retina.get_activity(model.dt).copy().ravel()
                    )
                    # responses[sheet.name][i, j, :] = (
                    #     sheet.get_activity(model.dt).copy().ravel()
                    # )
                    logger.info(f"responses {str(responses)}")

            #logger.info(f"responses {str(responses)}")

        if filename != None:
            import pickle

            f = open(filename + "_resp.pickle", "wb")
            pickle.dump(responses, f)
            f.close()
    else:
        import pickle

        f = open(filename + "_resp.pickle", "rb")
        responses = pickle.load(f)
        f.close()

    # lets calculate the orientation preference and selectivity
    angles = [2 * numpy.pi / num_orientation * i for i in range(num_orientation)]
    angles_as_complex_numbers = numpy.cos(angles) + numpy.sin(angles) * 1j

    orientation_preference_maps = {}
    orientation_selectivity_maps = {}

    pylab.figure()
    for k in responses.keys():
        # first let's select the response to each orientation as the maximum response across phases
        resp = numpy.max(responses[k], axis=1)
        logger.info(f"resp {resp}")

        sheet_size = numpy.sqrt(len(resp[0]))
        for i in range(len(angles)):
            if i > 0:
                pylab.subplot(1, len(angles), i)

                pylab.imshow(
                    numpy.reshape(resp[i], (int(sheet_size), int(sheet_size))),
                    interpolation="none",
                    cmap="gray",
                )
                pylab.colorbar()

        # calculate selectivity and preference as the angle and magnitude of the mean of the responses projected into polar coordinates based on the orientation angle
        # to which they were collected
        polar_resp = resp * numpy.array(angles_as_complex_numbers)[:, numpy.newaxis]
        polar_mean_resp = numpy.mean(polar_resp, axis=0)

        orientation_preference_maps[k] = (
            numpy.angle(polar_mean_resp) + 4 * numpy.pi
        ) % (numpy.pi * 2)
        orientation_selectivity_maps[k] = numpy.absolute(polar_mean_resp)

    pylab.savefig(filename + "_or_resps.png", dpi=200)

    pylab.figure()
    # make a plot of the maps if asked for
    if plot:
        gs = gridspec.GridSpec(2, len(responses.keys()))
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.1, wspace=0.1)

        for i, s in enumerate(responses.keys()):
            sheet_size = numpy.sqrt(len(orientation_preference_maps[k]))
            pylab.subplot(gs[0, i])
            im = pylab.imshow(
                numpy.resize(orientation_preference_maps[k], (int(sheet_size), int(sheet_size))),
                vmin=0,
                vmax=2 * numpy.pi,
                interpolation="nearest",
                cmap="hsv",
            )
            pylab.axis("off")
            pylab.colorbar(im, fraction=0.046, pad=0.04)

            pylab.subplot(gs[1, i])
            im = pylab.imshow(
                numpy.resize(orientation_selectivity_maps[k], (int(sheet_size), int(sheet_size))),
                interpolation="nearest",
                cmap="gray",
            )
            pylab.axis("off")
            pylab.colorbar(im, fraction=0.046, pad=0.04)

        pylab.savefig(filename + "_maps.png", dpi=200)

    return orientation_preference_maps, orientation_selectivity_maps
