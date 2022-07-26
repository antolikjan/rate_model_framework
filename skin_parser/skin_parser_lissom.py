from typing import Dict, List
import numpy as np
import os
from logger import setup_main_logger
from rate_model import NoTimeconstantSheet

TIMESTAMP_POSITION = 2
FIRST_TAXEL_POSITION = 3
FILEPATH = f"{os.getcwd()}\\data.log"

np.set_printoptions(precision=20)

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

    @staticmethod
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        new_lst = []
        for i in range(0, len(lst), n):
            helper_list = lst[i:i + n]
            floats = [float(x) for x in helper_list]
            while len(floats) != n:
                floats.append(0.0)
            new_lst.append(np.array(floats))
        return new_lst

s = InputSheetSkin(FILEPATH)
s.load_data()

import time

import imagen.random
import numbergen
from imagen.transferfn import DivisiveNormalizeL1

#from analysis import *
from projections import *
from rate_model import *
from visualization import *
import projections
from skin_parser_analysis import fullfieldSineGratingOrientationTuningProtocol


if __name__ == "__main__":
    logger = setup_main_logger()

    logger.info("got logger")

    skin = InputSheetSkin(FILEPATH)
    skin.load_data()
    #logger.info(f"skinity skin {list(skin.skin_data.values())}")

    retina = InputSheet("Retina", 2.4, 25, None)
    lgn_on = NoTimeconstantSheet("LGN_ON", 1.6, 25, None)
    lgn_off = NoTimeconstantSheet("LGN_OFF", 1.6, 25, None)
    V1 = HomeostaticSheet(
        "V1", 1.0, 74, 0.002, init_threshold=0.04, mu=0.002, alpha=0.1, smoothing=0.91
    )

    lissom = Model([retina, V1], 0.001)


    run_for = 1
    t = time.time()
    with open("final_file.txt", "a") as my_file:
        for timepoint in range(len(skin.skin_data)):
          #  m = skin.skin_data.values()[timepoint].values()
            m = list(skin.skin_data.values())
            #logger.info(f"running {str(m[timepoint])}")
            retina.set_activity(m[timepoint])
            lissom.run(0.05)

            V1.applyHomeostaticThresh()

            my_file.write(str(V1.activities))

            print(
                timepoint,
                ":",
                "Expected time to run: ",
                ((time.time() - t) / (timepoint + 1)) * (run_for - timepoint),
                "s",
            )

            if timepoint == len(skin.skin_data) - 1:
                pass
                #pylab.figure()
                #display_model_state(lissom, filename="activity.png")
            lissom.reset()

        if False:
            import pickle

            f = open("lissom_long.pickle", "wb")
            pickle.dump(lissom, f)
            f.close()
        # print("fig1")
        # pylab.figure()
        # plot_projection(lgn_on_to_V1, filename="onProjection.png", downsample=0.5)
        #
        # print("fig2")
        # pylab.figure()
        # plot_projection(lgn_off_to_V1, filename="offProjection.png", downsample=0.5)

        print("fig3")
        pylab.figure()
        fullfieldSineGratingOrientationTuningProtocol(
            lissom,
            retina,
            skin=list(skin.skin_data.values()),
            sheets=[V1],
            num_orientation=8,
            num_phase=10,
            duration=0.02,
            frequency=4.0,
            filename="freq=4",
            reset=True,
            plot=True,
            load=False,
        )
        # print("fig4")
        # pylab.figure()
        # fullfield_sine_grating_orientation_tuning_protocol(
        #     lissom,
        #     retina,
        #     sheets=[V1],
        #     num_orientation=8,
        #     num_phase=10,
        #     duration=0.02,
        #     frequency=5.0,
        #     filename="freq=5",
        #     reset=True,
        #     plot=True,
        #     load=False,
        # )

        print("fig5")
        pylab.show()

    # if True:
    #    pylab.figure()
    #    plot_projection(lgn_on_to_V1)