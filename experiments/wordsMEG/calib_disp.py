"""
==========================
Display images for MEG
==========================
"""
import numpy as np
from expyfun import visual, ExperimentController
from expyfun.io import write_hdf5
import time
from PIL import Image
import os
from os import path as op
import glob

s = .5  # Image scale

bgcolor = [.5, .5, .5]

# Start instance of the experiment controller
with ExperimentController('ShowImages', full_screen=True, version='dev') as ec:
    #write_hdf5(op.splitext(ec.data_fname)[0] + '_trials.hdf5',
    #           dict(imorder_shuf=imorder_shuf,
    #                imtype_shuf=imtype_shuf))
    fr = 1/ec.estimate_screen_fs()  # Estimate frame rate
    adj = fr/2  # Adjustment factor for accurate flip
    # Wait to fill the screen
    ec.set_visible(False)
    # Set the background color to gray
    ec.set_background_color(bgcolor)


    # do the drawing, then flip
    ec.set_visible(True)
    frametimes = []
    buttons = []
    ec.listen_presses()
    last_flip = -1

    # Create a fixation dot
    fix = visual.FixationDot(ec, colors=('k', 'k'))
    fix.set_radius(4, 0, 'pix')
    fix.draw()

    # Show images
    count = 0
    gray_level = np.linspace(0,1,21)
    ec.listen_presses()
    key_pressed = False
    for trial in np.arange(0,20):
        bgcolor = [gray_level[trial], gray_level[trial], gray_level[trial]]

        ec.set_background_color(bgcolor)


        # The image is flipped ISI milliseconds after the blank
        last_flip = ec.flip(last_flip-adj)

        ec.check_force_quit()
        ec.wait_one_press(5)
    
                    