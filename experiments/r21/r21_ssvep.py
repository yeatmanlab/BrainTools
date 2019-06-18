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

# background color
testing = False
test_trig = [15]
if testing:
    bgcolor = [0., 0., 0., 1.]
else:
    bgcolor = [0.5, 0.5, 0.5, 1]

# Paths to images
#basedir = '/home/jyeatman/projects/MEG/images/'
basedir = os.path.join('C:\\Users\\neuromag\\Desktop\\jason\\wordStim')
if not os.path.isdir(basedir):
    basedir = os.path.join('/mnt/diskArray/projects/MEG/wordStim')

""" Words, False fonts (Korean), Faces, Objects """
imagedirs = ['falsefont', 'word_c254_p0']
stim_time = 20 # 40 s seq
base_rate = 6 # 6 reps per second
odd_rate = 3 # every odd_rate images. For example, every 3 images oddball appears: 2Hz (6/3)
n_images = stim_time * base_rate

init_time = 2
end_time = 2 # These are padding in the beginning and end
pad_time = init_time + end_time

#nimages = [30, 30, 30, 30]  # number of images in each category
#if len(nimages) == 1: # Does nothing....
#    nimages = np.repeat(nimages, len(imagedirs))
#n_totalimages = sum(nimages)
# ISIs to be used. Must divide evenly into nimages
#isis = np.arange(1., 1.51, 0.1) #np.arange(.62, .84, .02)
#imduration = 0.8  # Image duration 800 ms
s = .5  # Image scale

# Create a vector of ISIs in a random order. One ISI for each image
rng = np.random.RandomState(int(time.time()))
#ISI = np.tile(isis, int(np.ceil(n_totalimages/len(isis)))+1)
#rng.shuffle(ISI)
#ISI = ISI[:(n_totalimages)]

total_time = stim_time + pad_time

n_flickers = int(total_time/2) + 1 # 
n_target = int(stim_time / 5)

fix_seq = np.zeros(n_flickers) 
fix_seq[:n_target] = 1
rng.shuffle(fix_seq)
for i in range(0,len(fix_seq)-4):
    if (fix_seq[i] + fix_seq[i+1]) == 2:
        fix_seq[i+1] = 0
        if fix_seq[i+3] == 0:
            fix_seq[i+3] = 1
        elif fix_seq[i+4] == 0:
            fix_seq[i+4] = 1
            
# Creat a vector of dot colors for each ISI
c = ['r', 'b', 'y']
k = 0
m = 0
fix_color = []
for i in range(0,len(fix_seq)): 
    if fix_seq[i] == 1:
        fix_color.append('g')
    else:
        fix_color.append(c[k])
        k += 1
        k = np.mod(k,3)
    if k == 0:
        rng.shuffle(c)
        while fix_color[i] == c[0]:
            rng.shuffle(c)

# Build the path structure to each image in each image directory. Each image
# category is an entry into the list. The categories are in sequential order 
# matching imorder, but the images within each category are random

# Temporary variable with image names in order
tmp = sorted(glob.glob(os.path.join(basedir, imagedirs[0], '*')))
# Randomly grab nimages from the list
rng.shuffle(tmp)
baselist = tmp[:n_images]

tmp = sorted(glob.glob(os.path.join(basedir, imagedirs[1], '*')))
rng.shuffle(tmp)
oddlist = tmp

templist = []
k = 0
temptype = []
for i in np.arange(0,len(baselist)): #len(baselist)
    if np.mod(i,odd_rate) == odd_rate-1:
        templist.append(oddlist[k])
        k += 1
        temptype.append('Oddball')
    else:
        templist.append(baselist[i])
        temptype.append('Base')
templist = templist[:n_images]
temptype = temptype[:n_images]
paddlist = baselist[len(baselist)-pad_time*base_rate:]
paddtype = []
for i in np.arange(0,len(paddlist)/2):
    paddtype.append('Base')
imagelist = np.concatenate((paddlist[:init_time*base_rate],templist,paddlist[len(paddlist)-init_time*base_rate:]))
imtype = np.concatenate((paddtype,temptype,paddtype))

# Start instance of the experiment controller
with ExperimentController('ShowImages', full_screen=True, version='dev') as ec:
    #write_hdf5(op.splitext(ec.data_fname)[0] + '_trials.hdf5',
    #           dict(imorder_shuf=imorder_shuf,
    #                imtype_shuf=imtype_shuf))
    
    realRR = ec.estimate_screen_fs(n_rep=20)
    ec.write_data_line('realRR:', realRR) # we should keep a record of that measurement
    print('realRR:', realRR)   # and it should be visible to the operator
    realRR = round(realRR)
    assert realRR == 60 # the assertion will throw an error if realRR is ever something other than 60

    fr = 1./realRR
    adj = fr/2  # Adjustment factor for accurate flip
    # Wait to fill the screen
    ec.set_visible(False)
    # Set the background color to gray
    ec.set_background_color(bgcolor)

    n_frames = int(round(total_time * realRR))
    img_frames = int(round(realRR/base_rate))
    
    frame_img = np.arange(0,n_frames,img_frames)
    blank_img = frame_img + int(img_frames/2)
    start_frame = (len(paddlist)/2)*img_frames
    end_frame = start_frame + len(templist)*img_frames
#    x = np.linspace(0,np.pi,img_frames)
#    multi_factor = np.sin(x)
    
    jitter = np.arange(0,realRR*0.2) # 0~200 ms jitter
    
    temp_flicker = np.linspace(0,n_frames,n_flickers) # Get temp_flicker frames: every .5 s
    delay = []
    for i in np.arange(0,len(temp_flicker)):
        rng.shuffle(jitter)
        delay.append(jitter[0])
    frame_flicker = temp_flicker + delay # 
    
    # load up the image stack. The images in img_buffer are in the sequential 
    # non-shuffled order
    img = []
    for im in imagelist:
        img_buffer = np.array(Image.open(im), np.uint8) / 255.
        if img_buffer.ndim == 2:
            img_buffer = np.tile(img_buffer[:, :, np.newaxis], [1, 1, 3])
        img.append(visual.RawImage(ec, img_buffer, scale=s))
        ec.check_force_quit()

    # make a blank image
    blank = visual.RawImage(ec, np.tile(bgcolor[0], np.multiply([s, s, 1], img_buffer.shape)))
    bright = visual.RawImage(ec, np.tile([1.], np.multiply([s, s, 1], img_buffer.shape)))
    bright2 = visual.RawImage(ec, np.tile([1.], np.multiply([s, s, 1], [50, 50, 3])), pos = [0.8, -0.8])
    # Calculate stimulus size
    d_pix = -np.diff(ec._convert_units([[3., 0.], [3., 0.]], 'deg', 'pix'), axis=-1)

    # do the drawing, then flip
    ec.set_visible(True)
    frametimes = []
    buttons = []
    ec.listen_presses()
    last_flip = -1

    # Create a fixation dot
    fix = visual.FixationDot(ec, colors=('k', 'k'))
    fix.set_radius(5, 0, 'pix')
    fix.set_radius(4, 1, 'pix')
    fix.draw()

    # Display instruction (7 seconds).
    # They will be different depending on the run number
    t = visual.Text(ec,text='Button press when the dot turns green.',pos=[0,.1],font_size=40,color='k')
    t.draw()
    ec.flip()
    ec.wait_secs(5.0)

    # Show images
    count = 0 # This is for testing...
    
    # Initial blank
    init_blanktime = 1.
    fix.set_colors(colors=('k', 'k'))
    blank.draw(), fix.draw()
    ec.write_data_line('dotcolorFix', 'k')
    last_flip = ec.flip()

    # The iterable 'trial' randomizes the order of everything since it is
    # drawn from imorder_shuf
    trial = 0
    b_trial = 0
    frame = 0
    flicker = 0
    imageframe = []
    trig = 0
    time2Flip = 0
    t0 = time.time()
    while frame < n_frames:
        if frame == frame_flicker[flicker]:
            fix.set_colors(colors=([0,0,0],fix_color[flicker]))
            ec.write_data_line('dotcolorFix', fix_color[flicker])
            time2Flip = 1
            if flicker < len(frame_flicker)-2:
                flicker += 1
                
        if frame == frame_img[trial]:
            ec.write_data_line('imtype', imtype[trial])
            time2Flip = 1
            if frame == start_frame:
                ec.write_data_line('Start')
                trig = 1
            elif frame == end_frame:
                ec.write_data_line('End')
            
            img[trial].draw()
            if imtype[trial] == 'Oddball':
                bright2.draw()
            
            if trial < len(imtype)-1:
                trial += 1
        elif frame == blank_img[b_trial]:
            blank.draw()
            time2Flip = 1
            trig = 0
            if b_trial < len(imtype)-1:
                b_trial += 1
        else:
            time2Flip = 0
            trig = 0
            
        fix.draw()
        if trig:
            ec.stamp_triggers(trig, check='int4', wait_for_last=False)
            trig = 0
        if time2Flip:
            last_flip = ec.flip()
            frametimes.append(last_flip)
        ec.get_presses()
#        ec.wait_for_presses(max_wait = 0.1)
        
#        ec.check_force_quit()
        while time.time()-t0 < (frame+1)*fr - fr/60:
            ec.check_force_quit()
        frame += 1
        
    # Now the experiment is over and we show 5 seconds of blank
    print "\n\n Elasped time: %0.4f secs" % (time.time()-t0)
    print "\n\n Targeted time: %0.4f secs" % total_time
#    blank.draw(), fix.draw()
#    ec.flip()
#    ec.wait_secs(5.0)
#    pressed = ec.get_presses()  # relative_to=0.0
    
    