
def nlr_organizeMEG_mnefun(raw_dir, subs, sess):

    import os
    import shutil
    import glob

    # define subjects and sessions
    raw_dir = '/mnt/diskArray/projects/MEG/nlr/mnetest'
    subs = ['202_dd', '203_am']
    sess = [['150827', '150919', '151013', '151103'],
            ['150831', '150922', '151009', '151029']]
    adir = '/mnt/diskArray/projects/MEG/nlr/mnetest/mneanalysis/'
    # We will make a list of subjects where each subject+session combo is an
    # entry in the list
    subjects = []
    # Loop over the subjects
    for n, s in enumerate(subs):
        # Loop over the sessions for subject n
        for ss in sess[n]:
            # Make a new directory for the subject (s) session (ss) combo
            sdir = os.path.join(adir, s+ss)
            # make a directory for the raw files to be saved
            rfifdir = os.path.join(sdir, 'raw_fif')
            subjects.append(s+ss)  # append subjects with the subjectsession
            # Make directories if they do not exist
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            if not os.path.isdir(rfifdir):
                os.mkdir(rfifdir)
            # List all directory fif files in raw directory
            fiflist = glob.glob(os.path.join(raw_dir, s, ss, '*_raw.fif'))
            # Sort the list
            fiflist.sort()
            # Loop over the list of raw fif files
            for fnum, fname in enumerate(fiflist):
                fifdest = os.path.join(rfifdir, s+ss+fname[-10:])
                # WE ONLY COPY THE FILE OVER IF IT DOESN'T EXIST!
                if not os.path.isfile(fifdest):
                    # Copy this fif file to the destination and rename it to
                    # have the subject name. This is the assumption in mnefun
                    shutil.copy(os.path.join(raw_dir, s, ss, fname), fifdest)
                else:
                    print(fifdest + ' already exists: Skpping')
                # Make the prebad file that mnefun wants
                prebad = os.path.join(rfifdir, s+ss+'_prebad.txt')
                if not os.path.isfile(prebad):
                    open(prebad, 'w').close()

    # Return a list of all the subjects
    return(subjects)
