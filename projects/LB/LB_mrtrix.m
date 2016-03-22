sub_dirs = {'/home/jyeatman/projects/Lindamood_Bell/MRI/child/control/session2/LB7_20130802/20130802_1028/dti80trilin_mrtrix'}
sub_group=1;   
afq = AFQ_Create('sub_dirs',sub_dirs,'sub_group',sub_group,'computeCSD',1,'clip2rois',0);
afq = AFQ_set(afq,'overwritefibers',1)
AFQ_run([],[],afq)
