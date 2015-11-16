function fsl_applyXform(xformIm, refIm, xform, outname)

cmd = sprintf('/usr/share/fsl/5.0/bin/flirt -in %s -ref %s -out %s -applyxfm -init %s -interp nearestneighbour', xformIm, refIm, outname,xform);

system(cmd)