function sessiondir = getsessions(subdir)
% Find the folders labeled by date and pick desired session:
% this is esoteric to bde lab convention with yyyy/mm/dd - will crash otherwise
cd(subdir)
sessiondir = dir(pwd);
% session folders have form yyyy/mm/dd, so length is 8 chars:
sessiondir = sessiondir(cellfun(@length, {sessiondir.name})==8);
sessiondir = cat(1, {sessiondir(:).name});