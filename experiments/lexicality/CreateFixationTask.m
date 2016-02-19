function [dcolor, colors] = CreateFixationTask(nframes)
% Create fixation task of random color changes
%
% [dcolor, colors] = CreateFixationTask(nframes)
%
% Create a random changing fixation dot.

% 12 colors
colors = hsv(12).*255;
% make red (the target) half as likely as the others
colors = vertcat(colors, colors(2:12,:));
% Create a random vector indexing into the colors
dcolor = ceil(rand(1, nframes).*(size(colors,2)+1));
% Add numbers so it matches KNK's format
dcolor = [-1 -dcolor -1 1];