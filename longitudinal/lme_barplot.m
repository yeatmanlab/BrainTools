function lme_barplot(lme, subplots, colors)
% Plot the outpute of lmefit as a barplot
%
% lme_barplot(lme, subplots, colors)
%
% lme      - 1xn structure containing the output of the lmefit function.
% subplots - There are two options for laying out the plots. If subplots=1
%            then the results of each lme are plotted on the same axis. The
%            colors for each line is determined by the colormap in colors.
%            Alternatively, subplots can be a 1x2 vector laying out the
%            subplot dimensions. In this case each lme will be plotted on
%            its own axis
% colors   - Either a colormap name (eg 'jet', 'autumn') or a nx3 matrix of
%            RGB values denoting the color for each lme plot


