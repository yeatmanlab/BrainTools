% Make images of single words and pseudowords at different contrast and
% noise levels. This is the code that has been used for MEG experiments on
% reading.
%
% Dependencies:
% vistadisp - https://github.com/vistalab/vistadisp

% read in words
[~,~,w]=xlsread('Word_Pseduoword_List.xlsx');

% Upper or lower case?
% wfun = @upper;
wfun = @lower;

% Remove boarder?
removeboarder = 1;

%% loop over words and render
for ii = 2:101
   wordIm(:,:,ii-1) = uint8(renderText(wfun(w{ii,1}),'courier',20,5));
end

if removeboarder ==1
    m = mean(wordIm,3);
    my = mean(m,2);
    mx = mean(m,1);
    ry = [min(find(my>0)) max(find(my>0))];
    rx = [min(find(mx>0)) max(find(mx>0))];
    wordIm = wordIm(ry(1):ry(2),rx(1):rx(2),:);
end

wordIm(wordIm == 0) = 127;
wordIm(wordIm == 1) = 254;


cvals = [131  133  137 141 150  254]
pvals = [0 .1 .2 .3 .4 .5 .6 .7 .8 .9]

% loop over noise and contrast levels and render
for ii = 1:length(cvals)
    wordIm(wordIm > 127) = cvals(ii);
    % Write out pngs
    for s = 1:length(pvals)
        mkdir(sprintf('word_c%d_p%d',cvals(ii),round(pvals(s)*100)));
        for jj = 1:100
            imwrite(phaseScramble(wordIm(:,:,jj),pvals(s)),fullfile(sprintf('word_c%d_p%d',cvals(ii),round(pvals(s)*100)),[w{jj+1,1} '.png']));
        end
    end
end

%% loop over pseudowords and render
for ii = 2:151
   nonwordIm(:,:,ii-1) = uint8(renderText(wfun(w{ii,2}),'courier',20,5));
end
nonwordIm(nonwordIm == 0) = 127;
nonwordIm(nonwordIm == 1) = 254;

% render across different contrast and noise levels
for ii = 1:length(cvals)
    nonwordIm(nonwordIm > 127) = cvals(ii);
    % Write out pngs
    for s = 1:length(pvals)
        mkdir(sprintf('nonword_c%d_p%d',cvals(ii),round(pvals(s)*100)));
        for jj = 1:150
            imwrite(phaseScramble(nonwordIm(:,:,jj),pvals(s)),fullfile(sprintf('nonword_c%d_p%d',cvals(ii),round(pvals(s)*100)),[w{jj+1,2} '.png']));
        end
    end
end
%% loop over letter strings and render
for ii = 2:151
   rlIm(:,:,ii-1) = uint8(renderText(wfun(w{ii,3}),'courier',20,5));
end
rlIm(rlIm == 0) = 127;
rlIm(rlIm == 1) = 254;

% render across different contrast and noise levels
for ii = 1:length(cvals)
    rlIm(rlIm > 127) = cvals(ii);
    % Write out pngs
    for s = 1:length(pvals)
        mkdir(sprintf('rand_c%d_p%d',cvals(ii),round(pvals(s)*100)));
        for jj = 1:150
            imwrite(phaseScramble(rlIm(:,:,jj),pvals(s)),fullfile(sprintf('rand_c%d_p%d',cvals(ii),round(pvals(s)*100)),[w{jj+1,3} '.png']));
        end
    end
end
%% loop over bigrams and render
for ii = 2:151
   biIm(:,:,ii-1) = uint8(renderText(wfun(w{ii,4}),'courier',20,5));
end
biIm(biIm == 0) = 127;
biIm(biIm == 1) = 254;

% render across different contrast and noise levels
for ii = 1:length(cvals)
    biIm(biIm > 127) = cvals(ii);
    % Write out pngs
    for s = 1:length(pvals)
        mkdir(sprintf('bigram_c%d_p%d',cvals(ii),round(pvals(s)*100)));
        for jj = 1:150
            imwrite(phaseScramble(biIm(:,:,jj),pvals(s)),fullfile(sprintf('bigram_c%d_p%d',cvals(ii),round(pvals(s)*100)),[w{jj+1,4} '.png']));
        end
    end
end
