%%
clear variables

session1 = {'102_rs160618','103_ac150609','105_bb150713','110_hh160608','127_am151022', ...
       '130_rw151221','132_wp160919','133_ml151124','145_ac160621','150_mg160606', ...
       '151_rd160620','152_tc160422','160_ek160627','161_ak160627','163_lf160707', ...
       '164_sf160707','170_gm160613','172_th160614','174_hs160620','179_gm160701', ...
       '180_zd160621','187_nb161017','201_gs150818','203_am150831', ...
       '204_am150829','205_ac151123','206_lm151119','207_ah160608','211_lb160617', ...
       'nlr_gb310170614','nlr_kb218170619','nlr_jb423170620','nlr_gb267170620','nlr_jb420170621', ...
       'nlr_hb275170622','197_bk170622','nlr_gb355170606','nlr_gb387170608','nlr_hb205170825', ...
       'nlr_ib217170831','nlr_ib319170825','nlr_jb227170811','nlr_jb486170803','nlr_kb396170808', ...
       'nlr_ib357170912'};
goodReaders = [ 0,  1,  2,  4,  5,  6,  9, 10, 11, 21, 27, 36, 38, 39, 40, 43] + 1;
poorReaders = [ 3,  7,  8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, ...
       28, 29, 30, 31, 32, 33, 34, 35, 37, 41, 42, 44]+1;
cd(fullfile('/mnt/scratch/NLR_MEG4'))
%%
% cd(fullfile('/mnt/diskArray/projects/MEG/nlr/bdata'))
subject = {'102_RS','103_AC','105_BB','110_HH','127_AM', ...
        '130_RW','132_WP','133_ML','145_AC','150_MG', ...
        '151_RD','152_TC','160_EK','161_AK','163_LF', ...
        '164_SF','170_GM','172_TH','174_HS','179_GM', ...
        '180_ZD','187_NB','201_GS','203_AM', ...
        '204_AM','205_AC','206_LM','207_AH','211_LB', ...
        'GB310','KB218','JB423','GB267','JB420', ...
        'HB275','HA072','GB355','GB387','HB205', ...
        'IB217','IB319','JB227','JB486','KB396', ...
        'IB357'};
dotTime = cell(length(subject),6);

for sub = 1: length(subject)
    cd(fullfile(sprintf('/mnt/scratch/NLR_MEG4/%s/%s',cell2mat(session1(sub)),'lists')))
    d = dir(sprintf('%s*.tab',subject{sub}));
    
    if length(d) > 6
        for j = 1: 6
            clear A
            fid = fopen(d(j).name);

            temp = textscan(fid, '%s');
            xxx = find(strcmp(temp{1},'value'));
                
            T = temp{1}(xxx+1:end);

            for i = 1: size(T,1)/3
                A(i,1) = T(3*(i-1)+1);
                A(i,2) = T(3*(i-1)+2);
                A(i,3) = T(3*(i-1)+3);
            end

            kk = 1;
            mute = 0;
            fixTime = [];
            for k = 1: size(A,1)
                if strcmp(A(k,2),'dotcolorFix')
                    mute = mute+1;
                    if strcmp(A(k,3),'r') && mod(mute,3)==1
                        fixTime = [fixTime; str2num(cell2mat(A(k,1)))];
                        kk = kk + 1;
                    end
                elseif strcmp(A(k,2),'dotcolorIm')
                    mute = mute + 1;
                    if mute == 2
                        startTime = str2num(cell2mat(A(k,1)));
                    end
                    if strcmp(A(k,3),'r')
                        fixTime = [fixTime; str2num(cell2mat(A(k,1)))];
                        kk = kk + 1;
                    end
                end
            end

            fixTime = fixTime - startTime;
            dotTime{sub,j} = fixTime;
            fclose(fid);
        end
    else
        for j = 1: length(d)
            clear A
            fid = fopen(d(j).name);

            temp = textscan(fid, '%s');

            T = temp{1}(14:end);

            for i = 1: size(T,1)/3
                A(i,1) = T(3*(i-1)+1);
                A(i,2) = T(3*(i-1)+2);
                A(i,3) = T(3*(i-1)+3);
            end

            kk = 1;
            mute = 0;
            fixTime = [];
            for k = 1: size(A,1)
                if strcmp(A(k,2),'dotcolorFix')
                    mute = mute+1;
                    if strcmp(A(k,3),'r') && mod(mute,3)==1
                        fixTime = [fixTime; str2num(cell2mat(A(k,1)))];
                        kk = kk + 1;
                    end
                elseif strcmp(A(k,2),'dotcolorIm')
                    mute = mute + 1;
                    if mute == 2
                        startTime = str2num(cell2mat(A(k,1)));
                    end
                    if strcmp(A(k,3),'r')
                        fixTime = [fixTime; str2num(cell2mat(A(k,1)))];
                        kk = kk + 1;
                    end
                end
            end

            fixTime = fixTime - startTime;
            dotTime{sub,j} = fixTime;
            fclose(fid);
        end
    end
end

%%
sampleRate = 1200;
hit_rate = NaN * ones(1,length(session1));
overall_rt = NaN * ones(1,length(session1));
for sub = 1: length(session1)
    cd(fullfile(sprintf('/mnt/scratch/NLR_MEG4/%s/%s',cell2mat(session1(sub)),'lists')))
    
    if strcmp(session1(sub),'164_sf160707')
        runs = [1 3 5];
    elseif strcmp(session1(sub),'170_gm160613')
        runs = [1 3 5];
    elseif strcmp(session1(sub),'nlr_ib357170912')
        runs = [1 5];
    else
        runs = [1 3 5];
    end
    
    switch cell2mat(session1(sub))
        case '103_ac150609'
            key_string = '132';
        case {'130_rw151221','nlr_gb310170614','211_lb160617','nlr_gb387170608', ...
                'nlr_kb218170619','nlr_jb423170620','nlr_gb267170620', ...
                'nlr_jb420170621','nlr_hb275170622','197_bk170622', ...
                'nlr_gb355170606','nlr_ib217170831','nlr_ib319170825', ...
                'nlr_jb486170803','nlr_kb396170808','nlr_ib357170912'}
            key_string = '228';
        otherwise
            key_string = '116';
    end
    
    for run = 1: length(runs)
        switch cell2mat(session1(sub))
            case {'174_hs160620','nlr_hb205170825','nlr_jb227170811'}
                pressTime{sub,run} = NaN;
            otherwise
                
                clear A
                fn = sprintf('ALL_%s_%d-eve.lst',cell2mat(session1(sub)),runs(run));
                
                fid = fopen(fn);
                temp = textscan(fid, '%s');
                
                for i = 1: size(temp{1},1)/3
                    A(i,1) = temp{1}(3*(i-1)+1);
                    A(i,2) = temp{1}(3*(i-1)+2);
                    A(i,3) = temp{1}(3*(i-1)+3);
                end
                startTime = str2num(cell2mat(A(1,1)));
                tmepTime = [];
                for k = 1: size(A,1)
                    if strcmp(A(k,3),key_string)
                        tmepTime = [tmepTime; str2num(cell2mat(A(k,1)))];
                    end
                end
                pressTime{sub,run} = (tmepTime-startTime)./sampleRate;
                fclose(fid);
        end
    end
end

%%
resWindow = 1.5; % 1.5 s response window
for sub = 1: length(session1)
    for run = 1: 3
        for i = 1: length(dotTime{sub,2*run-1})
            response{sub,run}(i) = any(find(pressTime{sub,run} > dotTime{sub,2*run-1}(i) & ...
                pressTime{sub,run} <= dotTime{sub,2*run-1}(i) + resWindow));
            if response{sub,run}(i)
                temp = min(find(pressTime{sub,run} > dotTime{sub,2*run-1}(i) & ...
                    pressTime{sub,run} <= dotTime{sub,2*run-1}(i) + resWindow));
                rt{sub,run}(i) = pressTime{sub,run}(temp) - dotTime{sub,2*run-1}(i);
            else
                rt{sub,run}(i) = NaN;
            end
        end
    end
end

for sub = 1: length(session1)
    for i = 1: 3
        allResponse{sub} = [response{sub,1} response{sub,2} response{sub,3}];
        allRT{sub} = [rt{sub,1} rt{sub,2} rt{sub,3}];
    end
end

meanResponse = [];
meanRT = [];
for sub = 1: length(session1)
    meanResponse = [meanResponse nansum(allResponse{sub})/length(~isnan(allResponse{sub}))];
    meanRT = [meanRT nanmedian(allRT{sub})];
end

l = zeros(1,length(session1));
l(goodReaders) = 1;

controlResponse = 100*meanResponse(~isnan(meanRT) & l);
errControlRes = std(controlResponse)/sqrt(length(controlResponse));
controlRT = meanRT(~isnan(meanRT) & l);
errControlRT = std(controlRT)/sqrt(length(controlRT));

dysResponse = 100*meanResponse(~isnan(meanRT) & ~l);
errDysRes = std(dysResponse)/sqrt(length(dysResponse));
dysRT = meanRT(~isnan(meanRT) & ~l);
errDysRT = std(dysRT)/sqrt(length(dysRT));

figure(1); clf;
subplot(1,2,1); hold on;
bar([1,2],[mean(controlResponse) mean(dysResponse)],'FaceColor',[.4 .4 .4])
plot([1,1],[mean(controlResponse)-errControlRes, mean(controlResponse)+errControlRes],'-','Color',[0 0 0])
plot([2,2],[mean(dysResponse)-errDysRes, mean(dysResponse)+errDysRes],'-','Color',[0 0 0])
set(gca,'YLim',[0 100],'XTick',1:2,'XTickLabel',{'Typical','Struggling'},'TickDir','out')
ylabel('Hit rate (%)')

subplot(1,2,2); hold on;
bar([1,2],[mean(controlRT) mean(dysRT)],'FaceColor',[.4 .4 .4])
plot([1,1],[mean(controlRT)-errControlRT, mean(controlRT)+errControlRT],'-','Color',[0 0 0])
plot([2,2],[mean(dysRT)-errDysRT, mean(dysRT)+errDysRT],'-','Color',[0 0 0])
set(gca,'YLim',[0 1],'XTick',1:2,'XTickLabel',{'Typical','Struggling'},'TickDir','out')
ylabel('Reaction time (s)')

[n,p]=ttest2(controlResponse, dysResponse)
[n2,p2]=ttest2(controlRT, dysRT)