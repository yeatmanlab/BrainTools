% function runRSVP(sub_num, trial)

% Set number of runs
nruns = 20;

% set items in an rsvp stream
nitems = 10;

% Possible target locations
targetlocs = 3:8;

% Preallocate a cell array to collect data.
C = cell((nruns*36),5);

% ISI in ms
isi = [10 20 40 80 160];


target = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', ...
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
nottarget = {'!' '@' '%' '&' '*' '#' '<' '>' '='};

% Generate RSVP stream
stim = RandSample(nottarget, [nitems,nruns, length(isi)]);
% Add a target to a random location in every other run
corresp = zeros(size(stim,2),length(isi))
for jj = 1:length(isi)
    for ii = 1:2:size(stim,2)
        corresp(ii,jj) = 1;
        stim(RandSample(targetlocs,1),ii, jj) = RandSample(target,1);
    end
    % Shuffle the run order since we don't want it to be in a predictable order
    shuffindx = Shuffle(1:size(stim,2));
    stim(:,:,jj) = stim(:,shuffindx,jj);
    corresp(:,jj) = corresp(shuffindx,jj);
end


% Perform standard setup for Psychtoolbox
PsychDefaultSetup(2);

% Define black, white, and gray
black = BlackIndex(0);
white = WhiteIndex(0);
gray = white / 2;

ListenChar(2);

% Open the window
PsychImaging('PrepareConfiguration');
PsychImaging('AddTask', 'General', 'UseRetinaResolution');
[window, rect] = PsychImaging('OpenWindow', 0, gray);
%[0 0 1280 600]);
HideCursor;

% Get the center coordinates of the screen
[centerX, centerY] = RectCenter(rect);

% Get the size of the screen window in pixels
[screenXpixels, screenYpixels] = Screen('WindowSize', window);

% % Disable all keys except for space bar, down arrow, right arrow
% oldenablekeys = RestrictKeysForKbCheck([66,117,115]); %([44,79,81]);

%% Instructrions

% Display instructions for the task
instructions = 'Press the right arrow if you saw a letter and press the left arrow if you did not\n Press space to begin.\n';
Screen('TextFont', window, 'Courier');
Screen('TextSize', window, 40);
DrawFormattedText(window, instructions, 'center','center', 0, [], [], [], 1.5);
Screen('Flip', window);

% Wait until user presses a key
[~, ~, ~] = KbWait([], 2);

%% Run an RSVP stream
for thisisi = 1:length(isi)
    for ii = 1:size(stim,2)
        
        Screen('TextFont', window, 'Courier');
        Screen('TextSize', window, 80);
        DrawFormattedText(window, '+', 'center','center', 0, [], [], [], 1.5);
        Screen('Flip', window);
        WaitSecs(1);
        for item = 1:size(stim,1)
            Screen('TextFont', window, 'Courier');
            Screen('TextSize', window, 80);
            
            % Draw the stim item
            DrawFormattedText(window, stim{item,ii}, 'center','center', 0, [], [], [], 1.5);
            
            % Save the time the screen was flipped
            stimulusStartTime(thisisi,item,ii) = Screen('Flip', window);
            
            % Wait for the desired ISI
            WaitSecs(isi(thisisi)./1000);
        end
        
        DrawFormattedText(window, '+', 'center','center', 0, [], [], [], 1.5);
        Screen('Flip', window);
        trialend = GetSecs;
        KbWait;
        [keyIsDown,secs,keyCode] = KbCheck;
        key{thisisi,ii} = KbName(keyCode);
        respTime(thisisi,ii) = GetSecs - trialend;
    end
end
%% close out

finished = 'Press space to continue.';
Screen('TextFont', window, 'Courier');
Screen('TextSize', window, 40);
DrawFormattedText(window, finished, 'center', 'center', 0);
Screen('Flip', window);

KbWait;
sca;
Screen('close')
ListenChar(0);

raw = key;

%% Make a plot where x is ISI and y is % correct
% convert key strokes to booleans
for col = 1:length(isi)
   for row = 1:nruns 
      if ismember(key(col, row),'RightArrow')
          key{col, row} = 1;
      elseif ismember(key(col, row),'DownArrow')
          key{col, row} = 0;
      end
   end
end
% transpose corresp matrix to agree
answers = corresp';
% calculate number of correct responses
num_correct = zeros(length(isi), 2);
for col = 1:length(isi)
    for row = 1:nruns
        if answers(col, row) == key{col, row}
            num_correct(col, 1) = num_correct(col, 1) + 1;
        end
    end
end
% append col with percentages
for row = 1:length(isi)
    num_correct(row, 2) = (num_correct(row,1)/nruns);
end
% plot 
figure; hold;
x = isi; y = num_correct(:,2)';
bar(x,y);

