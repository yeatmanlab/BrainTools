


function barplot_ROIs(clustsum,clustnum_mgh,hemi,data,M,outpath)
% load in the cluster number surface
[vol] = load_mgh(clustnum_mgh);
clusters = zeros(1,max(vol));
% collect all clusters into a single matrix, where each column is a
% different cluster
for ii = 1:numel(vol)
    if(vol(ii) ~= 0)
        rr = 1;
        while rr <= rows(clusters)
            if clusters(rr,vol(ii)) ~= 0
                rr = rr + 1;
            else
                break
            end
        end
        clusters(rr,vol(ii)) = ii;
    end
end
for ii = 1:cols(clusters)
    vertices = [];
    % eliminate all trailing zeros from the cluster data
    vertices = clusters(find(clusters(:,ii),1,'first'):find(clusters(:,ii),1,'last'),ii);
    jj = 1;
    set(0,'DefaultFigureVisible','off');
    fig = figure;
    corr_val = 0;
    hold on
    consol_mtx = [];
    highreader_avg_thickness = [];
    lowreader_avg_thickness = [];
    % constructs a matrix where each column is the subjects data for each
    % vertex
    for rr = 1:numel(vertices)
        consol_mtx(:,jj) = data(:,vertices(rr));
        jj = jj + 1;
    end
    hh = 1;
    ll = 1;
    for rr = 1:rows(consol_mtx)
        if M(rr,2) < 85
            lowreader_avg_thickness(ll)= mean(consol_mtx(rr,:));
            ll = ll + 1;
        else
            highreader_avg_thickness(hh) = mean(consol_mtx(rr,:));
            hh = hh +1;
        end
    end
    c = categorical({'high','low'});
    bar(c,[mean(highreader_avg_thickness),mean(lowreader_avg_thickness)],0.5,'r')
    errorbar([mean(highreader_avg_thickness),mean(lowreader_avg_thickness)],[std(highreader_avg_thickness),std(lowreader_avg_thickness)],'.')
    xlabel('High vs Low readers')
    ylabel(strcat('Cortical Thickness in',[' '],clustsum.Annot(ii)))
    hold off
    saveas(fig,char(fullfile(outpath,strcat('highvlow_tp1_',hemi,clustsum.Annot(ii),'.jpg'))))
end
end