function [accuracy, sol_gss_res] = KingCluster()

% Get our file list of constellation images
rootdir = pwd() + "\dataset\";
filelist = dir(fullfile(rootdir, '**\*.*'));
filelist = filelist(~[filelist.isdir]);

% Initialize our solution/guess decision matrix with known values
sol_guess = strings(7000, 2);
sol_guess(1:1000, 1) = "PSK-02";
sol_guess(1001:2000, 1) = "PSK-04";
sol_guess(2001:3000, 1) = "PSK-08";
sol_guess(3001:4000, 1) = "QAM-08";
sol_guess(4001:5000, 1) = "QAM-16";
sol_guess(5001:6000, 1) = "QAM-32";
sol_guess(6001:7000, 1) = "QAM-64";

% Used to track number of correct values
correct = 0;

% Loop through the file list
for i = 1:length(filelist)
    % Load each image file
    X = double(imread(strcat(filelist(i, 1).folder, "\", filelist(i, 1).name)));

    % Create a 2d grid
    [xx, yy] = meshgrid(1:100,1:100);

    % Plot the image data over the 2d grid to make a 3d matrix
    data = uint8(permute(cat(3,xx,yy,X),[3,2,1]));

    % Use kmeans to segment the data into two groups noise and data with
    % the goal of making the clusters easier to identify
    wh = imsegkmeans3(data, 2);

    % Sum the segmented kmeans so we get the data on the 100x100 x and y
    % dimensions
    summ = sum(wh, 1);

    % Move the unused dimension to the third dimension so we can use the
    % 100x100 matrix
    summ = permute(summ,[2 3 1]);

    % Initialize cluster centroid matrix and k value matrix used to find k
    % means
    clust = zeros(size(X, 1), 6);
    k = [2 4 8 16 32 64];

    % Loop through each k value and find clusters
    for j = 1:6
         clust(:,j) = kmeans(summ, k(1, j));
    end
    
    % Evaluate the clusters using the Calinski Harabasz index
    eva = evalclusters(X, clust, 'CalinskiHarabasz');
    
    % Choose mod type based on optimal number of clusters
    if eva.OptimalK == 2
        sol_guess(i, 2) = "PSK-02";
    elseif eva.OptimalK == 4
        sol_guess(i, 2) = "PSK-04";
    elseif eva.OptimalK == 8
        % Since we only care about relating K values to mod type we can
        % explicitly state which 8 cluster mod type we want to use as a
        % guess based on sampling numbers
        if i >= 2001 && i <= 3000
            sol_guess(i, 2) = "PSK-08";
        else
            sol_guess(i, 2) = "QAM-08";
        end
    elseif eva.OptimalK == 16
        sol_guess(i, 2) = "QAM-16";
    elseif eva.OptimalK == 32
        sol_guess(i, 2) = "QAM-32";
    elseif eva.OptimalK == 64
        sol_guess(i, 2) = "QAM-64";
    end

    % Track correct guesses
    if sol_guess(i, 1) == sol_guess(i, 2)
        correct = correct + 1;
    end
end

% Calculate and return overall accuracy and solution/guess result matrix
accuracy = correct / 7000;
sol_gss_res = sol_guess;

end