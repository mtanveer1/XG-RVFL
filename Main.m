%% XG-RVFL main experiment script
% -------------------------------------------------------------------------
% This script performs hyperparameter search with 5-fold cross-validation
% for the proposed XG-RVFL model.
%
%  - Each dataset is normalized to zero mean and unit variance (z-score).
%  - Binary labels in {-1, +1} are mapped to {0,1} and then shifted to {1,2}.
%  - We evaluate via 5-fold cross-validation. Each fold uses ~80% training
%    and ~20% testing, and we report mean ± std test accuracy over folds.
%  - We run a grid search over:
%        C              : regularization parameter (lambda in the paper)
%        N              : number of hidden nodes
%        activation     : index selecting the nonlinear activation
%        a              : asymmetry/steepness parameter of XG loss
%        eta            : bound-control parameter η > 0 in XG loss 
%
%  - For each hyperparameter set, training and evaluation are done by
%    XG_RVFL_Function(), which:
%        * builds the RVFL feature matrix Z = [X , H]
%        * optimizes Θ using the NAG-based, inversion-free solver
%          (Algorithm 2 in the paper) instead of matrix inversion. 
%
%  - Hardware in the paper: MATLAB R2023a, Windows 10, Intel i7-6700 CPU,
%    16 GB RAM. This script is written to be portable. :contentReference[oaicite:6]{index=6}
%
%
% -------------------------------------------------------------------------

clear;
clc;

%% USER: specify where your data lives and where to save results.

% NOTE: Make sure "dataDir" exists and "resultsFile" can be created.

dataDir     = '<PATH_TO_YOUR_DATASETS_FOLDER>';   % e.g. './Data/Binary/UCI/'
resultsFile = '<PATH_TO_RESULTS_FILE>';           % e.g. './Results/UCI_Binary.txt'

Result = fopen(resultsFile,'w');

% Write header line for the results summary file
fprintf(Result, ...
    "DataSetName\t BestMeanTrainAccuracy\t BestStdTrainAccuracy\t BestMeanTestAccuracy\t BestStdTestAccuracy\t Best_C\t Best_N\t Best_Activation\t Best_a\t Best_eta\t \n");

% List all .mat datasets in the specified directory
Directory = dir(fullfile(dataDir, '*.mat'));
l = length(Directory);

for u = 1:l

    %% Load dataset
    t1 = fullfile(dataDir, Directory(u).name);

    % Each .mat is expected to contain a numeric matrix of size
    % [num_samples x (num_features+1)]
    % where the last column is the label.
    %
    % If your file instead stores a struct with X and Y separately,
    % adapt this block accordingly.
    All_data = importdata(t1);

    % Map {-1,+1} labels to {0,1} so that we can then shift to {1,2}
    [m,n] = size(All_data);
    for i = 1:m
        if All_data(i,n) == -1
            All_data(i,n) = 0;
        end
    end

    % z-score normalize features (column-wise standardization)
    All_X_data = zscore(All_data(:,1:end-1)')';
    % shift labels by +1 so they start at 1 instead of 0
    All_Y_data = All_data(:,end) + 1;

    % record number of classes (for multi-class setting)
    classes = unique(All_Y_data);
    option.nclass = length(classes);

    % combine back into a single matrix [features | label]
    All_data = [All_X_data, All_Y_data];

    [length_train, ~] = size(All_data);

    %% Hyperparameter ranges (match paper description)
    if length_train < 100
        option.m = 2^3;
    else
        option.m = 2^6;
    end

    % Grid search ranges:
    C_range        = 10.^(-5:1:5);        % regularization parameter values
    Node_range     = 3:20:203;            % number of hidden nodes
    Act_range      = 1:1:6;               % activation function index
    a_range        = [-2,-1.5,-1,-0.5,0.5,1,1.5,2]; % asymmetry/steepness 'a' (a ≠ 0) 
    eta_range      = 0.5:0.5:2;           % bound-control parameter η > 0 

    % Track the best configuration across the grid
    BestMeanTestAccuracy = 0;
    no_part = 5;  % 5-fold CV (80/20 splits per fold)

    % These will store stats for the best combo
    % E2(1) = BestMeanTrainAccuracy
    % E2(2) = BestStdTrainAccuracy
    % E2(3) = BestMeanTestAccuracy
    % E2(4) = BestStdTestAccuracy

    for ii = 1:length(C_range)
        option.C = C_range(ii);

        for jj = 1:length(Node_range)
            option.N = Node_range(jj);

            for ll = 1:length(Act_range)
                option.activation = Act_range(ll);
               

                for mm = 1:length(a_range)
                    option.a = a_range(mm);

                    for nn = 1:length(eta_range)
                        option.eta = eta_range(nn);

                        %% Manual 5-fold CV over contiguous blocks
                        block_size = length_train / (no_part * 1.0);

                        % We'll store fold-wise results here
                        % TempResult(part,:) = [TrainAcc, TestAcc, TrainTime, ValidTime]
                        % TempTestingAccuracy(part) = TestAcc
                        TempResult = zeros(no_part,4);
                        TempTestingAccuracy = zeros(no_part,1);

                        part = 0;
                        t_1  = 0;
                        t_2  = 0;

                        % loop over folds
                        while ceil((part+1) * block_size) <= length_train

                            % determine the index range for this fold
                            if part == 0
                                t_1 = ceil(part*block_size);
                                t_2 = ceil((part+1)*block_size);

                                DataTest  = All_data(t_1+1 : t_2, :);
                                DataTrain = All_data(t_2+1 : length_train, :);

                            elseif part == no_part-1
                                t_1 = ceil(part*block_size);
                                t_2 = ceil((part+1)*block_size);

                                DataTest  = All_data(t_1+1 : t_2, :);
                                DataTrain = All_data(1 : t_1, :);

                            else
                                t_1 = ceil(part*block_size);
                                t_2 = ceil((part+1)*block_size);

                                DataTest  = All_data(t_1+1 : t_2, :);
                                DataTrain = [All_data(1:t_1,:); All_data(t_2+1:length_train,:)];
                            end

                            % split into X (features) and Y (labels) for train/test
                            trainX = DataTrain(:,1:end-1);
                            trainY = DataTrain(:,end);
                            testX  = DataTest(:,1:end-1);
                            testY  = DataTest(:,end);

                            % === CORE CALL ===
                            % XG_RVFL_Function must:
                            %   - build hidden layer
                            %   - construct Z = [X, H]
                            %   - train Θ with Nesterov Accelerated Gradient (Algorithm 2),
                            %     no matrix inversion, using XG loss (Eq. (4)).
                            %   - return:
                            %       EVAL_Train(1,1) = train accuracy (%)
                            %       EVAL_Test(1,1)  = test  accuracy (%)
                            %       train_time      = training runtime (sec)
                            %       valid_time      = inference/runtime on test (sec)
                            [EVAL_Train, EVAL_Test, train_time, valid_time] = ...
                                XG_RVFL_Function(trainX, trainY, testX, testY, option);

                            % store fold stats
                            TempResult(part+1,:)        = [EVAL_Train(1,1), EVAL_Test(1,1), train_time, valid_time];
                            TempTestingAccuracy(part+1) = EVAL_Test(1,1);

                            part = part + 1;
                        end % folds

                        % summarize performance for this hyperparameter combo
                        meanCV = mean(TempTestingAccuracy);
                        stdCV  = std(TempTestingAccuracy);

                        % check if this is the best so far
                        if BestMeanTestAccuracy < meanCV
                            BestMeanTestAccuracy = meanCV;

                            % record stats
                            E2(1,1) = mean(TempResult(:,1)); % BestMeanTrainAccuracy
                            E2(1,2) = std(TempResult(:,1));  % BestStdTrainAccuracy
                            E2(1,3) = meanCV;                % BestMeanTestAccuracy
                            E2(1,4) = stdCV;                 % BestStdTestAccuracy

                            % store best hyperparameters
                            best.C          = option.C;
                            best.N          = option.N;
                            best.activation = option.activation;
                            best.a          = option.a;
                            best.eta        = option.eta;
                        end

                        % optional early stop if perfect accuracy reached
                        if BestMeanTestAccuracy == 100
                            break;
                        end

                    end % eta_range
                    if BestMeanTestAccuracy == 100
                        break;
                    end
                end % a_range
                if BestMeanTestAccuracy == 100
                    break;
                end
            end % Act_range
            if BestMeanTestAccuracy == 100
                break;
            end
        end % Node_range
        if BestMeanTestAccuracy == 100
            break;
        end
    end % C_range

    %% Write best result for this dataset to results file
    fprintf(Result, ...
        "%s\t %0.4f\t %0.4f\t %0.4f\t %0.4f\t %0.10f\t %0.4f\t %d\t %0.6f\t %0.6f\t\n", ...
        Directory(u).name, ...
        E2(1,1), E2(1,2), E2(1,3), E2(1,4), ...
        best.C, best.N, best.activation, best.a, best.eta);

end % dataset loop

fclose('all');
