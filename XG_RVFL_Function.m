function [EVAL_Train, EVAL_Test, TrainTime, TestTime] = XG_RVFL_Function(trainX, trainY, testX, testY, option)
% XG_RVFL_FUNCTION
% -------------------------------------------------------------------------
% Train and evaluate the proposed XG-RVFL model on a given train/test split.
%
% This implements the inversion-free Nesterov Accelerated Gradient (NAG)
% optimization described in the manuscript
% "Towards Robust and Inversion-Free Randomized Neural Networks:
%  The XG-RVFL Framework". 
%
% INPUTS:
%   trainX  : [n_train x d] training features
%   trainY  : [n_train x 1] class labels in {1,...,K}
%   testX   : [n_test  x d] test features
%   testY   : [n_test  x 1] class labels in {1,...,K}
%
%   option struct must contain:
%       option.N            number of hidden nodes in the random layer
%       option.C            regularization / penalty coefficient (λ in paper)
%       option.activation   activation index for hidden layer:
%                           1=sigmoid, 2=sine, 3=tribas, 4=radbas,
%                           5=tansig, 6=ReLU
%       option.nclass       number of classes K
%       option.m            minibatch size k used by NAG (Algorithm 2) 
%       option.a            XG-loss asymmetry/slope parameter "a"
%       option.eta          XG-loss bounding/robustness parameter "η">0
%
% OUTPUTS:
%   EVAL_Train : training evaluation (e.g. accuracy %), from Evaluate()
%   EVAL_Test  : test     evaluation (e.g. accuracy %), from Evaluate()
%   TrainTime  : wall-clock training time in seconds
%   TestTime   : wall-clock inference+evaluation time in seconds
%
% KEY IDEAS:
%   • The RVFL hidden layer is randomized once (W, bias), not trained.
%   • The output weights Θ (called 'beta' below) are learned using a
%     Nesterov Accelerated Gradient solver with momentum γ and an
%     exponentially decaying learning rate μ(t).
%   • NO matrix inversion is performed (unlike standard RVFL closed-form).
%     This is the “inversion-free” claim of XG-RVFL. 
%   • The loss is the proposed XG loss, which is asymmetric, bounded,
%     and robust to outliers. Its partial derivative ∂L_XG/∂u is given
%     in compute_dLdu_XG() below.
%
%   Θ is learned on a minibatch of size m (k in Algorithm 2). At test time,
%   we reuse the same Θ and random layer to classify all samples.
%
% -------------------------------------------------------------------------

    %% === Unpack options ===
    N           = option.N;
    C           = option.C;
    activation  = option.activation;
    nclass      = option.nclass;

    m           = option.m;      % minibatch size k for NAG
    a           = option.a;      % XG-loss "a"
    eta         = option.eta;    % XG-loss "η"
    s           = 1;             % scale for random weight init


    %% === One-hot encode training labels ===
    % trainY is expected to be in {1,2,...,nclass}
    trainY_onehot = zeros(numel(trainY), nclass);
    for c = 1:nclass
        trainY_onehot(trainY == c, c) = 1;
    end

    %% === Build deterministic minibatch of size m ===

    alltrain = [trainX, trainY_onehot];
    rng(0);
    numRows = size(alltrain, 1);
    permIdx = randperm(numRows);
    batch_data = alltrain(permIdx(1:m), :);

    trainX_batch = batch_data(:, 1:end-nclass);      % [m x d]
    trainY_batch = batch_data(:, end-nclass+1:end);  % [m x K]

    [numBatchSamples, numFeat] = size(trainX_batch);


    %% === Random hidden layer construction (RVFL feature map) ===
    % Generate random affine transform, apply nonlinearity, then concatenate
    % with direct link and output bias term:
    %
    %    H = act( X W + bias )
    %    Z = [ X , H , 1 ]
    %
    % This matches the standard RVFL architecture with direct links,
    % as described in the paper. :contentReference[oaicite:7]{index=7}

    tic; % start training timer

    W    = (rand(numFeat, N) * 2 * s - 1);  % [d x N]
    bias = s * rand(1, N);                  % [1 x N]
    H    = trainX_batch * W + repmat(bias, numBatchSamples, 1);

    switch activation
        case 1
            H = sigmoid(H);
        case 2
            H = sin(H);
        case 3
            H = tribas(H);
        case 4
            H = radbas(H);
        case 5
            H = tansig(H);
        case 6
            H = relu(H);
        otherwise
            error('Unknown activation index. Please document mapping in README.');
    end

    Z_batch = [trainX_batch, H];
    Z_batch = [Z_batch, ones(numBatchSamples,1)]; % append bias term
    % Z_batch is m x (d+N+1)


    %% === Nesterov Accelerated Gradient (Algorithm 2) ===
    % We now learn Θ (called 'beta') using:
    %   - Momentum γ
    %   - Learning rate μ(t) with exponential decay
    %   - Max iterations I_max
    %
    % Objective:
    %   J(Θ) = ||Θ||_2^2 + C * sum_{i=1}^m L_XG( u_i ),
    % where u_i = Z_i Θ - y_i  (row-wise residual), and L_XG is the
    % proposed XG loss. 
    %
    % Gradient wrt Θ:
    %   ∇J(Θ) = 2Θ + C * Z_batch' * (∂L_XG/∂u)
    %
    % NAG update:
    %   Θ_tilde = Θ_t + γ v_t
    %   v_{t+1} = γ v_t - μ(t) * ∇J(Θ_tilde)
    %   Θ_{t+1} = Θ_t + v_{t+1}
    %
    % with μ(t+1) = μ(t) * exp(-α). 

    max_iter = 500;       % I_max
    tol      = 1e-6;      % stopping tolerance
    lr       = 0.01;      % μ(0)
    decay    = 0.1;       % α (learning rate decay factor)
    gamma    = 0.6;       % γ (momentum)

    beta          = ones(size(Z_batch, 2), nclass) * 0.01; % Θ^(0)
    velocity      = zeros(size(Z_batch, 2), nclass);       % v^(0)
    beta_previous = inf;

    for t = 1:max_iter
        % Look-ahead weight
        beta_look = beta + gamma * velocity;

        % Residuals u = ZΘ - Y  (m x K)
        residual = Z_batch * beta_look - trainY_batch;

        % Compute dL/du for XG loss (vectorized over all samples/classes)
        dL_du = compute_dLdu_XG(residual, a, eta);    % [m x K]

        % Gradient components:
        %   grad_reg     = 2 * beta_look
        %   grad_data    = C * Z_batch' * dL_du
        grad_reg  = 2 * beta_look;
        grad_data = C * (Z_batch' * dL_du);
        grad_tot  = grad_reg + grad_data;

        % NAG update
        velocity = gamma * velocity - lr * grad_tot;
        beta     = beta + velocity;

        % Exponential learning-rate decay
        lr = lr * exp(-decay);

        % Early stopping if parameters stabilize
        if norm(beta - beta_previous, 'fro') < tol
            break;
        else
            beta_previous = beta;
        end
    end
    % 'beta' is now the learned output weight matrix Θ.


    %% === Training evaluation ===
    train_scores = Z_batch * beta;                 % [m x K]
    [~, pred_train_labels] = max(train_scores, [], 2);

    % Recover minibatch ground-truth labels from one-hot
    [~, true_train_labels] = max(trainY_batch, [], 2);

    EVAL_Train = Evaluate(true_train_labels, pred_train_labels);

    TrainTime = toc; % end training timer


    %% === Testing / inference ===
    % Forward the ENTIRE test set through the SAME random layer (W,bias)
    % and classify using the learned Θ ("beta").
    tic;

    numTestSamples = size(testX, 1);

    H_test = testX * W + repmat(bias, numTestSamples, 1);

    switch activation
        case 1
            H_test = sigmoid(H_test);
        case 2
            H_test = sin(H_test);
        case 3
            H_test = tribas(H_test);
        case 4
            H_test = radbas(H_test);
        case 5
            H_test = tansig(H_test);
        case 6
            H_test = relu(H_test);
    end

    Z_test = [testX, H_test];
    Z_test = [Z_test, ones(numTestSamples,1)];

    test_scores = Z_test * beta;                 % [n_test x K]
    [~, pred_test_labels] = max(test_scores, [], 2);

    % testY is already class index {1,...,K}
    EVAL_Test = Evaluate(testY, pred_test_labels);

    TestTime = toc;
end


%% ===================================================================== %%
%% Gradient of XG Loss w.r.t. residual u                                  %
%% ===================================================================== %%
function dL_du = compute_dLdu_XG(u, a, eta)
% compute_dLdu_XG
% -------------------------------------------------------------------------
% Compute ∂L_XG(u)/∂u elementwise for the proposed XG loss.
%
% INPUT:
%   u   : [m x K] residual matrix, where u(i,j) = (ZΘ)_ij - Y_ij
%   a   : XG-loss parameter "a" (controls asymmetry / slope)
%   eta : XG-loss parameter "η" (>0) (controls bounded influence)
%
% OUTPUT:
%   dL_du : [m x K] matrix of partial derivatives ∂L_XG/∂u(i,j)
%
% NOTE:
%   This is the exact derivative you provided (previously implemented
%   inside computeGradient). We now vectorize it.
%
% Original scalar formula for each residual u_ij was:
%
%   dL/du = [ a * (a*u + 1) * exp(a*u) - a ] / [ 1 + a*eta*u*(exp(a*u)-1) ]^2
%
% Where:
%   - The numerator grows approximately linearly for small |u|, but
%     includes exp(a*u) to create asymmetry (a can weight positive vs
%     negative residuals differently).
%   - The denominator has the (1 + a*eta*u*(exp(a*u)-1))^2 term, which
%     suppresses very large residuals, giving the bounded influence
%     behavior reported in the manuscript. :contentReference[oaicite:10]{index=10}
%
% This derivative drives robustness and is what replaces the standard
% squared loss gradient in RVFL training.
% -------------------------------------------------------------------------

    % Precompute exp(a*u) elementwise
    au      = a .* u;              % a*u
    exp_au  = exp(au);             % exp(a*u)
    % Numerator: a * (a*u + 1) * exp(a*u) - a
    numerator = a .* (au + 1) .* exp_au - a;

    % Denominator base: 1 + a*eta*u*(exp(a*u) - 1)
    denom_base = 1 + (a .* eta .* u .* (exp_au - 1));

    % Square the denominator
    denom = denom_base .^ 2;

    % Final derivative
    dL_du = numerator ./ denom;
end
