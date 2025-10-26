function EVAL = Evaluate(ACTUAL, PREDICTED)
% EVALUATE
% -------------------------------------------------------------------------
% Compute evaluation metrics for classification.
%
% INPUTS:
%   ACTUAL    : [N x 1] true class labels (integers 1..K)
%   PREDICTED : [N x 1] predicted class labels (integers 1..K)
%
% OUTPUT:
%   EVAL      : row vector, where
%               EVAL(1,1) = overall classification accuracy in %
%
% -------------------------------------------------------------------------

    % Safety check: make sure inputs are column vectors
    ACTUAL    = ACTUAL(:);
    PREDICTED = PREDICTED(:);

    % Overall accuracy (%)
    correct    = sum(ACTUAL == PREDICTED);
    total      = numel(ACTUAL);
    accuracy   = 100 * (correct / total);

    % Return as row vector for compatibility with the rest of the code
    EVAL = [accuracy];
end
