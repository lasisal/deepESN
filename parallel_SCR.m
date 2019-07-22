% A sample script for generating  training and testing data; 
% training and testing an ESN on a NARMA time series prediction task.
clear all; 

%%%% generate the training data

sequenceLength = 1000;

disp('Generating data ............');
fprintf('Sequence Length %g\n', sequenceLength );

systemOrder = 3 ; % set the order of the NARMA equation
[inputSequence outputSequence] = generate_NARMA_sequence(sequenceLength , systemOrder) ; 


%%%% split the data into train and test

train_fraction = 0.5 ; % use 50% in training and 50% in testing
[trainInputSequence, testInputSequence] = ...
    split_train_test(inputSequence,train_fraction);
[trainOutputSequence,testOutputSequence] = ...
    split_train_test(outputSequence,train_fraction);



%%%% generate an esn 
nInputUnits = 1; nInternalUnits = 30; nOutputUnits = 1; 
paraNum = 3;
% 

predictedTrainOutput = zeros(400,paraNum);
predictedTestOutput = zeros(400,paraNum);
for i=1:paraNum
esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
    'spectralRadius',0.5,'inputScaling',[0.1],'inputShift',[0], ...
    'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling', 0, ...
    'type', 'plain_esn'); 

%%% VARIANTS YOU MAY WISH TO TRY OUT
% (Comment out the above "esn = ...", comment in one of the variants
% below)

% % Use a leaky integrator ESN
% esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
%     'spectralRadius',0.5,'inputScaling',[0.1;0.1],'inputShift',[0;0], ...
%     'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling', 0, ...
%     'type', 'leaky_esn'); 
% 
% % Use a time-warping invariant ESN (makes little sense here, just for
% % demo's sake)
% esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
%     'spectralRadius',0.5,'inputScaling',[0.1;0.1],'inputShift',[0;0], ...
%     'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling', 0, ...
%     'type', 'twi_esn'); 

% % Do online RLS learning instead of batch learning.
% esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
%       'spectralRadius',0.4,'inputScaling',[0.1;0.5],'inputShift',[0;1], ...
%       'teacherScaling',[0.3],'teacherShift',[-0.2],'feedbackScaling',0, ...
%       'learningMode', 'online' , 'RLS_lambda',0.9999995 , 'RLS_delta',0.000001, ...
%       'noiseLevel' , 0.00000000) ; 

esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;

%%%% train the ESN
nForgetPoints = 100 ; % discard the first 100 points
[trainedEsn stateMatrix] = ...
    train_esn(trainInputSequence, trainOutputSequence, esn, nForgetPoints) ; 

%%%% save the trained ESN
% save_esn(trainedEsn, 'esn_narma_demo_1'); 

%%%% plot the internal states of 4 units
nPoints = 200 ; 
plot_states(stateMatrix,[1 2 3 4], nPoints, i, 'traces of first 4 reservoir units') ; 

% compute the output of the trained ESN on the training and testing data,
% discarding the first nForgetPoints of each
nForgetPoints = 100 ; 

predictedTrainOutput(:,i) = test_esn(trainInputSequence, trainedEsn, nForgetPoints);
predictedTestOutput(:,i) = test_esn(testInputSequence,  trainedEsn, nForgetPoints) ; 
end
totPredictedTrainOutput = sum(predictedTrainOutput,2)/paraNum;
totPredictedTestOutput = sum(predictedTestOutput,2)/paraNum;

% create input-output plots
nPlotPoints = 100 ; 
plot_sequence(trainOutputSequence(nForgetPoints+1:end,:), totPredictedTrainOutput, nPlotPoints,...
    'training: teacher sequence (red) vs predicted sequence (blue)');
plot_sequence(testOutputSequence(nForgetPoints+1:end,:), totPredictedTestOutput, nPlotPoints, ...
    'testing: teacher sequence (red) vs predicted sequence (blue)') ; 

%%%%compute NRMSE training error
trainError = compute_NRMSE(totPredictedTrainOutput, trainOutputSequence); 
disp(sprintf('train NRMSE = %s', num2str(trainError)))

%%%%compute NRMSE testing error
testError = compute_NRMSE(totPredictedTestOutput, testOutputSequence); 
disp(sprintf('test NRMSE = %s', num2str(testError)))

