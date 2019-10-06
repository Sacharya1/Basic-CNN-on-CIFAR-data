

clc 
clear all;

dataSet1 = load ('cifar-10-matlab/data_batch_1.mat');
dataSet2 = load ('cifar-10-matlab/data_batch_2.mat');
dataSet3 = load ('cifar-10-matlab/data_batch_3.mat');
dataSet4 = load ('cifar-10-matlab/data_batch_4.mat');
dataSet5 = load ('cifar-10-matlab/data_batch_5.mat');
% testSet = load ('cifar-10-batches-mat/test_batch.mat');
Labels = load ('cifar-10-matlab/batches.meta.mat');
TrainDataSet = [dataSet1.data; dataSet2.data;dataSet3.data;dataSet4.data;dataSet5.data ];
TrainLabels = [dataSet1.labels; dataSet2.labels;dataSet3.labels;dataSet4.labels;dataSet5.labels];

ModifiedTrainingData= reshape(TrainDataSet',[32,32,3,50000]) ;

% ModifiedTrainingData= TrainDataSet ;


layers = [ ...
    imageInputLayer([32 32 3])
%     batchNormalizationLayer
%     reluLayer
    convolution2dLayer(5,64)
    reluLayer
    convolution2dLayer(5,64)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(384)
    reluLayer
    fullyConnectedLayer(192)
    reluLayer
    fullyConnectedLayer(10) 
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'MaxEpochs',40,...
    'InitialLearnRate',1e-4, ...
    'Verbose',true,...
    'Plots','training-progress');
net=trainNetwork(ModifiedTrainingData, categorical(TrainLabels),layers,options);
CNN_CIFAR_Final=net;
save CNN_CIFAR_Final;

testSet = load ('cifar-10-matlab/test_batch.mat');
TestDataSet= testSet.data;
ModifiedTestSet= reshape(TestDataSet',[32,32,3,10000]) ;
TestLabels= testSet.labels;
% output= CNN_CIFAR(ModifiedTestSet);
output=classify(CNN_CIFAR_Final, ModifiedTestSet);
% trainPerformance = perform(CNN_CIFAR,TestLabels,output);
% Accuracy= sum(output==TestLabels)/numel(TestLabels);
[M,~,N]= unique(output);

k=0;
for i=1:10000
    if (N(i)-TestLabels(i) )==1
        k=k+1;
    end
end
Accuracy_in_Percent=(k/10000)*100

% Weights for layer 1
 W1 = CNN_CIFAR_Final.Layers(2).Weights;
 
% Weights for layer 2

 W2 = CNN_CIFAR_Final.Layers(4).Weights;
 
%  First 5 weights of layer 1

 W1(:,:,1,1)
 W1(:,:,2,1)
 W1(:,:,3,1)
 W1(:,:,1,2)
 W1(:,:,2,2)
 
 %  First 5 weights of layer 2


 W2(:,:,1,1)
 W2(:,:,2,1)
 W2(:,:,3,1)
 W2(:,:,1,2)
 W2(:,:,2,2)
 
 W1=rescale(W1);
 figure1
 montage(W1);
 W2=rescale(W2);
 figure2
 montage(W2);

 