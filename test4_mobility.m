%% mobility data set using convolutional neural network
clear all
close all
load('06-Mar-2018 13:45:16.mat') % cell id 4

clear net
A=1:length(data); 
frac=0.3;
mon=6;
len=length(data);


test_data=data(month_change(mon)+1:month_change(mon+1),:);

for i=1:length(test_data)
    temp=reshape(de2bi(1*test_data(i,[2:1:10,12,13]),24,'left-msb'),1,[]);
    test_data_input1(i,:)=temp;
    
end
test_data_input=test_data_input1;
test_data_output_temp=(1*de2bi(test_data(:,15),11,'left-msb'));
test_data_output_temp2=[test_data_output_temp ~test_data_output_temp]';
k=1;
for i=1:11
    test_data_output([k,k+1],:)=test_data_output_temp2([i,11+i],:);
    k=k+2;
    
    
end

%

training_data1=data(month_change(mon-3)+1:month_change(mon),:);
%training_data=data(s2,:);
[val, id]=sort(training_data1(:,15));
training_data=training_data1(id,:);
for i=1:length(training_data)
    temp=reshape(de2bi(1*training_data(i,[2:1:10,12,13]),24,'left-msb'),1,[]);
    training_data_input1(i,:)=temp;
  
end
training_data_input=training_data_input1;

training_data_target_temp=(1*de2bi(training_data(:,15),11,'left-msb'));
training_data_target_temp2=[training_data_target_temp ~training_data_target_temp]';
k=1;
for i=1:11
    training_data_target([k,k+1],:)=training_data_target_temp2([i,11+i],:);
    k=k+2;
    
    
end
%
TrainX = single(permute(reshape(training_data_input', [24 11 1 size(training_data_input,1)]), [2 1 3 4]));
TestX = single(permute(reshape(test_data_input', [24 11 1 size(test_data_input,1)]), [2 1 3 4]));

train_x = single(TrainX(:, :, :, 1:size(training_data_input,1)));
train_y=training_data_target';

test_x = single(TestX(:, :, :, 1:size(test_data_input,1)));
test_y=test_data_output';

%%
numTrainImages = size(train_x,4);
figure
idx = randperm(numTrainImages,20);
for i = 1:numel(idx)
    subplot(4,5,i)    
    imshow(train_x(:,:,:,idx(i)));
    drawnow
end
%%
clear layers;
clear net;
layers = [
    imageInputLayer([11 24 1])
    
    convolution2dLayer(2,50,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,40,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,30,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(22)
    softmaxLayer
    reluLayer
    regressionLayer
    %classificationLayer
    ];
options = trainingOptions('sgdm', ...
    'MaxEpochs',3, ...
    'InitialLearnRate',1e-3,...
    'LearnRateSchedule','piecewise',...
    'ValidationData',{test_x,test_y},...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%%
net = trainNetwork(train_x,train_y,layers,options);

%%
clear dt;
dt=predict(net,test_x);

 %
count=0;
for i=1:length(test_data_output)
    buffer=[];
    for j=1:2:22
        if test_y(i,j)>=test_y(i,j+1)
            k=1;
        else k=0;
        end
            buffer=[buffer k];
        
    
    end
    
   d1=bi2de(buffer,'left-msb');
    buffer=[];
    for j=1:2:22
        if dt(i,j)>=dt(i,j+1)
            k=1;
        else k=0;
        end
            buffer=[buffer k];
        
    
    end
    
   d2=bi2de(buffer,'left-msb');
   
   %d2=bi2de((dt(:,i)>=1),'left-msb');
   if (d1==d2)
       count=count+1;
   end
    
end
corrected_answer=count;
%corrected_answer=sum((dt == test_data_output));
errr=(length(test_data_output)-corrected_answer)/length(test_data_output)*100
