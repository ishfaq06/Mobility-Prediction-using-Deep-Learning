%% mobility data set
clear all
close all
load('06-Mar-2018 13:45:16.mat') % cell id 4

clear net
A=1:length(data); 
frac=0.3;
mon=6;
len=length(data);
%idx=randperm(numel(A));
%s1=A(idx(1:frac*length(data)));
test_data=data(month_change(mon)+1:month_change(mon+1),:);
%test_data=data(s1,:);
for i=1:length(test_data)
    temp=reshape(de2bi(1*test_data(i,[2:1:10,12,13]),24,'left-msb'),1,[]);
    test_data_input1(i,:)=temp;
    %t=bi2de(reshape(temp,4,8),'left-msb');
end
test_data_input=test_data_input1';
%test_data_input=(test_data(:,[2:1:10,12,13]))';
test_data_output_temp=(1*de2bi(test_data(:,15),11,'left-msb'));
test_data_output_temp2=[test_data_output_temp ~test_data_output_temp]';
k=1;
for i=1:11
    test_data_output([k,k+1],:)=test_data_output_temp2([i,11+i],:);
    k=k+2;
end

%
%subSet2=A(idx(frac*length(data)+1:end));
%s2=sort(subSet2);
%s2=subSet2;
training_data1=data(month_change(mon-3)+1:month_change(mon),:);
%training_data=data(s2,:);
[val, id]=sort(training_data1(:,15));
training_data=training_data1(id,:);
for i=1:length(training_data)
    temp=reshape(de2bi(1*training_data(i,[2:1:10,12,13]),24,'left-msb'),1,[]);
    training_data_input1(i,:)=temp;
  %  t=bi2de(reshape(temp,4,8),'left-msb');
end
training_data_input=training_data_input1';
%training_data_input=(training_data(:,[2:1:10,12,13]))';
training_data_target_temp=(1*de2bi(training_data(:,15),11,'left-msb'));
training_data_target_temp2=[training_data_target_temp ~training_data_target_temp]';
k=1;
for i=1:11
    training_data_target([k,k+1],:)=training_data_target_temp2([i,11+i],:);
    k=k+2;
    
    
end

%input_data=(data(:,2:2:12))';
%output_data=(data(:,15))';


%% using feed forward or patternnet
clear net;
%net = patternnet([100],'trainscg');
net = feedforwardnet([75],'trainscg');
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'tansig';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainParam.epochs=1000;
net.trainParam.max_fail=10;
[net,tr] = train(net,training_data_input,training_data_target);
%% using auto encoder
clear deepnet;
clear autoenc1;
clear autoenc2;
clear softnet;
rng('default')
hiddenSize1 = 100;
autoenc1 = trainAutoencoder(training_data_input,hiddenSize1, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
%view(autoenc1)
%
feat1 = encode(autoenc1,training_data_input);
%
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
%view(autoenc2)
%
feat2 = encode(autoenc2,feat1);
%
softnet = trainSoftmaxLayer(feat2,training_data_target,'MaxEpochs',500);
%view(softnet)
%
deepnet = stack(autoenc1,autoenc2,softnet);

%
deepnet = train(deepnet,training_data_input,training_data_target);
clear dt;
%%
dt=deepnet(test_data_input);
% testing
%%
clear dt;
dt=net(test_data_input);
 %dt=round(dt);

 %%
count=0;
for i=1:length(test_data)
    buffer=[];
    for j=1:2:22
        if test_data_output(j,i)>=test_data_output(j+1,i)
            k=1;
        else k=0;
        end
            buffer=[buffer k];
        
    
    end
    
   d1=bi2de(buffer,'left-msb');
    buffer=[];
    for j=1:2:22
        if dt(j,i)>=dt(j+1,i)
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
%%
clear test_data
clear test_data_input
clear test_data_output
test_data=data(month_change(4)+1:month_change(4+1),:);
%test_data=data(s1,:);
test_data_input=(test_data(:,[2:1:10,12,13]))';
test_data_output_temp=(test_data(:,15))';
for i=1:length(test_data_output_temp)
    
    test_data_output(test_data_output_temp(i),i)=1;
    
end
%%
a=[1 0 1];
b=[a ~a]
%%
clear all
close all
load('06-Mar-2018 13:45:16.mat') % cell id 4

clear net
A=1:length(data); 
frac=0.3;
mon=6;
len=length(data);


%idx=randperm(numel(A));


%s1=A(idx(1:frac*length(data)));
test_data=data(month_change(mon)+1:month_change(mon+1),:);
%test_data=data(s1,:);
%{
for i=1:length(test_data)
    temp=reshape(de2bi(1*test_data(i,[2:1:10,12,13]),24,'left-msb'),1,[]);
    test_data_input1(i,:)=temp;
    %t=bi2de(reshape(temp,4,8),'left-msb');
end
test_data_input=test_data_input1';

%}
test_data_input=(test_data(:,[2:1:10,12,13]))';

%test_data_input=(test_data(:,[2:1:10,12,13]))';
test_data_output_temp=(1*de2bi(test_data(:,15),11,'left-msb'));
test_data_output_temp2=[test_data_output_temp ~test_data_output_temp]';
k=1;
for i=1:11
    test_data_output([k,k+1],:)=test_data_output_temp2([i,11+i],:);
    k=k+2;
    
    
end

%
%subSet2=A(idx(frac*length(data)+1:end));
%s2=sort(subSet2);
%s2=subSet2;
training_data1=data(month_change(mon-3)+1:month_change(mon),:);
%training_data=data(s2,:);
[val, id]=sort(training_data1(:,15));
training_data=training_data1(id,:);
%{
for i=1:length(training_data)
    temp=reshape(de2bi(1*training_data(i,[2:1:10,12,13]),24,'left-msb'),1,[]);
    training_data_input1(i,:)=temp;
  %  t=bi2de(reshape(temp,4,8),'left-msb');
end
training_data_input=training_data_input1';
%}
training_data_input=(training_data(:,[2:1:10,12,13]))';
training_data_target_temp=(1*de2bi(training_data(:,15),11,'left-msb'));
training_data_target_temp2=[training_data_target_temp ~training_data_target_temp]';
k=1;
for i=1:11
    training_data_target([k,k+1],:)=training_data_target_temp2([i,11+i],:);
    k=k+2;
    
    
end
