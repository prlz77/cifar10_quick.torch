--------------------------------------------------------------------------------
--  DESCRIPTION:  Torch adaptation of caffe (& cuda-convnet) cifar10_quick
--                (https://github.com/BVLC/caffe/tree/master/examples/cifar10)
--       AUTHOR:  prlz77
--  INSTITUTION:  ISELAB@CVC-UAB
--      VERSION:  1.0
--      CREATED:  26/07/2016
--------------------------------------------------------------------------------

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'

local nninit = require('nninit')

local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -r,--learningRate  (default 0.001)       learning rate, for SGD only
   -b,--batchSize     (default 100)         batch size
   -m,--momentum      (default 0.9)         momentum, for SGD only
   --maxEpoch         (default 10)          max epoch
   --trainPath        (default ./cifar10-train.t7)
   --testPath         (default ./cifar10-test.t7)
]]

-- model definition
model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3, 32, 5,5, 1,1, 2,2):init('weight',nninit.normal, 0, 0.0001):init('bias',nninit.constant, 0)) --> 32*32
model:add(cudnn.SpatialMaxPooling(3,3, 2,2):ceil()) --> 32*16
model:add(cudnn.ReLU())
model:add(cudnn.SpatialConvolution(32, 32, 5,5, 1,1, 2,2):init('weight',nninit.normal, 0, 0.01):init('bias',nninit.constant, 0)) -->  32 * 16
model:add(cudnn.ReLU())
model:add(cudnn.SpatialAveragePooling(3,3, 2,2))  --> 32*7
model:add(cudnn.SpatialConvolution(32, 64, 5,5, 1,1, 2,2):init('weight',nninit.normal, 0, 0.01):init('bias',nninit.constant, 0)) --> 64*7
model:add(cudnn.ReLU())
model:add(cudnn.SpatialAveragePooling(3,3, 2,2)) --> 64*3
model:add(nn.Reshape(64*3*3))
model:add(nn.Linear(64*3*3,64):init('weight',nninit.normal, 0, 0.1):init('bias',nninit.constant, 0))
model:add(cudnn.ReLU())
model:add(nn.Linear(64,10):init('weight',nninit.normal, 0, 0.1):init('bias',nninit.constant, 0))
model:cuda()

-- set learning rates as in Caffe
local params_lr_m = model:clone()
for i = 1,#params_lr_m.modules do
	if params_lr_m.modules[i].bias ~= Nil then
		params_lr_m.modules[i].bias:fill(2)
		params_lr_m.modules[i].weight:fill(1)
	end
end
params_lr = params_lr_m:getParameters()

-- preparate loggers
os.execute('mkdir -p ' .. sys.dirname(opt.save))
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
trainLogger:setNames{'epoch','train_acc'}
testLogger:setNames{'epoch','test_acc'}

-- load dataset
print('Shuffling and preprocessing...')
local train = torch.load(opt.trainPath)
local indices = torch.randperm(train.data:size(1))
train.data = train.data:index(1,indices:long())
train.label = train.label:index(1,indices:long()):add(1)
train.data = train.data:float()
local test = torch.load(opt.testPath)
local indices = torch.randperm(test.data:size(1))
test.data = test.data:index(1,indices:long())
test.label = test.label:index(1,indices:long()):add(1)
test.data = test.data:float()

-- mean subtraction
local mean_v = torch.FloatTensor(3,32,32)
for c = 1,3 do
	mean_v[{c,{},{}}] = torch.mean(train.data[{{},c,{},{}}], 1)
end

for i = 1,train.data:size(1) do
	train.data[{i,{},{},{}}]:add(-mean_v)
end
for i = 1,test.data:size(1) do
	test.data[{i,{},{},{}}]:add(-mean_v)
end
print('done.')

local criterion = nn.CrossEntropyCriterion():cuda()
-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

local data = torch.CudaTensor(opt.batchSize, 3, 32, 32)
local labels = torch.CudaTensor(opt.batchSize)

-- same schedule as caffe version
local schedule = {[1] = 0.001, [9] = 0.0001}
local epoch = 1
local currentSample
local train_acc = 0

-- forward / backward
function f(x)
	gradParameters:zero()
	if x ~= parameters then
		parameters:copy(x)
	end
	data[{}] = train.data[{{currentSample,(currentSample + opt.batchSize - 1)},{},{},{}}]
	labels[{}] = train.label[{{currentSample,(currentSample + opt.batchSize - 1)}}]
	local output = model:forward(data)
	max, ind = torch.max(output, 2)
	train_acc = train_acc + ind:eq(labels):sum() / opt.batchSize
	local err = criterion:forward(output, labels)
	loss = loss + err
	local df_do = criterion:backward(output, labels)
	model:backward(data, df_do)
	return err, gradParameters
end

-- epoch loop
function ftrain()
	model:training()
	loss = 0
	train_acc = 0
  -- keep same schedule as caffe
	opt.learningRate = schedule[epoch] or opt.learningRate
	sgdState = sgdState or {
		learningRate = opt.learningRate,
		learningRateDecay = 0,
		learningRates = params_lr,
		dampening = 0,
		momentum = opt.momentum,
		weightDecay = opt.wDecay
	}
	sgdState.learningRate = opt.learningRate
	print('lr: '..sgdState.learningRate)
	print('momentum: '..sgdState.momentum)
	print('wDecay: '..sgdState.weightDecay)
	for sample = 1,train.data:size(1),opt.batchSize do
		currentSample = sample
		optim.sgd(f, parameters, sgdState) -- SGD step
		xlua.progress(sample + opt.batchSize - 1, train.data:size(1))
	end
  -- output accuracy / loss
	print('train_loss: '..(loss / (train.data:size(1) / opt.batchSize)))
	print('train_acc: '..(100 * train_acc / (train.data:size(1) / opt.batchSize)))
	trainLogger:add{epoch, 100 * train_acc / (train.data:size(1) / opt.batchSize)}
end

-- test epoch
function ftest()
	model:evaluate()
	local test_loss = 0
	local test_acc = 0
	local counter = 0
	for sample = 1,test.data:size(1),opt.batchSize do
		currentSample = sample
		data[{}] = test.data[{{currentSample,(currentSample + opt.batchSize - 1)},{},{},{}}]
		labels[{}] = test.label[{{currentSample,(currentSample + opt.batchSize - 1)}}]
		local output = model:forward(data)
		test_loss = test_loss + criterion:forward(output, labels)
		maxs, indices = torch.max(output, 2)
		test_acc = test_acc + (indices:eq(labels):sum()/ opt.batchSize)
		xlua.progress(sample + opt.batchSize - 1, test.data:size(1))
		counter  = counter + 1
	end
  -- output accuracy
	print('test_acc: '..(100 * test_acc / counter))
	testLogger:add{epoch, 100 * test_acc / counter}
end

-- main loop
while epoch <= opt.maxEpoch do
	print('epoch '..epoch)
	ftrain()
	ftest()
	epoch = epoch + 1
end

print('done')
