require 'nn'
require 'nngraph'
require 'Gaussian'
require 'GaussianKLCriterion'
require 'optim'

function createVariationalAE(inputSize, hiddenSize, zSize)
   -- Input to hidden
   local hIn = nn.Sequential()
   hIn:add(nn.Linear(inputSize,hiddenSize))
   hIn:add(nn.Tanh())

   -- Hidden to Z's parameters
   local zParam = nn.ConcatTable()
   zParam:add(nn.Linear(hiddenSize,zSize))
   zParam:add(nn.Linear(hiddenSize,zSize))

   local encoder = nn.Sequential()
   encoder:add(hIn)
   encoder:add(zParam)
   
   -- decoder includes random sampling
   local decoder = nn.Sequential()
   decoder:add(nn.Gaussian())
   decoder:add(nn.Linear(zSize, hiddenSize))
   decoder:add(nn.Tanh())
   decoder:add(nn.Linear(hiddenSize,inputSize))
   decoder:add(nn.Sigmoid())
   
   return encoder, decoder
end

function loadMnist(datadir)
   data = {
      train = torch.load(paths.concat(datadir, 
                                      'train_32x32.t7'),
                                      'ascii').data,
      test = torch.load(paths.concat(datadir, 
                                     'test_32x32.t7'), 
                                     'ascii').data
   }
   
   --Rescale to 0..1 and resize into a form amenable to batching
   data.train:div(255):resize(60000,1024)
   data.test:div(255):resize(10000,1024)
   
   return data
end

local data = loadMnist('data', false)
local encoder,decoder = createVariationalAE(data.train:size(2), 400, 10)

-- Use binary cross-entropy for reconstruction
local reconstructionCriterion = nn.BCECriterion()
local klCriterion = nn.GaussianKLCriterion()

local nEpochs = 10 -- 10
local batchSize = 100
local inputSize = data.train:size(2)

local config = {
    learningRate = 1E-3,
}
local state = {}

-- Add both networks together so we can get one big parameter vector
local va = nn.Sequential():add(encoder):add(decoder)
local params, gparams = va:getParameters()

for epoch=1,nEpochs do
   local N = data.train:size(1)
   local shuffle = torch.randperm(N)
   local x = torch.Tensor(N,inputSize)
   local lb = 0.0

   for i=0,N-1,batchSize do
      xlua.progress(i, N-1)
      for j=1,batchSize do
         x[j]:copy(data.train[shuffle[j+i]])
      end
      
      local opfunc = function(theta) 
         if theta ~= params then
            params:copy(theta)
         end
         encoder:zeroGradParameters()
         decoder:zeroGradParameters()
         
         -- Compute Z's parameters
         local zpar = encoder:forward(x)

         print('Compute Z')

         -- Contribution of KL divergence term
         local kldiv = klCriterion:forward(zpar)
         local dkl_dzpar = klCriterion:backward(zpar)
         
         local xhat = decoder:forward(zpar)

         -- Contribution of reconstruction error
         local err = reconstructionCriterion:forward(xhat, x)
         local derr_dxhat = reconstructionCriterion:backward(xhat, x)
         
         local nlb = kldiv + err
         local dlb_dzpar = dkl_dzpar + decoder:backward(zpar, derr_dxhat)
         
         encoder:backward(x, dl_dzpar)
         
         return nlb, gparams
      end
      
      _, nlb = optim.adagrad(opfunc, params, config, state)
      
      lb = lb - nlb            
   end
   
   print(epoch .. ' : ' .. lb/N)
end
