local _ = require 'moses'

dofile 'GaussianKLCriterion.lua'
dofile 'Gaussian.lua'

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1e-4

local function criterionJacobianTest1D(cri, input)
   local eps = 1e-6
   local _ = cri:forward(input)
   local dfdx = cri:updateGradInput(input)

   -- for each input perturbation, do central difference
   local centraldiff_dfdx = torch.Tensor():resizeAs(input):zero()
   local input_s = input:storage()
   local centraldiff_dfdx_s = centraldiff_dfdx:storage()
   for i=1,input:nElement() do
      -- f(xi + h)
      input_s[i] = input_s[i] + eps
      local fx1 = cri:forward(input)
      -- f(xi - h)
      input_s[i] = input_s[i] - 2*eps
      local fx2 = cri:forward(input)
      -- f'(xi) = (f(xi + h) - f(xi - h)) / 2h
      local cdfx = (fx1 - fx2) / (2*eps)
      -- store f' in appropriate place
      centraldiff_dfdx_s[i] = cdfx
      -- reset input[i]
      input_s[i] = input_s[i] + eps
   end

   -- compare centraldiff_dfdx with :updateGradInput()
   local err = (centraldiff_dfdx - dfdx):abs():max()
   mytester:assertlt(err, precision, 'error in difference between central difference ' .. centraldiff_dfdx:sum() .. ' and :updateGradInput ' .. dfdx:sum())
end

local sbptest = {}

function wrap(m, inputSizes)
   local function splitInput(input, inputSizes)
      local inputs = {}
      local from = 1
      for i=1,#inputSizes do         
         local to = from + inputSizes[i] - 1
         inputs[i] = input[{{from,to}}]
         from = to+1
      end
      return inputs
   end

   local function join(inputs, size)
      local result = torch.Tensor(size)
      local from = 1
      for i=1,#inputs do         
         local to = from + inputs[i]:size(1) - 1
         result[{{from,to}}]:copy(inputs[i])
         from = to+1
      end
      return result
   end

   local wrapper = {}
   
   function wrapper:forward(input)
      torch.manualSeed(1)
      self.output = m:forward(splitInput(input, inputSizes))
      return self.output
   end

   function wrapper:updateGradInput(input, gradOutput)
      return join(m:updateGradInput(splitInput(input, inputSizes), gradOutput), input:size())
   end

   function wrapper:accGradParameters(input, gradOutput)      
   end
   
   function wrapper:zeroGradParameters()
   end

   return wrapper
end

function sbptest.GaussianKLCriterion()
   local N = 20
   local input = torch.rand(2*N)
   local cri = nn.GaussianKLCriterion()
   
   criterionJacobianTest1D(wrap(cri, {N,N}), input)
end

function sbptest.Gaussian()
   local N = 20
   local input = torch.rand(N*2)
   local target = torch.rand(N)

   local err = jac.testJacobian(wrap(nn.Gaussian(), {N,N}), input)
   mytester:assertlt(err,1e-3, 'error on state ')
end

mytester:add(sbptest)

print 'Running tests'
jac = nn.Jacobian
sjac = nn.SparseJacobian
mytester:run()
