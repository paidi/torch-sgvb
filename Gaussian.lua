-- Stochastic layer of Gaussian variables
-- Takes a mean and a stddev vector as input
-- Output are random gaussian samples using the reparameterisation trick

require 'nn'

local Gaussian, parent = torch.class('nn.Gaussian', 'nn.Module')

function Gaussian:__init()
   parent.__init(self)
   self.gradInput = {}
end

function Gaussian:updateOutput(input)   
   if not self.output or self.output:size() ~= input[1]:size() then
      self.output = input[1]:clone()
      self.gradInput[1] = torch.zeros(input[1]:size())
      self.gradInput[2] = torch.zeros(input[2]:size())
   else
      self.output:copy(input[1])
   end
   
   -- Sample epsilon
   self.eps = torch.randn(input[1]:size())

   -- mean + eps \cdot stddev
   self.output:addcmul(self.eps, input[2])

   return self.output
end

function Gaussian:updateGradInput(input, gradOutput)
   self.gradInput[1]:copy(gradOutput)
   self.gradInput[2]:copy(gradOutput)
   self.gradInput[2]:cmul(self.eps)
   return self.gradInput
end
   
