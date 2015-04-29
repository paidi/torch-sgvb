-- A criterion for computing the KL-divergence between a Gaussian and N(0,I)
-- We assume the Gaussian is presented as a pair of vectors, one for the mean
-- and one for the standard deviation
-- See Kingma and Welling (http://arxiv.org/pdf/1312.6114v10.pdf) Appendix B

require 'nn'
require 'torch'

local GaussianKLCriterion, parent = torch.class('nn.GaussianKLCriterion', 
                                                'nn.Criterion')
function GaussianKLCriterion:__init()
   parent.__init(self)
   self.gradInput = {}
end

function GaussianKLCriterion:updateOutput(input)
   local mean = input[1]
   local variance = torch.pow(input[2],2)
   
   if not self.divergences then
      self.divergences = torch.log(variance):add(1)
   else
      self.divergences:copy(variance)
      self.divergences:log():add(1)      
   end

   self.divergences:add(-1, torch.pow(mean, 2))
   self.divergences:add(-1, variance)
   
   return self.divergences:sum() * 0.5
end

function GaussianKLCriterion:updateGradInput(input, _)
   local mean = input[1]
   local stddev = input[2]
   
   if #self.gradInput == 0 or self.gradInput[1]:size() ~= mean:size() then
      self.gradInput[1] = mean:clone():mul(-1)
      self.gradInput[2] = stddev:clone():mul(-1)
   else
      self.gradInput[1]:copy(mean):mul(-1)
      self.gradInput[2]:copy(stddev):mul(-1)
   end
   
   -- And the log term for the standard deviation
   self.gradInput[2]:add(torch.ones(stddev:size()):cdiv(input[2]))
   
   return self.gradInput
end
